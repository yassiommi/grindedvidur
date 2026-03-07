from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class PddReplicaScheduler(BaseReplicaScheduler):
    """Scheduler implementing Prefill-Decode Disaggregation (PDD).

    This scheduler segregates requests into separate prefill and decode phases,
    enabling them to be processed in distinct batches (potentially on separate GPUs).
    KV cache is transferred from prefill to decode phase with I/O overhead modeling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # PDD-specific state
        self._prefill_request_queue: List[Request] = []
        self._decode_preempted_queue: List[Request] = []
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages

    def on_schedule(self) -> List[Batch]:
        """Schedule requests from global queue into prefill/decode batches.

        Returns:
            List of batches to execute (prefill or decode batches)
        """
        # First, route new requests to prefill queue
        while self._request_queue:
            request = self._request_queue.pop(0)
            self._prefill_request_queue.append(request)

        # Create batches from appropriate queue
        batches = []

        # Try to create prefill batches first
        prefill_batch = self._get_next_prefill_batch()
        if prefill_batch:
            batches.append(prefill_batch)

        # Then try to create decode batches
        decode_batch = self._get_next_decode_batch()
        if decode_batch:
            batches.append(decode_batch)

        return batches

    def _can_allocate_request(self, request: Request) -> bool:
        """Check if memory is available for a new request."""
        if request.id not in self._allocation_map:
            # New request: check for prefill token allocation
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # Existing request: need space for decode token
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        """Allocate memory blocks for a request."""
        if request.id not in self._allocation_map:
            # New request: allocate for prefill
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        # Existing request: allocate for one more decode token if needed
        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        if num_tokens_required == 0:
            return

        if num_tokens_required == 1:
            self.allocate(request.id, 1)

    def _get_next_prefill_batch(self) -> Batch:
        """Create a prefill-only batch from requests needing prefill."""
        requests = []
        num_tokens = []
        num_batch_tokens = 0

        # Process requests that still need prefill
        skipped = []
        while self._prefill_request_queue:
            if len(requests) >= self._max_micro_batch_size:
                break

            request = self._prefill_request_queue.pop(0)

            # Only include requests that haven't completed prefill
            if request.is_prefill_complete:
                self._decode_preempted_queue.append(request)
                continue

            if not self._can_allocate_request(request):
                skipped.append(request)
                continue

            self._allocate_request(request)

            # Compute how many prefill tokens to process
            next_num_tokens = min(
                request.num_prefill_tokens - request.num_processed_tokens,
                self._config.max_tokens_in_batch - num_batch_tokens,
            )
            next_num_tokens = max(0, next_num_tokens)

            if next_num_tokens == 0:
                skipped.append(request)
                continue

            num_batch_tokens += next_num_tokens
            requests.append(request)
            num_tokens.append(next_num_tokens)

        # Re-add skipped requests to maintain fairness
        self._prefill_request_queue = skipped + self._prefill_request_queue

        if not requests:
            return None

        batch = Batch(self._replica_id, requests, num_tokens)

        # Compute KV cache metadata when prefill batch is created
        # (This will be used during transfer)
        for request in requests:
            if not request.is_prefill_complete:
                request.compute_kv_cache_metadata(
                    bytes_per_token=self._config.kv_cache_bytes_per_token,
                    source_gpu=0,
                    dest_gpu=1,
                )

        return batch

    def _get_next_decode_batch(self) -> Batch:
        """Create a decode-only batch from requests with prefill completed."""
        requests = []
        num_tokens = []

        # Process requests that have completed prefill
        skipped = []
        while self._decode_preempted_queue:
            if len(requests) >= self._max_micro_batch_size:
                break

            request = self._decode_preempted_queue.pop(0)

            # Only include requests that have completed prefill
            if not request.is_prefill_complete:
                skipped.append(request)
                continue

            if request.completed:
                self.free(request.id)
                continue

            if not self._can_allocate_request(request):
                skipped.append(request)
                continue

            self._allocate_request(request)

            # Decode phase processes one token at a time
            requests.append(request)
            num_tokens.append(1)

        # Re-add skipped requests
        self._decode_preempted_queue = skipped + self._decode_preempted_queue

        if not requests:
            return None

        return Batch(self._replica_id, requests, num_tokens)

    def _get_next_batch(self) -> Batch:
        """Get next batch (overrides base class for compatibility).

        For PDD, this tries to get either prefill or decode batch.
        """
        batches = self.on_schedule(0)
        return batches[0] if batches else None

    def on_batch_end(self, batch: Batch) -> None:
        """Handle batch completion.

        Args:
            batch: Completed batch
        """
        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            elif request.is_prefill_complete:
                # Move to decode queue for next iteration
                if request not in self._decode_preempted_queue:
                    self._decode_preempted_queue.append(request)
            else:
                # Still in prefill phase, re-add to prefill queue
                if request not in self._prefill_request_queue:
                    self._prefill_request_queue.append(request)
