from typing import Optional, Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.entities.kv_cache_metadata import KvCacheMetadata
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        num_processed_tokens: int = 0,
        token_ids: Optional[Tuple[int, ...]] = None,
    ):
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_tokens = num_decode_tokens
        self._num_processed_tokens = num_processed_tokens

        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0

        # Token IDs for prefix cache matching (None = prefix caching disabled)
        self._token_ids: Optional[Tuple[int, ...]] = token_ids

        # Prefix cache tracking
        self._prefix_cache_hit_tokens: int = 0  # tokens found in cache
        self._original_prefill_tokens: int = num_prefill_tokens  # before cache reduction

        # KV cache metadata for PDD (Prefill-Decode Disaggregation)
        self._kv_cache_metadata: Optional[KvCacheMetadata] = None
        self._kv_transfer_started_at: float = 0.0
        self._kv_transfer_completed_at: float = 0.0
        self._is_kv_transferred: bool = False

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        return self._scheduling_delay

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self.num_decode_tokens

    @property
    @check_scheduled
    def execution_time(self) -> float:
        return self._execution_time

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self._execution_time / self.num_decode_tokens

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self._model_execution_time / self.num_decode_tokens

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1

    @property
    def token_ids(self) -> Optional[Tuple[int, ...]]:
        return self._token_ids

    @property
    def prefix_cache_hit_tokens(self) -> int:
        return self._prefix_cache_hit_tokens

    @property
    def original_prefill_tokens(self) -> int:
        return self._original_prefill_tokens

    @property
    def prefix_cache_hit_ratio(self) -> float:
        if self._original_prefill_tokens == 0:
            return 0.0
        return self._prefix_cache_hit_tokens / self._original_prefill_tokens

    def apply_prefix_cache_hit(self, num_cached_tokens: int) -> None:
        """Reduce prefill tokens based on prefix cache hit.

        The cached tokens don't need to be re-prefilled, so we reduce the
        effective prefill token count. The original count is preserved for metrics.

        Args:
            num_cached_tokens: Number of prefix tokens found in cache.
        """
        if num_cached_tokens <= 0:
            return
        num_cached_tokens = min(num_cached_tokens, self._num_prefill_tokens)
        self._prefix_cache_hit_tokens = num_cached_tokens
        self._num_processed_tokens = num_cached_tokens

    @property
    def kv_cache_metadata(self) -> Optional[KvCacheMetadata]:
        return self._kv_cache_metadata

    @property
    def is_kv_transferred(self) -> bool:
        return self._is_kv_transferred

    @property
    def kv_transfer_started_at(self) -> float:
        return self._kv_transfer_started_at

    @property
    def kv_transfer_completed_at(self) -> float:
        return self._kv_transfer_completed_at

    def compute_kv_cache_metadata(
        self, bytes_per_token: float, source_gpu: int = 0, dest_gpu: int = 1
    ) -> None:
        """Compute and store KV cache metadata for this request.

        Args:
            bytes_per_token: KV cache bytes per token
            source_gpu: Source GPU ID (prefill GPU)
            dest_gpu: Destination GPU ID (decode GPU)
        """
        self._kv_cache_metadata = KvCacheMetadata(
            request_id=self._id,
            num_tokens=self._num_prefill_tokens,
            bytes_per_token=bytes_per_token,
            source_gpu=source_gpu,
            dest_gpu=dest_gpu,
        )

    def on_kv_transfer_start(self, time: float) -> None:
        """Record KV transfer start time."""
        self._kv_transfer_started_at = time

    def on_kv_transfer_complete(
        self, time: float, transfer_time_ms: float, prefetch_overlap_ms: float = 0.0
    ) -> None:
        """Record KV transfer completion and metrics.

        Args:
            time: Completion timestamp
            transfer_time_ms: Transfer latency in milliseconds
            prefetch_overlap_ms: Optional prefetch overlap with compute
        """
        if self._kv_cache_metadata is None:
            return
        self._kv_transfer_completed_at = time
        self._kv_cache_metadata.transfer_time_ms = transfer_time_ms
        self._kv_cache_metadata.prefetch_overlap_ms = prefetch_overlap_ms
        self._kv_cache_metadata.transferred_at = time
        # Compute transfer throughput in GB/s
        transfer_time_s = transfer_time_ms / 1000.0
        if transfer_time_s > 0:
            self._kv_cache_metadata.transfer_throughput_gbps = (
                self._kv_cache_metadata.cache_bytes / 1e9 / transfer_time_s
            )
        self._is_kv_transferred = True

    def on_batch_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at
        )

        if self._scheduled:
            return

        if self._num_restarts > 0:
            self._scheduled = True
            return

        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True

    def on_batch_end(
        self,
        time: float,
        num_tokens_processed: int,
    ) -> None:
        self._num_processed_tokens += num_tokens_processed
        self._latest_iteration_completed_at = time

        assert self._num_processed_tokens <= self.total_tokens

        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            # we get one decode token when the prefill processing completes
            self._num_processed_tokens += 1

            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                self._prefill_completed_at = time

        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            logger.debug(f"Request {self._id} completed at {self._completed_at}")

    def on_batch_stage_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            self._preempted_time += time - self._latest_stage_completed_at
        self._preempted = False

    def on_batch_stage_end(
        self,
        time: float,
        execution_time: float,
        model_execution_time: float,
    ) -> None:
        self._execution_time += execution_time
        self._model_execution_time += model_execution_time
        self._latest_stage_completed_at = time
        self._preempted = True

    def to_dict(self) -> dict:
        result = {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "scheduled_at": self._scheduled_at,
            "scheduling_delay": self._scheduling_delay,
            "preempted_time": self._preempted_time,
            "completed_at": self._completed_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "num_processed_tokens": self._num_processed_tokens,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "num_restarts": self._num_restarts,
        }
        if self._prefix_cache_hit_tokens > 0:
            result["prefix_cache_hit_tokens"] = self._prefix_cache_hit_tokens
            result["original_prefill_tokens"] = self._original_prefill_tokens
            result["prefix_cache_hit_ratio"] = self.prefix_cache_hit_ratio
        if self._kv_cache_metadata is not None:
            result["kv_cache_bytes"] = self._kv_cache_metadata.cache_bytes
            result["kv_transfer_time_ms"] = self._kv_cache_metadata.transfer_time_ms
            result["kv_transfer_throughput_gbps"] = (
                self._kv_cache_metadata.transfer_throughput_gbps
            )
            result["kv_prefetch_overlap_ms"] = (
                self._kv_cache_metadata.prefetch_overlap_ms
            )
        return result

    def restart(self):
        logger.debug(f"Restarting request {self._id}")

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens

        self._num_processed_tokens = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts += 1
