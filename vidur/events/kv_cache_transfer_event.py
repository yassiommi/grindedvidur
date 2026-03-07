from typing import List

from vidur.entities.batch import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class KvCacheTransferEvent(BaseEvent):
    """Event modeling inter-GPU KV cache transfer between prefill and decode stages.

    This event simulates the I/O overhead of transferring KV cache data from the
    prefill GPU to the decode GPU(s) in a disaggregated prefill-decode setup.
    """

    def __init__(
        self,
        time: float,
        replica_id: int,
        stage_id: int,
        batch: Batch,
        pcie_bandwidth_gbps: float = 50.0,
        enable_prefetch_overlap: bool = True,
    ):
        """Initialize KvCacheTransferEvent.

        Args:
            time: Event time (when prefill stage ends)
            replica_id: Target replica ID for decode stage
            stage_id: Next stage ID (decode stage, typically 1)
            batch: Batch with requests that completed prefill
            pcie_bandwidth_gbps: PCIe bandwidth in GB/s
            enable_prefetch_overlap: Whether to model prefetch overlap with decode compute
        """
        super().__init__(time, EventType.KV_CACHE_TRANSFER)
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._batch = batch
        self._pcie_bandwidth_gbps = pcie_bandwidth_gbps
        self._enable_prefetch_overlap = enable_prefetch_overlap

    def _compute_kv_transfer_time(self) -> float:
        """Compute KV cache transfer time in milliseconds.

        Calculates the total KV cache size from all requests in the batch and
        divides by PCIe bandwidth to get transfer latency.

        Returns:
            Transfer time in milliseconds.
        """
        total_kv_bytes = 0.0
        for request in self._batch.requests:
            if request.kv_cache_metadata is not None:
                total_kv_bytes += request.kv_cache_metadata.cache_bytes

        if total_kv_bytes == 0:
            return 0.0

        # Transfer time = bytes / bandwidth
        bandwidth_bytes_per_s = self._pcie_bandwidth_gbps * 1e9
        transfer_time_s = total_kv_bytes / bandwidth_bytes_per_s
        transfer_time_ms = transfer_time_s * 1000.0

        return transfer_time_ms

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        """Handle KV cache transfer event.

        Computes transfer time, records metrics, and generates the next batch
        stage arrival event for the decode stage.

        Returns:
            List of next events to be scheduled.
        """
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        # Compute KV transfer time
        transfer_time_ms = self._compute_kv_transfer_time()

        # Record KV transfer metrics for each request
        for request in self._batch.requests:
            if request.kv_cache_metadata is not None:
                request.on_kv_transfer_complete(
                    time=self.time + transfer_time_ms / 1000.0,
                    transfer_time_ms=transfer_time_ms,
                    prefetch_overlap_ms=0.0,  # TODO: compute actual overlap
                )

        # Record metrics
        metrics_store.on_kv_cache_transfer(
            time=self.time,
            replica_id=self._replica_id,
            batch_id=self._batch.id,
            transfer_time_ms=transfer_time_ms,
            num_requests=len(self._batch.requests),
        )

        logger.debug(
            f"KV Cache Transfer Event: batch {self._batch.id}, "
            f"transfer_time={transfer_time_ms:.3f}ms, "
            f"num_requests={len(self._batch.requests)}"
        )

        # Generate next event: batch arrives at decode stage
        next_events = [
            BatchStageArrivalEvent(
                self.time + transfer_time_ms / 1000.0,
                self._replica_id,
                self._stage_id,
                self._batch,
            )
        ]

        return next_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id,
            "pcie_bandwidth_gbps": self._pcie_bandwidth_gbps,
        }
