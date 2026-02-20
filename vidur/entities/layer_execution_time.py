"""Per-layer execution time breakdown for compute, I/O, and communication.

Each layer in a pipeline stage gets its own LayerExecutionTime, enabling
Gantt-style visualization and GPU-initiated KV cache prefetching simulation.

The scheduling model reflects real GPU hardware with three independent units:
  - SM (Streaming Multiprocessors): Runs compute (GEMMs, attention core)
  - DMA engine: Transfers KV cache over PCIe (can overlap with SM compute)
  - NVLink/NIC: Runs all-reduce communication (serialized after SM compute)

Within a layer, the timeline is:
  SM:   [===Attention===][===MLP===]
  DMA:  [---KV prefetch for NEXT layer---]   (overlaps with SM)
  NCCL:                   [===All-reduce===]  (after MLP, before next layer)

A layer's wall-clock time = max(compute + comm, dma_remaining) where
dma_remaining is any prefetch I/O not yet finished by the time compute ends.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerExecutionTime:
    """Timing breakdown for a single transformer layer."""

    layer_index: int  # Global layer index within the model

    # Compute times (ms) — runs on SM
    attention_compute_time: float = 0.0  # pre_proj + rope + prefill/decode + post_proj + kv_save + norm
    mlp_compute_time: float = 0.0  # up_proj + act + down_proj + norm (or MoE expert GEMM)

    # I/O times (ms) — runs on DMA engine, can overlap with SM compute
    kv_cache_load_time: float = 0.0  # Loading KV cache from host via PCIe
    weight_load_time: float = 0.0  # Loading expert weights from HBM (MoE)

    # Communication times (ms) — runs on NVLink/NIC, serialized after compute
    tensor_parallel_comm_time: float = 0.0  # All-reduce within TP group
    expert_parallel_comm_time: float = 0.0  # Dispatch/combine for EP (MoE)

    # Prefetch overlap (ms) — time saved by overlapping I/O with compute
    prefetch_overlap_savings: float = 0.0

    # MoE-specific
    is_moe_layer: bool = False
    num_active_experts: int = 0
    routing_time: float = 0.0

    # Scheduled timestamps (filled during simulation, in seconds)
    # These represent the wall-clock start/end of the layer
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Per-stream scheduled intervals (ms offsets from layer start)
    # Filled by the overlap scheduler in ExecutionTime
    compute_start: float = 0.0   # SM stream start
    compute_end: float = 0.0     # SM stream end (compute + comm)
    io_start: float = 0.0        # DMA stream start
    io_end: float = 0.0          # DMA stream end
    comm_start: float = 0.0      # NCCL stream start
    comm_end: float = 0.0        # NCCL stream end

    @property
    def compute_time(self) -> float:
        """Total compute time for this layer (ms)."""
        return self.attention_compute_time + self.mlp_compute_time + self.routing_time

    @property
    def io_time(self) -> float:
        """Total I/O time for this layer (ms)."""
        return self.kv_cache_load_time + self.weight_load_time

    @property
    def comm_time(self) -> float:
        """Total communication time for this layer (ms)."""
        return self.tensor_parallel_comm_time + self.expert_parallel_comm_time

    @property
    def effective_io_time(self) -> float:
        """I/O time after subtracting prefetch overlap (ms)."""
        return max(0.0, self.io_time - self.prefetch_overlap_savings)

    @property
    def total_time(self) -> float:
        """Total layer wall-clock time including overlap benefits (ms).

        This is the actual scheduled duration: max of all stream end times
        minus the layer start offset, or the sum-based estimate if not yet
        scheduled.
        """
        if self.compute_end > 0 or self.io_end > 0 or self.comm_end > 0:
            # Use scheduled stream timings
            return max(self.compute_end, self.io_end, self.comm_end)
        # Fallback: sequential with prefetch deduction
        return self.compute_time + self.effective_io_time + self.comm_time

    @property
    def total_time_no_overlap(self) -> float:
        """Total layer time without any overlap (ms)."""
        return self.compute_time + self.io_time + self.comm_time

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "attention_compute_time": self.attention_compute_time,
            "mlp_compute_time": self.mlp_compute_time,
            "kv_cache_load_time": self.kv_cache_load_time,
            "weight_load_time": self.weight_load_time,
            "tensor_parallel_comm_time": self.tensor_parallel_comm_time,
            "expert_parallel_comm_time": self.expert_parallel_comm_time,
            "prefetch_overlap_savings": self.prefetch_overlap_savings,
            "is_moe_layer": self.is_moe_layer,
            "num_active_experts": self.num_active_experts,
            "routing_time": self.routing_time,
            "compute_time": self.compute_time,
            "io_time": self.io_time,
            "comm_time": self.comm_time,
            "effective_io_time": self.effective_io_time,
            "total_time": self.total_time,
            # Per-stream intervals (ms offsets from layer start)
            "compute_start": self.compute_start,
            "compute_end": self.compute_end,
            "io_start": self.io_start,
            "io_end": self.io_end,
            "comm_start": self.comm_start,
            "comm_end": self.comm_end,
            # Absolute timestamps (seconds)
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
