"""Per-layer execution time breakdown for compute, I/O, and communication.

Each layer in a pipeline stage gets its own LayerExecutionTime, enabling
Gantt-style visualization and GPU-initiated KV cache prefetching simulation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerExecutionTime:
    """Timing breakdown for a single transformer layer."""

    layer_index: int  # Global layer index within the model

    # Compute times (ms)
    attention_compute_time: float = 0.0  # pre_proj + rope + prefill/decode + post_proj + kv_save + norm
    mlp_compute_time: float = 0.0  # up_proj + act + down_proj + norm

    # I/O times (ms) - memory bandwidth bound operations
    kv_cache_load_time: float = 0.0  # Loading KV cache from HBM
    weight_load_time: float = 0.0  # Loading expert weights (for MoE layers)

    # Communication times (ms)
    tensor_parallel_comm_time: float = 0.0  # All-reduce within TP group
    expert_parallel_comm_time: float = 0.0  # Dispatch/combine for EP (MoE)

    # Prefetch overlap (ms) - time saved by overlapping I/O with prev layer compute
    prefetch_overlap_savings: float = 0.0

    # MoE-specific
    is_moe_layer: bool = False
    num_active_experts: int = 0
    routing_time: float = 0.0

    # Timestamps (filled during simulation, in seconds)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

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
        """Total layer execution time including overlap benefits (ms)."""
        return self.compute_time + self.effective_io_time + self.comm_time

    @property
    def total_time_no_overlap(self) -> float:
        """Total layer time without prefetch overlap (ms)."""
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
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
