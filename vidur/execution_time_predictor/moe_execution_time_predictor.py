"""MoE-aware execution time predictor using InferSim's FLOPs-based approach.

For MoE models (DeepSeek-V3, Mixtral, Qwen-MoE), this predictor computes:
- Expert routing overhead
- Grouped GEMM for routed experts (compute-bound vs I/O-bound)
- Shared expert computation
- Expert parallelism communication (dispatch/combine)
- MLA (Multi-head Latent Attention) projection costs

Based on InferSim's simulation methodology:
  - FLOPs = 2 * M * N * K for each GEMM
  - Latency = GFLOPs / (GPU_TFLOPS * 1024 * MFU)
  - Expert loading time = expert_params / memory_bandwidth
  - MoE layer time = max(compute_time, load_time) + shared_expert_time
"""

import math

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Batch
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from vidur.logger import init_logger

logger = init_logger(__name__)


class MoEExecutionTimePredictor(SklearnExecutionTimePredictor):
    """Extends sklearn predictor with MoE-specific timing from InferSim."""

    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

        self._is_moe = getattr(self._model_config, 'is_moe', False)
        self._expert_parallel_size = getattr(replica_config, 'expert_parallel_size', 1)

        if self._is_moe:
            self._setup_moe_params()

    def _setup_moe_params(self):
        """Configure MoE-specific parameters from model config."""
        mc = self._model_config
        self._num_routed_experts = mc.num_routed_experts
        self._num_experts_per_tok = mc.num_experts_per_tok
        self._num_shared_experts = mc.num_shared_experts
        self._expert_intermediate_size = mc.expert_intermediate_size
        self._hidden_size = mc.embedding_dim

        # GPU specs for FLOPs-based estimation
        dc = self._replica_config.device_config
        self._gpu_fp16_tflops = dc.fp16_flops / 1e12  # Convert to TFLOPS
        self._gpu_memory_gb = dc.total_memory_gb
        # Estimate memory bandwidth from device type
        self._gpu_mem_bw_gbs = self._estimate_mem_bandwidth()

        # Expert params size (3 matrices: gate, up, down for gated MLP)
        bytes_per_param = 2  # FP16
        self._expert_params_bytes = (
            3 * self._hidden_size * self._expert_intermediate_size * bytes_per_param
        )
        # Number of local experts per GPU
        self._local_experts = max(1, self._num_routed_experts // self._expert_parallel_size)

        # Default MFU for MoE GEMM operations (from InferSim benchmarks)
        self._moe_gemm_mfu = 0.15  # Grouped GEMM typically achieves lower MFU

        logger.info(
            f"MoE config: {self._num_routed_experts} experts, "
            f"{self._num_experts_per_tok} per token, "
            f"{self._num_shared_experts} shared, "
            f"EP={self._expert_parallel_size}"
        )

    def _estimate_mem_bandwidth(self) -> float:
        """Estimate GPU memory bandwidth in GB/s from device config."""
        device_type = self._replica_config.device.lower()
        bandwidth_map = {
            "a100": 2039,
            "h100": 3350,
            "h200": 4800,
            "h800": 2744,
            "a40": 696,
        }
        return bandwidth_map.get(device_type, 2039) * 0.8  # 80% efficiency

    @staticmethod
    def _gemm_flops(m: int, k: int, n: int) -> float:
        """FLOPs for a single GEMM: 2*M*N*K (following InferSim convention)."""
        return 2.0 * m * n * k

    def _get_moe_routing_time(self, batch: Batch) -> float:
        """Router/gating network time (small GEMM: hidden -> num_experts)."""
        if not self._is_moe:
            return 0.0

        num_tokens = sum(batch.num_tokens)
        # Router: hidden_size -> num_routed_experts
        router_flops = self._gemm_flops(num_tokens, self._hidden_size, self._num_routed_experts)
        router_gflops = router_flops / 1e9
        # Small GEMM has low MFU
        mfu = 0.05
        latency_s = router_gflops / (self._gpu_fp16_tflops * 1024 * mfu)
        return latency_s * 1e3  # ms

    def _get_moe_expert_compute_time(self, batch: Batch) -> float:
        """Routed expert compute time using InferSim's approach.

        FLOPs = 3 * hidden_size * intermediate_size * num_experts_per_tok * batch_size
        (3 for gate_proj, up_proj, down_proj in gated MLP)
        Latency = GFLOPs / (GPU_TFLOPS * 1024 * MFU)
        """
        if not self._is_moe:
            return 0.0

        num_tokens = sum(batch.num_tokens)
        routed_flops = (
            3.0 * self._gemm_flops(1, self._hidden_size, self._expert_intermediate_size)
            * num_tokens * self._num_experts_per_tok
        )
        routed_gflops = routed_flops / 1e9
        latency_s = routed_gflops / (self._gpu_fp16_tflops * 1024 * self._moe_gemm_mfu)

        # Shared expert computation
        shared_time = 0.0
        if self._num_shared_experts > 0:
            # Shared experts use dense GEMM - higher MFU
            shared_flops = (
                3.0 * self._gemm_flops(
                    num_tokens,
                    self._hidden_size,
                    self._expert_intermediate_size * self._num_shared_experts,
                )
            )
            shared_gflops = shared_flops / 1e9
            shared_mfu = 0.3  # Dense GEMM achieves higher MFU
            shared_time = shared_gflops / (self._gpu_fp16_tflops * 1024 * shared_mfu)

        return (latency_s + shared_time) * 1e3  # ms

    def _get_moe_expert_load_time(self, batch: Batch) -> float:
        """Expert weight loading time from HBM (I/O bound).

        Following InferSim: load_time = expert_params * local_experts / mem_bandwidth
        """
        if not self._is_moe:
            return 0.0

        total_expert_bytes = self._expert_params_bytes * self._local_experts
        load_time_s = total_expert_bytes / (self._gpu_mem_bw_gbs * 1024 * 1024 * 1024)
        return load_time_s * 1e3  # ms

    def _get_expert_parallel_comm_time(self, batch: Batch) -> float:
        """Expert parallelism dispatch/combine communication time.

        Following InferSim's Comm model for DeepEP:
        - Dispatch: send tokens to expert-owning GPUs
        - Combine: gather results back
        """
        if not self._is_moe or self._expert_parallel_size <= 1:
            return 0.0

        num_tokens = sum(batch.num_tokens)
        nc = self._replica_config.node_config

        # Each token needs to be sent to num_experts_per_tok experts
        # across expert_parallel_size GPUs
        bytes_per_element = 2  # FP16
        # Dispatch: tokens * hidden_size * num_experts_per_tok
        dispatch_bytes = (
            num_tokens * self._hidden_size * self._num_experts_per_tok * bytes_per_element
        )
        # Combine: same volume back
        combine_bytes = dispatch_bytes

        # Use intra-node bandwidth (NVLink) as default
        intra_bw = nc.intra_node_bw_gbps if hasattr(nc, 'intra_node_bw_gbps') else 300  # GB/s
        total_bytes = dispatch_bytes + combine_bytes
        comm_time_s = total_bytes / (intra_bw * 1e9)
        return comm_time_s * 1e3  # ms
