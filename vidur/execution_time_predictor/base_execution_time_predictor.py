from abc import ABC, abstractmethod
from typing import List, Optional

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, ExecutionTime


# Efficiency factor applied to raw peak bandwidth (PCIe / HBM).
# Real workloads achieve ~80% of peak due to protocol overhead, access patterns, etc.
_BW_EFFICIENCY = 0.8


class BaseExecutionTimePredictor(ABC):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config

        # get configs
        self._replica_scheduler_provider = str(replica_scheduler_config.get_type())
        self._block_size = replica_scheduler_config.block_size
        self._cache_dir = metrics_config.cache_dir
        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )
        self._enable_kv_prefetch = replica_config.enable_kv_prefetch

        # PCIe bandwidth in bytes/s for KV cache I/O modeling.
        # KV cache prefetching transfers data from host memory to GPU over PCIe.
        device_config = replica_config.device_config
        pcie_bw = getattr(device_config, 'pcie_bandwidth_gb_per_s', 0.0)
        self._pcie_bw_bytes_per_s = pcie_bw * _BW_EFFICIENCY * (1024 ** 3)

        # HBM bandwidth in bytes/s (used for expert weight loading in MoE)
        hbm_bw = getattr(device_config, 'memory_bandwidth_gb_per_s', 0.0)
        self._mem_bw_bytes_per_s = hbm_bw * _BW_EFFICIENCY * (1024 ** 3)

        # KV cache bytes per token per layer (for bandwidth-based load time)
        self._kv_bytes_per_token_per_layer = self._compute_kv_bytes_per_token_per_layer()

    def _compute_kv_bytes_per_token_per_layer(self) -> float:
        """KV cache size in bytes for one token in one layer.

        Following InferSim's kvcache/kvcache.py:
        - MHA/GQA: 2 * num_kv_heads * head_dim * bytes_per_element
        - MLA: (kv_lora_rank + qk_rope_head_dim) * bytes_per_element
        - HYBRID_CSA_HCA: averaged compression factor across interleaved CSA/HCA layers
        """
        mc = self._model_config
        bytes_per_element = 2  # FP16/BF16

        attn_type = getattr(mc, 'attention_type', 'MHA')
        if attn_type == 'MLA' and getattr(mc, 'kv_lora_rank', None):
            return (mc.kv_lora_rank + mc.qk_rope_head_dim) * bytes_per_element

        if attn_type == 'HYBRID_CSA_HCA':
            # CSA layers: effective KV per token ≈ 1/csa_chunk_size of MHA
            # HCA layers: effective KV per token ≈ 1/hca_chunk_size of MHA
            # Layers alternate CSA/HCA → use average compression factor
            csa_factor = 1.0 / (getattr(mc, 'csa_chunk_size', None) or 64)
            hca_factor = 1.0 / (getattr(mc, 'hca_chunk_size', None) or 1024)
            avg_factor = (csa_factor + hca_factor) / 2.0
            head_dim = mc.embedding_dim // mc.num_q_heads
            return avg_factor * 2 * mc.num_kv_heads * head_dim * bytes_per_element

        # MHA / GQA
        head_dim = mc.embedding_dim // mc.num_q_heads
        return 2 * mc.num_kv_heads * head_dim * bytes_per_element

    def _get_kv_cache_load_time(self, batch: Batch) -> float:
        """KV cache load time per layer in ms, computed from PCIe bandwidth.

        Models KV cache being transferred from host memory to GPU over PCIe:
            kv_load_time = kv_bytes / PCIe_bandwidth

        Returns 0 if no decode tokens or no bandwidth info available.
        """
        if self._pcie_bw_bytes_per_s <= 0:
            return 0.0

        # Collect decode batch size and average KV length from the batch
        decode_bs = 0
        total_kv_tokens = 0
        for request in batch.requests:
            if request._is_prefill_complete:
                decode_bs += 1
                total_kv_tokens += request.num_processed_tokens

        if decode_bs == 0:
            return 0.0

        avg_kv_len = total_kv_tokens / decode_bs

        # kv_bytes / PCIe_bandwidth
        total_bytes = self._kv_bytes_per_token_per_layer * avg_kv_len * decode_bs
        load_time_s = total_bytes / self._pcie_bw_bytes_per_s
        return load_time_s * 1e3  # ms

    def get_execution_time(self, batch: Batch, pipeline_stage: int) -> ExecutionTime:
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._get_pipeline_parallel_communication_time(batch)
            )

        if self._replica_config.tensor_parallel_size == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)
            )

        # MoE times (default to 0 for dense models)
        moe_routing_time = self._get_moe_routing_time(batch)
        moe_expert_compute_time = self._get_moe_expert_compute_time(batch)
        moe_expert_load_time = self._get_moe_expert_load_time(batch)
        expert_parallel_comm_time = self._get_expert_parallel_comm_time(batch)

        # Per-layer weight load times (CPU-offloaded vs GPU-resident)
        per_layer_weight_load_times = self._get_per_layer_weight_load_times()

        # Bandwidth-based KV cache load time per layer
        kv_cache_load_time = self._get_kv_cache_load_time(batch)

        return ExecutionTime(
            self._num_layers_per_pipeline_stage,
            self._get_attention_rope_execution_time(batch),
            self._get_attention_kv_cache_save_execution_time(batch),
            self._get_attention_decode_execution_time(batch),
            self._get_attention_prefill_execution_time(batch),
            self._get_attention_layer_pre_proj_execution_time(batch),
            self._get_attention_layer_post_proj_execution_time(batch),
            self._get_mlp_layer_up_proj_execution_time(batch),
            self._get_mlp_layer_down_proj_execution_time(batch),
            self._get_mlp_layer_act_execution_time(batch),
            self._get_attn_norm_layer_act_execution_time(batch),
            self._get_mlp_norm_layer_act_execution_time(batch),
            self._get_add_layer_act_execution_time(batch),
            tensor_parallel_communication_time,
            pipeline_parallel_communication_time,
            self._get_schedule_time(batch),
            self._get_sampler_e2e_time(batch),
            self._get_prepare_inputs_e2e_time(batch),
            self._get_process_model_outputs_time(batch),
            self._get_ray_comm_time(batch),
            enable_kv_prefetch=self._enable_kv_prefetch,
            moe_routing_time=moe_routing_time,
            moe_expert_compute_time=moe_expert_compute_time,
            moe_expert_load_time=moe_expert_load_time,
            expert_parallel_comm_time=expert_parallel_comm_time,
            kv_cache_load_time_per_layer=kv_cache_load_time,
            per_layer_weight_load_times=per_layer_weight_load_times,
        )

    @abstractmethod
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_schedule_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_ray_comm_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    # Per-layer weight load times - returns None by default (uniform loading).
    # Overridden in MoE-aware predictors to support CPU-offloaded weights.
    def _get_per_layer_weight_load_times(self) -> Optional[List[float]]:
        if hasattr(self, 'get_per_layer_weight_load_times'):
            return self.get_per_layer_weight_load_times()
        return None

    # MoE methods - default to 0 for dense models, overridden in MoE-aware predictors
    def _get_moe_routing_time(self, batch: Batch) -> float:
        return 0.0

    def _get_moe_expert_compute_time(self, batch: Batch) -> float:
        return 0.0

    def _get_moe_expert_load_time(self, batch: Batch) -> float:
        return 0.0

    def _get_expert_parallel_comm_time(self, batch: Batch) -> float:
        return 0.0
