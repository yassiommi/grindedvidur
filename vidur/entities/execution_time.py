from typing import List, Optional

from vidur.entities.base_entity import BaseEntity
from vidur.entities.layer_execution_time import LayerExecutionTime


class ExecutionTime(BaseEntity):
    def __init__(
        self,
        num_layers_per_pipeline_stage: int,
        attention_rope_execution_time: float,
        attention_kv_cache_save_execution_time: float,
        attention_decode_execution_time: float,
        attention_prefill_execution_time: float,
        attention_layer_pre_proj_execution_time: float,
        attention_layer_post_proj_execution_time: float,
        mlp_layer_up_proj_execution_time: float,
        mlp_layer_down_proj_execution_time: float,
        mlp_layer_act_execution_time: float,
        attn_norm_time: float,
        mlp_norm_time: float,
        add_time: float,
        tensor_parallel_communication_time: float,
        pipeline_parallel_communication_time: float,
        schedule_time: float,
        sampler_e2e_time: float,
        prepare_inputs_e2e_time: float,
        process_model_outputs_time: float,
        ray_comm_time: float,
        # New: per-layer breakdowns and MoE/prefetch parameters
        layer_executions: Optional[List[LayerExecutionTime]] = None,
        enable_kv_prefetch: bool = False,
        # MoE fields
        moe_routing_time: float = 0.0,
        moe_expert_compute_time: float = 0.0,
        moe_expert_load_time: float = 0.0,
        expert_parallel_comm_time: float = 0.0,
        # KV cache load time per layer (ms), computed as kv_bytes / PCIe_bw.
        # Used ONLY for per-layer Gantt breakdown and prefetch-savings
        # calculation — the profiled attention_decode_execution_time already
        # includes KV loading implicitly, so this is *not* added to model_time.
        kv_cache_load_time_per_layer: float = 0.0,
    ) -> None:
        self._id = ExecutionTime.generate_id()

        self._num_layers_per_pipeline_stage = num_layers_per_pipeline_stage
        self._attention_rope_execution_time = attention_rope_execution_time
        self._attention_kv_cache_save_execution_time = (
            attention_kv_cache_save_execution_time
        )
        self._attention_decode_execution_time = attention_decode_execution_time
        self._attention_prefill_execution_time = attention_prefill_execution_time
        self._attention_layer_pre_proj_execution_time = (
            attention_layer_pre_proj_execution_time
        )
        self._attention_layer_post_proj_execution_time = (
            attention_layer_post_proj_execution_time
        )
        self._mlp_layer_up_proj_execution_time = mlp_layer_up_proj_execution_time
        self._mlp_layer_down_proj_execution_time = mlp_layer_down_proj_execution_time
        self._mlp_layer_act_execution_time = mlp_layer_act_execution_time
        self._mlp_norm_time = mlp_norm_time
        self._attn_norm_time = attn_norm_time
        self._add_time = add_time
        self._tensor_parallel_communication_time = tensor_parallel_communication_time
        self._pipeline_parallel_communication_time = (
            pipeline_parallel_communication_time
        )
        self._schedule_time = schedule_time
        self._sampler_e2e_time = sampler_e2e_time
        self._prepare_inputs_e2e_time = prepare_inputs_e2e_time
        self._process_model_outputs_time = process_model_outputs_time
        self._ray_comm_time = ray_comm_time

        # MoE fields
        self._moe_routing_time = moe_routing_time
        self._moe_expert_compute_time = moe_expert_compute_time
        self._moe_expert_load_time = moe_expert_load_time
        self._expert_parallel_comm_time = expert_parallel_comm_time

        # PCIe-based KV I/O estimate (ms) — for Gantt and prefetch only
        self._kv_cache_load_time_per_layer = kv_cache_load_time_per_layer

        # Per-layer breakdowns
        self._enable_kv_prefetch = enable_kv_prefetch
        self._layer_executions: List[LayerExecutionTime] = (
            layer_executions if layer_executions is not None else []
        )

        # Build per-layer breakdowns if not provided
        if not self._layer_executions:
            self._build_layer_executions()

        # Apply KV cache prefetch overlap if enabled
        if self._enable_kv_prefetch:
            self._apply_kv_prefetch_overlap()

    def _build_layer_executions(self) -> None:
        """Build per-layer execution time breakdowns from aggregate times."""
        for i in range(self._num_layers_per_pipeline_stage):
            attn_compute = (
                self._attention_layer_pre_proj_execution_time
                + self._attention_layer_post_proj_execution_time
                + self._attention_rope_execution_time
                + self._attention_kv_cache_save_execution_time
                + self._attention_decode_execution_time
                + self._attention_prefill_execution_time
                + self._attn_norm_time
            )

            is_moe = self._moe_expert_compute_time > 0
            if is_moe:
                mlp_compute = (
                    self._moe_routing_time
                    + self._moe_expert_compute_time
                    + self._mlp_norm_time
                )
            else:
                mlp_compute = (
                    self._mlp_layer_up_proj_execution_time
                    + self._mlp_layer_down_proj_execution_time
                    + self._mlp_layer_act_execution_time
                    + self._mlp_norm_time
                )

            layer = LayerExecutionTime(
                layer_index=i,
                attention_compute_time=attn_compute,
                mlp_compute_time=mlp_compute,
                kv_cache_load_time=self._kv_cache_load_time_per_layer,
                weight_load_time=self._moe_expert_load_time if is_moe else 0.0,
                tensor_parallel_comm_time=(
                    self._tensor_parallel_communication_time * 2
                ),
                expert_parallel_comm_time=(
                    self._expert_parallel_comm_time if is_moe else 0.0
                ),
                is_moe_layer=is_moe,
                routing_time=self._moe_routing_time if is_moe else 0.0,
            )
            self._layer_executions.append(layer)

    def _apply_kv_prefetch_overlap(self) -> None:
        """GPU-initiated I/O: prefetch KV cache for next layer overlapped with current compute.

        At the start of layer N, a prefetch of KV cache for layer N+1 is initiated.
        Layer N+1 cannot start computing until all I/O and comm from layer N is done.
        The overlap saves time equal to min(current_layer_compute, next_layer_kv_load).
        """
        for i in range(len(self._layer_executions) - 1):
            current = self._layer_executions[i]
            next_layer = self._layer_executions[i + 1]

            # Prefetch next layer's KV cache during current layer's compute
            prefetchable = next_layer.kv_cache_load_time
            available_overlap = current.compute_time
            overlap = min(prefetchable, available_overlap)
            next_layer.prefetch_overlap_savings = overlap

    @property
    def layer_executions(self) -> List[LayerExecutionTime]:
        return self._layer_executions

    @property
    def enable_kv_prefetch(self) -> bool:
        return self._enable_kv_prefetch

    @property
    def total_prefetch_savings_ms(self) -> float:
        """Total time saved by KV cache prefetching (ms)."""
        return sum(l.prefetch_overlap_savings for l in self._layer_executions)

    # ---- MoE properties ----

    @property
    def moe_routing_time(self) -> float:
        return self._moe_routing_time

    @property
    def moe_expert_compute_time(self) -> float:
        return self._moe_expert_compute_time

    @property
    def moe_expert_load_time(self) -> float:
        return self._moe_expert_load_time

    @property
    def expert_parallel_comm_time(self) -> float:
        return self._expert_parallel_comm_time

    # ---- Original properties (backward compatible) ----

    def _get_mlp_layer_execution_time(self) -> float:
        if self._moe_expert_compute_time > 0:
            return (
                self._moe_routing_time
                + max(self._moe_expert_compute_time, self._moe_expert_load_time)
                + self._expert_parallel_comm_time
                + self._tensor_parallel_communication_time
                + self._mlp_norm_time
            )
        return (
            self._mlp_layer_up_proj_execution_time
            + self._mlp_layer_down_proj_execution_time
            + self._mlp_layer_act_execution_time
            + self._tensor_parallel_communication_time
            + self._mlp_norm_time
        )

    def _get_attention_layer_execution_time(self) -> float:
        return (
            self._attention_layer_pre_proj_execution_time
            + self._attention_layer_post_proj_execution_time
            + self._attention_rope_execution_time
            + self._attention_kv_cache_save_execution_time
            + self._attention_decode_execution_time
            + self._attention_prefill_execution_time
            + self._tensor_parallel_communication_time
            + self._attn_norm_time
        )

    def _get_block_execution_time(self) -> float:
        return (
            self._get_attention_layer_execution_time()
            + self._get_mlp_layer_execution_time()
            + self._add_time
        )

    def _get_cpu_overhead(self) -> float:
        return (
            self._schedule_time
            + self._sampler_e2e_time
            + self._prepare_inputs_e2e_time
            + self._process_model_outputs_time
            + self._ray_comm_time
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers_per_pipeline_stage

    @property
    def mlp_layer_up_proj_execution_time(self) -> float:
        return self._mlp_layer_up_proj_execution_time

    @property
    def mlp_layer_down_proj_execution_time(self) -> float:
        return self._mlp_layer_down_proj_execution_time

    @property
    def mlp_layer_act_execution_time(self) -> float:
        return self._mlp_layer_act_execution_time

    @property
    def mlp_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_pre_proj_time(self) -> float:
        return self._attention_layer_pre_proj_execution_time

    @property
    def attention_post_proj_time(self) -> float:
        return self._attention_layer_post_proj_execution_time

    @property
    def attention_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_rope_execution_time(self) -> float:
        return self._attention_rope_execution_time

    @property
    def attention_kv_cache_save_execution_time(self) -> float:
        return self._attention_kv_cache_save_execution_time

    @property
    def attention_decode_execution_time(self) -> float:
        return self._attention_decode_execution_time

    @property
    def attention_prefill_execution_time(self) -> float:
        return self._attention_prefill_execution_time

    @property
    def pipeline_parallel_communication_time(self) -> float:
        return self._pipeline_parallel_communication_time

    @property
    def schedule_time(self) -> float:
        return self._schedule_time

    @property
    def sampler_e2e_time(self) -> float:
        return self._sampler_e2e_time

    @property
    def prepare_inputs_e2e_time(self) -> float:
        return self._prepare_inputs_e2e_time

    @property
    def process_model_outputs_time(self) -> float:
        return self._process_model_outputs_time

    @property
    def ray_comm_time(self) -> float:
        return self._ray_comm_time

    @property
    def mlp_norm_time(self) -> float:
        return self._mlp_norm_time

    @property
    def attn_norm_time(self) -> float:
        return self._attn_norm_time

    @property
    def add_time(self) -> float:
        return self._add_time

    @property
    def model_time(self) -> float:
        """Model execution time in seconds.

        The base time always comes from the profiled per-operation values
        (attention_decode_execution_time etc.), which already implicitly
        include KV cache loading.

        When KV prefetch is enabled, we subtract the prefetch savings
        (computed from the *bandwidth-based* KV load estimate) from
        the profiled base time.  This avoids double-counting: the base
        time uses real profiled values, and only the prefetch *delta*
        comes from the first-principles bandwidth model.
        """
        block_execution_time = self._get_block_execution_time()
        pipeline_stage_execution_time = (
            block_execution_time * self._num_layers_per_pipeline_stage
        )
        base_ms = (
            pipeline_stage_execution_time + self.pipeline_parallel_communication_time
        )

        if self._enable_kv_prefetch and self._layer_executions:
            base_ms -= self.total_prefetch_savings_ms

        return base_ms * 1e-3

    @property
    def model_time_ms(self) -> float:
        return self.model_time * 1e3

    @property
    def total_time(self) -> float:
        # return in seconds
        return self.model_time + self._get_cpu_overhead() * 1e-3
