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
        # Per-layer expert weight load times (ms).  When n_cpu_moe > 0,
        # the first N entries use PCIe bandwidth (slow) and the rest use
        # HBM bandwidth (fast).  None means uniform (all layers use
        # moe_expert_load_time).
        per_layer_weight_load_times: Optional[List[float]] = None,
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

        # Per-layer expert weight load times (CPU-offloaded vs GPU-resident)
        self._per_layer_weight_load_times = per_layer_weight_load_times

        # Per-layer breakdowns
        self._enable_kv_prefetch = enable_kv_prefetch
        self._layer_executions: List[LayerExecutionTime] = (
            layer_executions if layer_executions is not None else []
        )

        # PDD: Inter-GPU KV cache transfer time (between prefill and decode stages)
        self._inter_gpu_kv_transfer_time: float = 0.0
        self._inter_gpu_kv_transfer_bytes: float = 0.0

        # Build per-layer breakdowns if not provided
        if not self._layer_executions:
            self._build_layer_executions()

        # Schedule layers with overlap across hardware streams
        self._schedule_layer_streams()

    def _build_layer_executions(self) -> None:
        """Build per-layer execution time breakdowns from aggregate times.

        When per_layer_weight_load_times is provided (n_cpu_moe > 0),
        each layer gets its own weight_load_time reflecting whether
        expert weights are on CPU (PCIe) or GPU (HBM).
        """
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
                # Use per-layer weight load time if available
                if self._per_layer_weight_load_times is not None:
                    layer_weight_load = self._per_layer_weight_load_times[i]
                else:
                    layer_weight_load = self._moe_expert_load_time
            else:
                mlp_compute = (
                    self._mlp_layer_up_proj_execution_time
                    + self._mlp_layer_down_proj_execution_time
                    + self._mlp_layer_act_execution_time
                    + self._mlp_norm_time
                )
                layer_weight_load = 0.0

            layer = LayerExecutionTime(
                layer_index=i,
                attention_compute_time=attn_compute,
                mlp_compute_time=mlp_compute,
                kv_cache_load_time=self._kv_cache_load_time_per_layer,
                weight_load_time=layer_weight_load,
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

    def _schedule_layer_streams(self) -> None:
        """Schedule layers across three hardware streams with overlap.

        Hardware streams (can operate concurrently):
          SM:   Compute (attention + MLP GEMMs)
          DMA:  PCIe transfers (KV cache prefetch for next layer)
          NCCL: All-reduce communication (runs after compute)

        Scheduling rules:
          1. Layer N compute can start when:
             - SM is free (previous layer's comm finished)
             - DMA finished loading layer N's KV (if prefetched by layer N-1)
          2. DMA prefetch of layer N+1's KV starts at same time as layer N compute
          3. NCCL all-reduce starts after layer N's SM compute finishes
          4. MoE weight loading overlaps with expert compute: max(compute, load)

        Without prefetch enabled, I/O runs sequentially before compute.
        """
        if not self._layer_executions:
            return

        # Track absolute time (ms) when each stream becomes free
        sm_free = 0.0    # When SM (compute + NCCL) finishes
        dma_free = 0.0   # When DMA engine finishes

        for i, layer in enumerate(self._layer_executions):
            compute_dur = layer.compute_time
            io_dur = layer.io_time          # KV load + weight load
            comm_dur = layer.comm_time      # all-reduce + EP comm

            if self._enable_kv_prefetch:
                # Layer N's KV was prefetched by layer N-1's DMA.
                # Layer N can start compute when BOTH:
                #   - SM is free (prev layer's comm done)
                #   - DMA is free (this layer's KV prefetch done)
                layer_start = max(sm_free, dma_free)

                # SM: runs compute starting at layer_start
                compute_start_abs = layer_start
                compute_end_abs = compute_start_abs + compute_dur

                # DMA: starts prefetching NEXT layer's KV at the same time as compute
                # For this layer, the DMA loads the next layer's KV cache
                if i < len(self._layer_executions) - 1:
                    next_io = self._layer_executions[i + 1].io_time
                else:
                    next_io = 0.0
                dma_start_abs = compute_start_abs
                dma_end_abs = dma_start_abs + next_io

                # Prefetch savings: how much of next layer's I/O overlaps with compute
                if i < len(self._layer_executions) - 1:
                    overlap = min(compute_dur, next_io)
                    self._layer_executions[i + 1].prefetch_overlap_savings = overlap

                # NCCL: starts after compute finishes
                comm_start_abs = compute_end_abs
                comm_end_abs = comm_start_abs + comm_dur

                # Record per-stream intervals relative to layer_start
                layer.compute_start = compute_start_abs - layer_start
                layer.compute_end = compute_end_abs - layer_start
                layer.io_start = dma_start_abs - layer_start
                layer.io_end = dma_end_abs - layer_start
                layer.comm_start = comm_start_abs - layer_start
                layer.comm_end = comm_end_abs - layer_start

                # Update stream free times
                sm_free = comm_end_abs
                dma_free = dma_end_abs

            else:
                # No prefetch: I/O runs sequentially before compute
                layer_start = sm_free

                # I/O first (must load KV before compute can use it)
                io_start_abs = layer_start
                io_end_abs = io_start_abs + io_dur

                # Compute after I/O
                compute_start_abs = io_end_abs
                compute_end_abs = compute_start_abs + compute_dur

                # Comm after compute
                comm_start_abs = compute_end_abs
                comm_end_abs = comm_start_abs + comm_dur

                layer.compute_start = compute_start_abs - layer_start
                layer.compute_end = compute_end_abs - layer_start
                layer.io_start = io_start_abs - layer_start
                layer.io_end = io_end_abs - layer_start
                layer.comm_start = comm_start_abs - layer_start
                layer.comm_end = comm_end_abs - layer_start

                sm_free = comm_end_abs
                dma_free = io_end_abs

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

    # ---- PDD (Prefill-Decode Disaggregation) properties ----

    @property
    def inter_gpu_kv_transfer_time_ms(self) -> float:
        """Inter-GPU KV cache transfer time in milliseconds (PDD)."""
        return self._inter_gpu_kv_transfer_time

    @property
    def inter_gpu_kv_transfer_bytes(self) -> float:
        """Total KV cache bytes transferred between GPUs (PDD)."""
        return self._inter_gpu_kv_transfer_bytes

    def set_inter_gpu_kv_transfer(
        self, transfer_time_ms: float, transfer_bytes: float = 0.0
    ) -> None:
        """Set inter-GPU KV cache transfer time (called by KvCacheTransferEvent).

        Args:
            transfer_time_ms: Transfer latency in milliseconds
            transfer_bytes: Optional total bytes transferred (for metrics)
        """
        self._inter_gpu_kv_transfer_time = transfer_time_ms
        self._inter_gpu_kv_transfer_bytes = transfer_bytes

    @property
    def effective_inter_gpu_kv_transfer_time_ms(self) -> float:
        """Inter-GPU KV transfer time after accounting for overlap with first decode layer compute.

        In PDD, the KV transfer can overlap with the first decode layer's compute
        if KV prefetch is enabled. This returns the effective (reduced) latency.
        """
        if not self._enable_kv_prefetch or not self._layer_executions:
            return self._inter_gpu_kv_transfer_time

        # Compute time of first decode layer
        first_layer_compute = (
            self._layer_executions[0].compute_time if self._layer_executions else 0.0
        )

        # Overlap is the minimum of transfer time and first layer compute
        overlap = min(self._inter_gpu_kv_transfer_time, first_layer_compute)
        return max(0.0, self._inter_gpu_kv_transfer_time - overlap)

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

    def _get_mlp_layer_execution_time(self, layer_idx: int = 0) -> float:
        if self._moe_expert_compute_time > 0:
            # Use per-layer weight load time if available (n_cpu_moe)
            if self._per_layer_weight_load_times is not None:
                load_time = self._per_layer_weight_load_times[layer_idx]
            else:
                load_time = self._moe_expert_load_time
            return (
                self._moe_routing_time
                + max(self._moe_expert_compute_time, load_time)
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

    def _get_block_execution_time(self, layer_idx: int = 0) -> float:
        return (
            self._get_attention_layer_execution_time()
            + self._get_mlp_layer_execution_time(layer_idx)
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

        When per_layer_weight_load_times is set (n_cpu_moe > 0), each
        layer may have a different MLP execution time due to CPU vs GPU
        weight loading, so we sum per-layer instead of multiplying.

        When KV prefetch is enabled, we subtract the prefetch savings
        (computed from the *bandwidth-based* KV load estimate) from
        the profiled base time.  This avoids double-counting: the base
        time uses real profiled values, and only the prefetch *delta*
        comes from the first-principles bandwidth model.
        """
        if self._per_layer_weight_load_times is not None:
            # Per-layer sum: each layer may have different weight load time
            pipeline_stage_execution_time = sum(
                self._get_block_execution_time(i)
                for i in range(self._num_layers_per_pipeline_stage)
            )
        else:
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
