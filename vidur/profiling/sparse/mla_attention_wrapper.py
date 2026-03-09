"""Wrapper for profiling MLA (Multi-head Latent Attention) projections.

Profiles the MLA-specific GEMM operations:
  - Q down/up projection (LoRA compression)
  - KV down projection (latent compression)
  - KV up projection (decompression for attention)
  - Output projection
  - RoPE application
"""

import os

import torch

from vidur.profiling.common.cuda_timer import CudaTimer

import sarathi.metrics.cuda_timer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.weight_utils import initialize_dummy_weights

from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.sparse.mla_attention_impl import MLAModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2
ACTIVE_STEPS = 10


class MLAAttentionWrapper:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        profile_method: str,
        output_dir: str,
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.profile_method = profile_method
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.model = MLAModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            num_repeat_steps=(
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                else 1
            ),
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def profile(self, num_tokens: int):
        hidden_states = torch.randn(
            num_tokens, self.hidden_size,
            device="cuda", dtype=torch.float16,
        )

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            self.model(hidden_states)
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)
            with record_function_tracer:
                self.model(hidden_states)

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            for _ in range(WARMUP_STEPS):
                self.model(hidden_states)

            torch.cuda.synchronize()
            self.timer_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self.model(hidden_states)

            torch.cuda.synchronize()
            time_stats = self.timer_stats_store.get_stats()

        stats = {
            "time_stats": time_stats,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "q_lora_rank": self.q_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "num_tokens": num_tokens,
        }
        self.timer_stats_store.clear_stats()

        return stats
