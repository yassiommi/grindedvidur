"""Wrapper for profiling CSA/HCA hybrid attention (DeepSeek-V4).

Profiles the attention-specific GEMM operations for both attention types:
  CSA: Q projection, chunk KV compression, sparse selector, output projection
  HCA: Q projection, heavy chunk compression, output projection
  mHC: residual stream additions (n_hc parallel streams)
"""

import os

import torch

from vidur.profiling.common.cuda_timer import CudaTimer

import sarathi.metrics.cuda_timer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.weight_utils import initialize_dummy_weights

from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.sparse.csa_hca_attention_impl import HybridCSAHCAModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2
ACTIVE_STEPS = 10


class CSAHCAAttentionWrapper:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        csa_chunk_size: int,
        csa_top_k: int,
        hca_chunk_size: int,
        n_hc: int,
        profile_method: str,
        output_dir: str,
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.csa_chunk_size = csa_chunk_size
        self.csa_top_k = csa_top_k
        self.hca_chunk_size = hca_chunk_size
        self.n_hc = n_hc
        self.profile_method = profile_method
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.model = HybridCSAHCAModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            csa_chunk_size=csa_chunk_size,
            csa_top_k=csa_top_k,
            hca_chunk_size=hca_chunk_size,
            n_hc=n_hc,
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
            "head_dim": self.head_dim,
            "csa_chunk_size": self.csa_chunk_size,
            "csa_top_k": self.csa_top_k,
            "hca_chunk_size": self.hca_chunk_size,
            "n_hc": self.n_hc,
            "num_tokens": num_tokens,
        }
        self.timer_stats_store.clear_stats()

        return stats
