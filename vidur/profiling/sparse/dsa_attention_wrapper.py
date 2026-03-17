"""Wrapper for profiling DSA (DeepSeek Sparse Attention) decode pipeline.

Profiles the full sparse attention pipeline per (num_tokens, seq_len) pair:

  Block A — Q Projection:
    dsa_q_down_proj, dsa_q_up_proj, dsa_rope

  Block B — KV Indexing:
    dsa_indexer_load, dsa_indexer_matmul, dsa_topk_select, dsa_kv_fetch

  Block C — Attention Compute:
    dsa_kv_up_proj, dsa_attn_score, dsa_o_proj

  Block-level:
    dsa_block_norm, dsa_block_residual

Unlike the MLA attention profiler (which sweeps over num_tokens at fixed
seq_len=num_tokens), the DSA profiler sweeps over (num_tokens, seq_len)
pairs because seq_len independently controls the indexer cache size and
the cost of Block B.

Usage:
    Called from vidur.profiling.sparse.main with DSA-specific config.
"""

import os

import torch

from vidur.profiling.common.cuda_timer import CudaTimer

import sarathi.metrics.cuda_timer
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.weight_utils import initialize_dummy_weights

from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.sparse.dsa_attention_impl import DSAModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2
ACTIVE_STEPS = 10


class DSAAttentionWrapper:
    """Profiling wrapper for DSA sparse attention.

    Instantiates a DSAModel and runs it across (num_tokens, seq_len) pairs
    to collect per-operation timing statistics.

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        kv_lora_rank: KV latent dimension
        q_lora_rank: Q latent dimension
        qk_nope_head_dim: Non-positional head dim
        qk_rope_head_dim: Rotary head dim
        v_head_dim: Value head dim
        dsa_selected_tokens: Tokens selected by indexer (default 2048)
        dsa_sliding_window: Sliding window tokens (default 512)
        max_seq_len: Maximum sequence length for allocation
        profile_method: Profiling method (cuda_event, kineto, etc.)
        output_dir: Directory for trace outputs
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        dsa_selected_tokens: int = 2048,
        dsa_sliding_window: int = 512,
        max_seq_len: int = 131072,
        profile_method: str = "cuda_event",
        output_dir: str = "profiling_outputs",
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
        self.dsa_selected_tokens = dsa_selected_tokens
        self.dsa_sliding_window = dsa_sliding_window
        self.max_seq_len = max_seq_len
        self.profile_method = profile_method
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        self.model = DSAModel(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            dsa_selected_tokens=dsa_selected_tokens,
            dsa_sliding_window=dsa_sliding_window,
            max_seq_len=max_seq_len,
            num_repeat_steps=(
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value
                else 1
            ),
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def profile(self, num_tokens: int, seq_len: int) -> dict:
        """Profile DSA decode for a given (num_tokens, seq_len) pair.

        Args:
            num_tokens: Number of decode tokens (typically 1 for decode, >1 for batch)
            seq_len: Total KV cache sequence length (controls indexer size)

        Returns:
            Dict with time_stats (per-operation timing) and config metadata.
        """
        hidden_states = torch.randn(
            num_tokens, self.hidden_size,
            device="cuda", dtype=torch.float16,
        )

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            # Warmup
            self.model(hidden_states, seq_len=seq_len)
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)
            with record_function_tracer:
                self.model(hidden_states, seq_len=seq_len)

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            # Warmup
            for _ in range(WARMUP_STEPS):
                self.model(hidden_states, seq_len=seq_len)

            torch.cuda.synchronize()
            self.timer_stats_store.clear_stats()

            # Active measurement
            for _ in range(ACTIVE_STEPS):
                self.model(hidden_states, seq_len=seq_len)

            torch.cuda.synchronize()
            time_stats = self.timer_stats_store.get_stats()

        stats = {
            "time_stats": time_stats,
            # Model config
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "q_lora_rank": self.q_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            # DSA config
            "dsa_selected_tokens": self.dsa_selected_tokens,
            "dsa_sliding_window": self.dsa_sliding_window,
            "dsa_attended": self.dsa_selected_tokens + self.dsa_sliding_window,
            # Input dimensions
            "num_tokens": num_tokens,
            "seq_len": seq_len,
        }
        self.timer_stats_store.clear_stats()

        return stats
