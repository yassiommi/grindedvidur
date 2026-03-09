"""Multi-head Latent Attention (MLA) profiling model.

Implements DeepSeek-V3's MLA for profiling:

  Encoding (prefill):
    1. q_down_proj: hidden -> q_lora_rank (compress Q)
    2. q_up_proj:   q_lora_rank -> num_heads * (qk_nope_dim + qk_rope_dim)
    3. kv_down_proj: hidden -> kv_lora_rank + qk_rope_dim (compress KV)
    4. RoPE on the rope portion of Q and K
    5. FlashAttention / paged attention kernel
    6. o_proj: num_heads * v_head_dim -> hidden

  Decoding:
    Steps 1-4 plus attention against compressed KV cache
    (KV cache stores kv_lora_rank + qk_rope_dim per token instead of
     2 * num_kv_heads * head_dim, which is the MLA compression benefit)

  The kv_up_proj is absorbed into the attention kernel at decode time
  (online decompression), but during profiling we measure it separately
  to capture the GEMM cost.
"""

import torch

from vidur.profiling.common.cuda_timer import CudaTimer


class MLAAttention(torch.nn.Module):
    """MLA attention block for profiling.

    This profiles the projection GEMMs and simulated attention, NOT the
    actual FlashAttention kernel (which is profiled separately by the
    attention profiler). The focus here is on the MLA-specific projections.
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Q compression: hidden -> q_lora_rank -> num_heads * (nope + rope)
        self.q_down_proj = torch.nn.Linear(
            hidden_size, q_lora_rank, bias=False
        )
        self.q_up_proj = torch.nn.Linear(
            q_lora_rank,
            num_heads * (qk_nope_head_dim + qk_rope_head_dim),
            bias=False,
        )

        # KV compression: hidden -> kv_lora_rank (+ rope portion handled separately)
        self.kv_down_proj = torch.nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )

        # KV decompression (used during decode for latent -> full KV)
        # In production this is absorbed into attention; we profile it separately
        self.kv_up_proj = torch.nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = torch.nn.Linear(
            num_heads * v_head_dim, hidden_size, bias=False
        )

        # Timers
        self._q_down_proj_timer = CudaTimer("mla_q_down_proj")
        self._q_up_proj_timer = CudaTimer("mla_q_up_proj")
        self._kv_down_proj_timer = CudaTimer("mla_kv_down_proj")
        self._kv_up_proj_timer = CudaTimer("mla_kv_up_proj")
        self._rope_timer = CudaTimer("mla_rope")
        self._attn_timer = CudaTimer("mla_attn_score")
        self._o_proj_timer = CudaTimer("mla_o_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        # Q compression
        with self._q_down_proj_timer:
            q_compressed = self.q_down_proj(hidden_states)  # [T, q_lora_rank]
        with self._q_up_proj_timer:
            q = self.q_up_proj(q_compressed)  # [T, H * (nope + rope)]

        # KV compression
        with self._kv_down_proj_timer:
            kv_compressed = self.kv_down_proj(hidden_states)  # [T, kv_lora_rank + rope]
            # Split compressed KV into latent and rope parts
            kv_latent = kv_compressed[:, :self.kv_lora_rank]
            kv_rope = kv_compressed[:, self.kv_lora_rank:]

        # KV decompression (latent -> full K,V for attention)
        with self._kv_up_proj_timer:
            kv_full = self.kv_up_proj(kv_latent)  # [T, H * (nope + v)]

        # Simulated RoPE (applied to rope portions of Q and K)
        with self._rope_timer:
            # Simulate RoPE by element-wise operations
            q_rope_portion = q[:, :self.num_heads * self.qk_rope_head_dim]
            _ = q_rope_portion * kv_rope.unsqueeze(1).expand(-1, self.qk_rope_head_dim)

        # Simulated attention (the actual kernel shape matters for profiling)
        with self._attn_timer:
            # Simulate attention output with correct shape
            attn_output = torch.randn(
                num_tokens, self.num_heads * self.v_head_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )

        # Output projection
        with self._o_proj_timer:
            output = self.o_proj(attn_output)  # [T, hidden]

        return output


class MLAModel(torch.nn.Module):
    """Wraps MLA attention in norm + residual for profiling."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        q_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        num_repeat_steps: int = 1,
    ):
        super().__init__()
        self.num_repeat_steps = num_repeat_steps

        self.norm = torch.nn.LayerNorm(hidden_size)
        self.attn = MLAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
        )

        self._norm_timer = CudaTimer("mla_block_norm")
        self._residual_timer = CudaTimer("mla_block_residual")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_repeat_steps):
            residual = hidden_states
            with self._norm_timer:
                hidden_states = self.norm(hidden_states)
            hidden_states = self.attn(hidden_states)
            with self._residual_timer:
                hidden_states = hidden_states + residual
        return hidden_states
