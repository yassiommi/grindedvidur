"""CSA and HCA hybrid attention profiling models (DeepSeek-V4).

DeepSeek-V4 replaces MLA with a two-type hybrid attention system interleaved
across transformer layers:

  CSA (Compressed Sparse Attention) — moderate compression:
    - Tokens chunked into groups of csa_chunk_size
    - Each chunk compressed to a summary KV representation
    - Additional sparse top-k token selectors per query
    - KV cache ≈ (seq_len / csa_chunk_size + csa_top_k) entries per layer

  HCA (Heavily Compressed Attention) — aggressive compression:
    - Larger chunks (hca_chunk_size) collapsed to a single KV entry
    - KV cache ≈ seq_len / hca_chunk_size entries per layer
    - ~16× more aggressive than CSA

  Combined at 1M context vs MHA baseline:
    - CSA: ~1/64 KV cache
    - HCA: ~1/1024 KV cache
    - Average: ~10% of MHA (matches DeepSeek-V4 reported figure)

  mHC (Manifold-Constrained Hyper-Connections):
    - Replaces standard residual addition with n_hc=4 parallel streams
    - Modeled here as n_hc sequential residual additions (same FLOP count)
"""

import torch

from vidur.profiling.common.cuda_timer import CudaTimer


class CSAAttention(torch.nn.Module):
    """Compressed Sparse Attention block for profiling.

    Models the projection GEMMs for CSA's two-stage KV compression:
    1. Chunk projection: compress a chunk of tokens to a summary KV entry
    2. Sparse selector: score tokens for top-k sparse attention
    3. Output projection: aggregate attention output back to hidden size
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        csa_chunk_size: int,
        csa_top_k: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.csa_chunk_size = csa_chunk_size
        self.csa_top_k = csa_top_k

        # Q projection: hidden → num_heads × head_dim
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Chunk KV compression: (chunk_size × hidden) → (num_heads × head_dim) summary
        # Modeled as linear over hidden_size (applied per chunk)
        self.chunk_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Sparse selector: score each token for top-k selection
        self.sparse_selector = torch.nn.Linear(hidden_size, csa_top_k, bias=False)

        # Output projection
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self._q_proj_timer = CudaTimer("csa_q_proj")
        self._chunk_proj_timer = CudaTimer("csa_chunk_proj")
        self._sparse_selector_timer = CudaTimer("csa_sparse_selector")
        self._attn_timer = CudaTimer("csa_attn_score")
        self._o_proj_timer = CudaTimer("csa_o_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape

        # Q projection
        with self._q_proj_timer:
            q = self.q_proj(hidden_states)  # [T, H*D]

        # Chunk KV compression: process each chunk of csa_chunk_size tokens
        # Simulate by processing num_chunks = ceil(T / csa_chunk_size) representatives
        num_chunks = max(1, (num_tokens + self.csa_chunk_size - 1) // self.csa_chunk_size)
        chunk_repr = hidden_states[:num_chunks]  # approximate: use first num_chunks tokens
        with self._chunk_proj_timer:
            kv_summary = self.chunk_proj(chunk_repr)  # [num_chunks, H*D]

        # Sparse selector: score all tokens for top-k sparse attention
        with self._sparse_selector_timer:
            sparse_scores = self.sparse_selector(hidden_states)  # [T, top_k]

        # Simulate attention output (correct shape for output projection timing)
        with self._attn_timer:
            attn_output = torch.randn(
                num_tokens, self.num_heads * self.head_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )

        # Output projection
        with self._o_proj_timer:
            output = self.o_proj(attn_output)  # [T, hidden]

        return output


class HCAAttention(torch.nn.Module):
    """Heavily Compressed Attention block for profiling.

    Models HCA's aggressive chunk-level compression:
    1. Chunk projection: collapse hca_chunk_size tokens to a single KV entry
    2. Output projection: aggregate back to hidden size

    KV cache at 1M tokens with hca_chunk_size=1024: only 976 entries per layer
    vs 1M entries for standard MHA — a ~1000× reduction.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        hca_chunk_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hca_chunk_size = hca_chunk_size

        # Q projection
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Heavy chunk KV compression: (hca_chunk_size × hidden) → single (num_heads × head_dim)
        self.chunk_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)

        # Output projection
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self._q_proj_timer = CudaTimer("hca_q_proj")
        self._chunk_proj_timer = CudaTimer("hca_chunk_proj")
        self._attn_timer = CudaTimer("hca_attn_score")
        self._o_proj_timer = CudaTimer("hca_o_proj")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape

        # Q projection
        with self._q_proj_timer:
            q = self.q_proj(hidden_states)  # [T, H*D]

        # Heavy chunk compression: num_chunks = ceil(T / hca_chunk_size)
        num_chunks = max(1, (num_tokens + self.hca_chunk_size - 1) // self.hca_chunk_size)
        chunk_repr = hidden_states[:num_chunks]
        with self._chunk_proj_timer:
            kv_compressed = self.chunk_proj(chunk_repr)  # [num_chunks, H*D]

        # Simulate attention output
        with self._attn_timer:
            attn_output = torch.randn(
                num_tokens, self.num_heads * self.head_dim,
                device=hidden_states.device, dtype=hidden_states.dtype,
            )

        # Output projection
        with self._o_proj_timer:
            output = self.o_proj(attn_output)  # [T, hidden]

        return output


class HybridCSAHCAModel(torch.nn.Module):
    """Interleaved CSA/HCA hybrid attention model for profiling.

    Even layers use CSA, odd layers use HCA, matching DeepSeek-V4's
    interleaving strategy. Also models mHC residual connections with
    n_hc parallel streams.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        csa_chunk_size: int,
        csa_top_k: int,
        hca_chunk_size: int,
        n_hc: int = 4,
        num_repeat_steps: int = 1,
    ):
        super().__init__()
        self.n_hc = n_hc
        self.num_repeat_steps = num_repeat_steps

        self.norm = torch.nn.LayerNorm(hidden_size)

        # One CSA and one HCA block (representing one interleaved pair)
        self.csa = CSAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            csa_chunk_size=csa_chunk_size,
            csa_top_k=csa_top_k,
        )
        self.hca = HCAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            hca_chunk_size=hca_chunk_size,
        )

        self._norm_timer = CudaTimer("hybrid_norm")
        self._residual_timer = CudaTimer("hybrid_residual")
        self._rope_timer = CudaTimer("hybrid_rope")

    def _mhc_residual(self, residual: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """mHC residual: n_hc parallel stream additions (modeled sequentially)."""
        with self._residual_timer:
            x = attn_out
            for _ in range(self.n_hc):
                x = x + residual
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_repeat_steps):
            # CSA layer (even)
            residual = hidden_states
            with self._norm_timer:
                hidden_states = self.norm(hidden_states)
            with self._rope_timer:
                # Simulate RoPE on compressed head representations
                _ = hidden_states * 0.1
            csa_out = self.csa(hidden_states)
            hidden_states = self._mhc_residual(residual, csa_out)

            # HCA layer (odd)
            residual = hidden_states
            with self._norm_timer:
                hidden_states = self.norm(hidden_states)
            hca_out = self.hca(hidden_states)
            hidden_states = self._mhc_residual(residual, hca_out)

        return hidden_states
