"""DeepSeek Sparse Attention (DSA) profiling model.

Implements the full DSA decode pipeline for on-device profiling.  DSA is
DeepSeek-V3's mechanism for long-context decoding: instead of attending to
all KV cache tokens, it uses a lightweight indexer to select the most
relevant tokens, then runs full attention only on that sparse subset.

Pipeline structure (per layer, single decode token):

  Block A (Q Projection)  ─┐
                            ├─ run in parallel ─→ Block C (Attention + Output)
  Block B (KV Indexing)   ─┘

Block A — Q Projection:
  1. q_down_proj : hidden → q_lora_rank          (MLA compression)
  2. q_up_proj   : q_lora_rank → H*(nope+rope)   (MLA decompression)
  3. RoPE on Q rope portion

Block B — KV Indexing (sparse token selection):
  1. Indexer K cache load   : read FP8 indexer keys for all seq_len tokens from HBM
  2. Indexer matmul          : Q_compressed × indexer_K^T → scores for all tokens
  3. Top-K selection         : pick top-k tokens (e.g. 2048) from scores
  4. KV fetch (gather)       : load full MLA KV latents for selected + sliding window tokens

Block C — Attention Compute (runs after both A and B complete):
  1. kv_up_proj  : decompress fetched KV latents → full K, V
  2. Attention   : Q × K^T → softmax → × V  (on sparse subset only)
  3. o_proj      : attention output → hidden
  4. Residual add

Each operation is wrapped in CudaTimer for empirical profiling.
"""

import torch

from vidur.profiling.common.cuda_timer import CudaTimer


class DSAAttention(torch.nn.Module):
    """DSA decode attention block for profiling.

    Profiles the full sparse attention pipeline including indexer-based
    token selection and sparse KV gather, in addition to the standard
    MLA projections.

    Args:
        hidden_size: Model hidden dimension (7168 for DeepSeek-V3)
        num_heads: Number of attention heads (128 for DeepSeek-V3)
        kv_lora_rank: KV latent dimension (512 for DeepSeek-V3)
        q_lora_rank: Q latent dimension (1536 for DeepSeek-V3)
        qk_nope_head_dim: Non-positional head dim (128 for DeepSeek-V3)
        qk_rope_head_dim: Rotary head dim (64 for DeepSeek-V3)
        v_head_dim: Value head dim (128 for DeepSeek-V3)
        dsa_selected_tokens: Number of top-k tokens selected by indexer (2048)
        dsa_sliding_window: Sliding window size always attended (512)
        max_seq_len: Maximum sequence length for KV cache allocation
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
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dsa_selected_tokens = dsa_selected_tokens
        self.dsa_sliding_window = dsa_sliding_window
        self.dsa_attended = dsa_selected_tokens + dsa_sliding_window
        self.max_seq_len = max_seq_len

        # Derived dimensions
        self.q_total_dim = num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        self.kv_up_out_dim = num_heads * (qk_nope_head_dim + v_head_dim)
        self.o_proj_in_dim = num_heads * v_head_dim

        # ── Block A: Q Projection layers ──
        self.q_down_proj = torch.nn.Linear(
            hidden_size, q_lora_rank, bias=False
        )
        self.q_up_proj = torch.nn.Linear(
            q_lora_rank, self.q_total_dim, bias=False
        )

        # ── Block B: Indexer matmul layer (Q_compressed × indexer_K^T) ──
        # The indexer K cache is FP8, kv_lora_rank dims per token.
        # We simulate the matmul as a linear layer with seq_len outputs.
        # The actual indexer_K is allocated per-call based on seq_len.

        # ── Block C: Attention compute layers ──
        self.kv_up_proj = torch.nn.Linear(
            kv_lora_rank, self.kv_up_out_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.o_proj_in_dim, hidden_size, bias=False
        )

        # ── Timers ──
        # Block A
        self._q_down_proj_timer = CudaTimer("dsa_q_down_proj")
        self._q_up_proj_timer = CudaTimer("dsa_q_up_proj")
        self._rope_timer = CudaTimer("dsa_rope")
        # Block B
        self._indexer_load_timer = CudaTimer("dsa_indexer_load")
        self._indexer_matmul_timer = CudaTimer("dsa_indexer_matmul")
        self._topk_timer = CudaTimer("dsa_topk_select")
        self._kv_fetch_timer = CudaTimer("dsa_kv_fetch")
        # Block C
        self._kv_up_proj_timer = CudaTimer("dsa_kv_up_proj")
        self._attn_score_timer = CudaTimer("dsa_attn_score")
        self._o_proj_timer = CudaTimer("dsa_o_proj")

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Run DSA decode for a single token (or small batch).

        Args:
            hidden_states: [num_tokens, hidden_size] — the decode token(s)
            seq_len: Total sequence length (KV cache size) for this call.
                     Controls indexer K cache size and gather volume.

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype
        attended = min(self.dsa_attended, seq_len)

        # ═══════════════════════════════════════════════════════════════
        # BLOCK A: Q Projection
        # ═══════════════════════════════════════════════════════════════

        # A1: q_down_proj — compress Q
        with self._q_down_proj_timer:
            q_compressed = self.q_down_proj(hidden_states)  # [T, q_lora_rank]

        # A2: q_up_proj — decompress Q to full head dims
        with self._q_up_proj_timer:
            q = self.q_up_proj(q_compressed)  # [T, H*(nope+rope)]

        # A3: RoPE on rope portion of Q
        with self._rope_timer:
            rope_dim = self.num_heads * self.qk_rope_head_dim
            q_rope = q[:, :rope_dim]
            # Simulate RoPE: element-wise multiply (cos/sin) — same cost
            _ = q_rope * torch.ones_like(q_rope)

        # ═══════════════════════════════════════════════════════════════
        # BLOCK B: KV Indexing Pipeline
        # ═══════════════════════════════════════════════════════════════

        # B1: Load indexer K cache from HBM (FP8, kv_lora_rank per token)
        with self._indexer_load_timer:
            # Simulate HBM read: allocate + copy FP8 indexer keys for all tokens
            # FP8 = 1 byte per element, kv_lora_rank elements per token
            indexer_k = torch.empty(
                seq_len, self.kv_lora_rank,
                device=device, dtype=torch.float16,
            )
            # Force the HBM read by copying from a source buffer
            indexer_k.copy_(torch.randn_like(indexer_k))

        # B2: Indexer matmul — Q_compressed × indexer_K^T → [T, seq_len] scores
        with self._indexer_matmul_timer:
            # q_compressed: [T, kv_lora_rank], indexer_k: [seq_len, kv_lora_rank]
            # We use q_compressed (from q_lora_rank) projected down to kv_lora_rank
            # In practice, q_compressed is re-projected; we approximate with a matmul
            scores = torch.mm(
                q_compressed[:, :self.kv_lora_rank],
                indexer_k.T,
            )  # [T, seq_len]

        # B3: Top-k selection — pick top-k tokens from scores
        with self._topk_timer:
            k = min(self.dsa_selected_tokens, seq_len)
            _, topk_indices = torch.topk(scores, k, dim=-1)  # [T, k]

        # B4: KV fetch — gather full MLA KV latents for selected + sliding window
        with self._kv_fetch_timer:
            # MLA KV cache: (kv_lora_rank + qk_rope_head_dim) per token, FP16
            kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
            # Simulate full KV cache for all seq_len tokens
            kv_cache = torch.randn(
                seq_len, kv_cache_dim,
                device=device, dtype=dtype,
            )
            # Gather the selected tokens (simulates scattered gather)
            # Use first token's indices for the gather
            gather_indices = topk_indices[0, :min(k, attended)]
            fetched_kv = kv_cache[gather_indices]  # [attended, kv_cache_dim]
            # Pad to full attended size if sliding window tokens would be added
            if fetched_kv.shape[0] < attended:
                sw_tokens = min(self.dsa_sliding_window, seq_len - k)
                if sw_tokens > 0:
                    sw_kv = kv_cache[:sw_tokens]
                    fetched_kv = torch.cat([fetched_kv, sw_kv], dim=0)

        # ═══════════════════════════════════════════════════════════════
        # BLOCK C: Attention Compute
        # ═══════════════════════════════════════════════════════════════
        actual_attended = fetched_kv.shape[0]

        # C1: KV decompression — kv_up_proj on fetched latents
        with self._kv_up_proj_timer:
            kv_latent = fetched_kv[:, :self.kv_lora_rank]
            kv_full = self.kv_up_proj(kv_latent)  # [attended, H*(nope+v)]

        # C2: Attention core — Q × K^T → softmax → × V
        with self._attn_score_timer:
            # Reshape for multi-head attention
            q_heads = q.view(
                num_tokens, self.num_heads,
                self.qk_nope_head_dim + self.qk_rope_head_dim,
            )
            # Split decompressed KV into K_nope and V
            kv_reshaped = kv_full.view(
                actual_attended, self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k_nope = kv_reshaped[:, :, :self.qk_nope_head_dim]
            v = kv_reshaped[:, :, self.qk_nope_head_dim:]

            # Compute attention scores: [T, H, 1] × [attended, H, nope]^T
            q_nope = q_heads[:, :, :self.qk_nope_head_dim]
            # Batched matmul: [H, T, nope] × [H, nope, attended] → [H, T, attended]
            attn_weights = torch.bmm(
                q_nope.permute(1, 0, 2),  # [H, T, nope]
                k_nope.permute(1, 2, 0),  # [H, nope, attended]
            )
            attn_weights = torch.softmax(attn_weights, dim=-1)
            # [H, T, attended] × [H, attended, v_dim] → [H, T, v_dim]
            attn_output = torch.bmm(
                attn_weights,
                v.permute(1, 0, 2),  # [H, attended, v_dim]
            )
            # Reshape back: [T, H*v_dim]
            attn_output = attn_output.permute(1, 0, 2).reshape(
                num_tokens, self.o_proj_in_dim
            )

        # C3: Output projection
        with self._o_proj_timer:
            output = self.o_proj(attn_output)  # [T, hidden]

        return output


class DSAModel(torch.nn.Module):
    """Wraps DSA attention in norm + residual for profiling.

    Mirrors the actual transformer block structure:
    hidden → LayerNorm → DSA Attention → residual add → output
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
        max_seq_len: int = 4096,
        num_repeat_steps: int = 1,
    ):
        super().__init__()
        self.num_repeat_steps = num_repeat_steps

        self.norm = torch.nn.LayerNorm(hidden_size)
        self.attn = DSAAttention(
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
        )

        self._norm_timer = CudaTimer("dsa_block_norm")
        self._residual_timer = CudaTimer("dsa_block_residual")

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        for _ in range(self.num_repeat_steps):
            residual = hidden_states
            with self._norm_timer:
                hidden_states = self.norm(hidden_states)
            hidden_states = self.attn(hidden_states, seq_len=seq_len)
            with self._residual_timer:
                hidden_states = hidden_states + residual
        return hidden_states
