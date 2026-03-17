# DeepSeek Sparse Attention (DSA) Profiler

This document describes the DSA profiler extension to Vidur's sparse model profiling infrastructure. The profiler collects empirical, per-operation timing data for the full Dynamic Sparse Attention decode pipeline used in DeepSeek-V3.

## Table of Contents

- [Overview](#overview)
- [Background: Dynamic Sparse Attention](#background-dynamic-sparse-attention)
- [DSA Pipeline Architecture](#dsa-pipeline-architecture)
  - [Block A: Q Projection](#block-a-q-projection)
  - [Block B: KV Indexing](#block-b-kv-indexing)
  - [Block C: Attention Compute](#block-c-attention-compute)
  - [Block Scheduling: A ∥ B → C](#block-scheduling-a--b--c)
- [Profiler Implementation](#profiler-implementation)
  - [File Structure](#file-structure)
  - [DSA Attention Implementation](#dsa-attention-implementation)
  - [DSA Profiling Wrapper](#dsa-profiling-wrapper)
  - [Integration with Sparse Profiler](#integration-with-sparse-profiler)
- [Profiled Operations](#profiled-operations)
- [Output Format](#output-format)
- [Usage](#usage)
- [Design Decisions](#design-decisions)

---

## Overview

Standard Multi-head Latent Attention (MLA) attends to all KV cache tokens during decode, making its cost linear in sequence length. For long-context inference (128K–1M+ tokens), this becomes the dominant bottleneck. DeepSeek-V3 addresses this with **Dynamic Sparse Attention (DSA)**: a lightweight indexer identifies the most relevant tokens, and full attention is computed only on a sparse subset.

The DSA profiler extends Vidur's existing sparse profiling infrastructure (`vidur/profiling/sparse/`) to empirically measure every operation in the DSA pipeline. This replaces the purely analytical estimates in `experiments/run_deepseek_dsa_analytical.py` with real GPU measurements.

---

## Background: Dynamic Sparse Attention

In a standard MLA decode step, the model:
1. Computes Q from the current token via MLA projections
2. Reads the full KV cache (all `seq_len` tokens)
3. Runs attention: Q × K^T → softmax → × V

At 128K sequence length with MLA, the KV cache read alone is ~140 MB per layer per step. At 1M tokens, it's ~1.1 GB per layer — far exceeding the time spent on compute.

DSA reduces this by:
1. Maintaining a **compressed indexer K cache** (FP8, `kv_lora_rank` dims per token) — a lightweight copy of the keys used only for scoring
2. Scoring all tokens cheaply using the compressed Q × indexer K matmul
3. **Selecting only the top-k** most relevant tokens (e.g., 2048)
4. **Fetching full MLA KV** only for selected tokens + a sliding window (e.g., 512)
5. Running full attention on this sparse subset (~2560 tokens regardless of seq_len)

This makes decode cost approximately constant in sequence length (after the indexer matmul and top-k, which are lightweight).

---

## DSA Pipeline Architecture

Each DSA decode layer executes three blocks:

```
  Block A (Q Projection)  ─┐
                            ├── run in parallel ──→ Block C (Attention + Output)
  Block B (KV Indexing)   ─┘
```

Layer time = max(Block A, Block B) + Block C

### Block A: Q Projection

Computes the full query vector for the current decode token using MLA's LoRA-style compression:

| Step | Operation | Shape | Description |
|------|-----------|-------|-------------|
| A1 | `q_down_proj` | [T, 7168] × [7168, 1536] | Compress hidden state to Q latent |
| A2 | `q_up_proj` | [T, 1536] × [1536, 24576] | Decompress Q latent to full Q |
| A3 | RoPE | element-wise on [T, 8192] | Apply rotary embeddings to rope portion |

At BS=1 decode, these GEMMs are entirely memory-bound (reading weights from HBM).

### Block B: KV Indexing

The sparse token selection pipeline — this is what makes DSA different from standard MLA:

| Step | Operation | Shape | Description |
|------|-----------|-------|-------------|
| B1 | Indexer K load | `seq_len × 512` bytes (FP8) | Read compressed indexer keys from HBM |
| B2 | Indexer matmul | [T, 512] × [512, seq_len] | Score all tokens: Q_compressed × indexer_K^T |
| B3 | Top-k select | seq_len → 2048 | GPU parallel partial sort to find best tokens |
| B4 | KV fetch | 2560 × 1152 bytes | Scattered gather of full MLA KV for selected tokens |

**B1** scales linearly with `seq_len` — at 128K tokens, the indexer K cache is 64 MB (FP8). At 1M tokens, it's 512 MB. This is the operation that makes DSA cost grow with context length, though it's much cheaper than reading full KV.

**B2** is a small matmul at BS=1 (dominated by the B1 read time).

**B3** uses GPU parallel top-k, approximately O(seq_len) with ~1 ms per 1M elements.

**B4** is a scattered gather — reads are non-contiguous, so effective bandwidth is lower than sequential reads. The total data volume is fixed at ~2560 × 1152 = 2.88 MB regardless of seq_len.

### Block C: Attention Compute

Runs after both A and B complete. Operates only on the sparse subset:

| Step | Operation | Shape | Description |
|------|-----------|-------|-------------|
| C1 | `kv_up_proj` | [2560, 512] × [512, 32768] | Decompress fetched KV latents to full K, V |
| C2 | Attention core | [H, T, 2560] matmul | Q × K^T → softmax → × V (sparse, 128 heads) |
| C3 | `o_proj` | [T, 16384] × [16384, 7168] | Output projection |
| C4 | Residual add | element-wise on [T, 7168] | Add residual connection |

**C1** is a large GEMM that becomes compute-bound at the 2560-token attended size — this is where DSA's compute cost concentrates.

### Block Scheduling: A ∥ B → C

Blocks A and B are independent and run in parallel. Block C depends on both:
- It needs Q from Block A
- It needs the fetched KV subset from Block B

The critical path is: `max(A, B) + C`. At short seq_lens, A dominates (B is fast). At long seq_lens, B dominates (indexer K cache read grows).

---

## Profiler Implementation

### File Structure

```
vidur/profiling/sparse/
├── dsa_attention_impl.py       # DSA model: DSAAttention + DSAModel with CudaTimers
├── dsa_attention_wrapper.py    # Profiling wrapper: manages warmup/measurement cycles
├── main.py                     # Entry point (updated to include DSA as step 3/4)
├── mla_attention_impl.py       # Existing MLA profiler (standard attention)
├── mla_attention_wrapper.py    # Existing MLA wrapper
├── sparse_mlp_impl.py          # Existing MoE MLP profiler
├── sparse_mlp_wrapper.py       # Existing MoE MLP wrapper
└── io_profiler.py              # Existing IO bandwidth profiler
```

### DSA Attention Implementation

**File:** `vidur/profiling/sparse/dsa_attention_impl.py`

Two classes:

**`DSAAttention(nn.Module)`** — The core DSA decode block with 11 CudaTimer-instrumented operations:

```python
# Block A timers
self._q_down_proj_timer    = CudaTimer("dsa_q_down_proj")
self._q_up_proj_timer      = CudaTimer("dsa_q_up_proj")
self._rope_timer           = CudaTimer("dsa_rope")

# Block B timers
self._indexer_load_timer   = CudaTimer("dsa_indexer_load")
self._indexer_matmul_timer = CudaTimer("dsa_indexer_matmul")
self._topk_timer           = CudaTimer("dsa_topk_select")
self._kv_fetch_timer       = CudaTimer("dsa_kv_fetch")

# Block C timers
self._kv_up_proj_timer     = CudaTimer("dsa_kv_up_proj")
self._attn_score_timer     = CudaTimer("dsa_attn_score")
self._o_proj_timer         = CudaTimer("dsa_o_proj")
```

Plus block-level timers (`dsa_block_norm`, `dsa_block_residual`) in the wrapping `DSAModel`.

Key design: `forward()` takes both `hidden_states` and `seq_len` as arguments. Unlike MLA profiling (which sweeps over `num_tokens` with KV cache = num_tokens), DSA profiling independently varies `seq_len` to measure how indexer cost scales with context length.

**`DSAModel(nn.Module)`** — Wraps DSAAttention with LayerNorm + residual, matching the actual transformer block structure.

### DSA Profiling Wrapper

**File:** `vidur/profiling/sparse/dsa_attention_wrapper.py`

**`DSAAttentionWrapper`** — Manages the profiling lifecycle:
1. Instantiates `DSAModel` with dummy weights on GPU (FP16)
2. For each `(num_tokens, seq_len)` pair:
   - Runs warmup steps (2 for CUDA events, 1 for record_function)
   - Clears timer statistics
   - Runs active measurement steps (10 iterations)
   - Collects per-operation timing stats (min/max/mean/median/std)
3. Returns a dict with timing stats + full config metadata

Supports both single-GPU (`--disable_ray`) and multi-GPU (Ray-distributed) modes.

### Integration with Sparse Profiler

**File:** `vidur/profiling/sparse/main.py`

The DSA profiler is integrated as step 3 of the 4-step sparse profiling pipeline:

```
[1/4] MoE MLP profiling          → sparse_mlp.csv
[2/4] MLA attention profiling    → mla_attention.csv
[3/4] DSA sparse attention       → dsa_attention.csv     ← NEW
[4/4] IO bandwidth profiling     → io_bandwidth.csv
```

DSA profiling is enabled when the model config has `"has_dsa": True` (currently DeepSeek-V3 only). It can be skipped with `--skip_dsa`.

The profiler sweeps over:
- **seq_len**: Logarithmic sweep (1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, ...) up to `--max_tokens`
- **num_tokens**: Batch sizes [1, 4, 16] to capture both single-decode and batch-decode behavior

Custom seq_len sweeps can be provided via `--dsa_seq_lens`.

---

## Profiled Operations

| Timer Name | Block | Operation | Scales With |
|------------|-------|-----------|-------------|
| `dsa_q_down_proj` | A | Q compression GEMM | num_tokens (fixed weight) |
| `dsa_q_up_proj` | A | Q decompression GEMM | num_tokens (fixed weight) |
| `dsa_rope` | A | Rotary position embedding | num_tokens |
| `dsa_indexer_load` | B | HBM read of indexer K cache | **seq_len** (linear) |
| `dsa_indexer_matmul` | B | Q × indexer_K^T scoring | seq_len × kv_lora_rank |
| `dsa_topk_select` | B | GPU parallel top-k | seq_len |
| `dsa_kv_fetch` | B | Scattered KV gather | dsa_attended (constant) |
| `dsa_kv_up_proj` | C | KV decompression GEMM | dsa_attended (constant) |
| `dsa_attn_score` | C | Multi-head attention | dsa_attended × num_heads |
| `dsa_o_proj` | C | Output projection GEMM | num_tokens (fixed weight) |
| `dsa_block_norm` | — | LayerNorm | num_tokens |
| `dsa_block_residual` | — | Residual addition | num_tokens |

---

## Output Format

The profiler outputs `dsa_attention.csv` with columns:

**Timing columns** (per operation, 5 stats each):
```
time_stats.dsa_q_down_proj.{min,max,mean,median,std}
time_stats.dsa_q_up_proj.{min,max,mean,median,std}
time_stats.dsa_rope.{min,max,mean,median,std}
time_stats.dsa_indexer_load.{min,max,mean,median,std}
time_stats.dsa_indexer_matmul.{min,max,mean,median,std}
time_stats.dsa_topk_select.{min,max,mean,median,std}
time_stats.dsa_kv_fetch.{min,max,mean,median,std}
time_stats.dsa_kv_up_proj.{min,max,mean,median,std}
time_stats.dsa_attn_score.{min,max,mean,median,std}
time_stats.dsa_o_proj.{min,max,mean,median,std}
time_stats.dsa_block_norm.{min,max,mean,median,std}
time_stats.dsa_block_residual.{min,max,mean,median,std}
```

**Config columns:**
```
hidden_size, num_heads, kv_lora_rank, q_lora_rank,
qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
dsa_selected_tokens, dsa_sliding_window, dsa_attended,
num_tokens, seq_len
```

Each row represents one `(num_tokens, seq_len)` measurement point.

---

## Usage

### Basic DSA profiling (DeepSeek-V3)
```bash
python -m vidur.profiling.sparse.main \
    --models deepseek-ai/DeepSeek-V3 \
    --max_tokens 131072 \
    --disable_ray
```

### DSA-only profiling with custom seq_lens
```bash
python -m vidur.profiling.sparse.main \
    --models deepseek-ai/DeepSeek-V3 \
    --skip_moe_mlp --skip_mla --skip_io \
    --dsa_seq_lens 4096 32768 131072 524288 1048576 \
    --disable_ray
```

### Override DSA parameters
```bash
python -m vidur.profiling.sparse.main \
    --models deepseek-ai/DeepSeek-V3 \
    --dsa_selected_tokens 4096 \
    --dsa_sliding_window 1024 \
    --max_tokens 131072 \
    --disable_ray
```

### Multi-GPU profiling with Ray
```bash
python -m vidur.profiling.sparse.main \
    --models deepseek-ai/DeepSeek-V3 \
    --max_tokens 131072 \
    --num_gpus 4
```

---

## Design Decisions

### Why a separate profiler from MLA?

The existing MLA profiler (`mla_attention_impl.py`) sweeps `num_tokens` and treats the KV cache as having the same length as the input sequence. This matches prefill behavior but doesn't capture DSA's decode-time dynamics where:

1. `seq_len` independently controls the indexer K cache size (Block B cost)
2. The attended token count is **fixed** at `dsa_selected + sliding_window`, regardless of `seq_len`
3. Blocks A and B run in parallel, so the critical path depends on which dominates

A separate profiler with a `(num_tokens, seq_len)` sweep space captures these dynamics.

### Indexer K simulation

The indexer K cache load (B1) is simulated with `torch.empty() + copy_()` to force an actual HBM read of the correct size. A simple `torch.randn()` would involve both allocation and compute (RNG), overstating the cost. The copy simulates the sequential HBM read pattern of loading cached indexer keys.

### Scattered gather simulation

The KV fetch (B4) uses `kv_cache[topk_indices]` — a gather operation with non-contiguous access patterns. This captures the real cache-unfriendly access pattern of sparse KV selection, unlike a simple contiguous read which would overestimate effective bandwidth.

### Attention core fidelity

Unlike the MLA profiler (which simulates attention with `torch.randn()`), the DSA profiler runs actual multi-head attention: `Q × K^T → softmax → × V`. This captures the true compute and memory access cost of the 128-head, 2560-token sparse attention, which is significant in Block C.
