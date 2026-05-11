"""Pure-analytical per-layer cost for DeepSeek-V3 DSA decode.

Bypasses the H100 profile CSVs to keep every number derivable from
first principles (HBM bandwidth + FLOPS-peak + bytes). Used by the
parallelism experiments where we need a stable ground truth.

Conventions
-----------
At BS=1 decode, every dense GEMM is memory-bound (just the weight
HBM read). We model each piece as:

    t_piece = max(t_compute, t_memory)
    t_compute = FLOPs / (peak_TFLOPS × MFU)            (peak FP16)
    t_memory  = bytes / HBM_BW                          (plus floor)

For the attention path, the *KV cache* is a separate IO stream (PCIe
in offload, HBM in HBM mode) and is *not* counted in the per-layer
compute. The compute is bracketed in three blocks:
    block_a = pre_norm + q_down + q_up + rope + kv_down (on new token)
    block_c = kv_up_proj (on attended 2560 tokens) + attn core
              + o_proj + residual
    moe     = norm + router + (dispatch + expert + combine)
              + shared_expert + residual

TP scaling
----------
- TP shards heads (attention) and intermediate dim (MLP). At BS=1
  every shardable GEMM is HBM-read-bound; sharding the weight
  divides its HBM read by TP.
- MoE expert GEMM is sharded by EP (not TP). TP doesn't help it.
- NVLink dispatch/combine cost in NVLink-message-count units; we
  treat as O(NUM_EXPERTS_PER_TOK) latency, unaffected by TP.
- MLA latent KV cache is *replicated* across the TP group
  (duplication tax). Counted only once on the IO stream.
"""

from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────
# H100 SXM constants
# ─────────────────────────────────────────────────────────────────────
HBM_BW_GBPS  = 1384.0        # peak measured HBM bandwidth
PCIE_BW_GBPS = 51.5          # Gen4 x16 unidirectional
HBM_FLOOR_MS = 0.015         # ~16 MB transition to bandwidth-bound
PCIE_FLOOR_MS = 0.020
NVLINK_LATENCY_US = 5.0      # per message

# FP16 tensor-core peak with conservative MFU for GEMMs at BS=1
H100_FP16_TFLOPS = 989.5
H100_MFU         = 0.50

GB = 1024 ** 3
MB = 1024 ** 2

# ─────────────────────────────────────────────────────────────────────
# Precision (default FP16 baseline; FP8 path used by *_fp8 variants)
# ─────────────────────────────────────────────────────────────────────
# Weight bytes per element.
W_BYTES_FP16 = 2
W_BYTES_FP8  = 1
# MLA KV bytes per element (latent + RoPE). DeepSeek-V3.2-Exp uses FP8
# for the latent, with a small FP16 footprint for the decoupled RoPE
# key. We model FP8 KV as 1 byte/elem for simplicity (576 B/token).
KV_BYTES_FP16 = 2
KV_BYTES_FP8  = 1

# ─────────────────────────────────────────────────────────────────────
# DeepSeek-V3 architecture constants
# ─────────────────────────────────────────────────────────────────────
NUM_LAYERS          = 61
HIDDEN              = 7168
NUM_HEADS           = 128
Q_LORA_RANK         = 1536
KV_LORA_RANK        = 512
QK_NOPE             = 128
QK_ROPE             = 64
V_HEAD              = 128
Q_TOTAL_DIM         = NUM_HEADS * (QK_NOPE + QK_ROPE)        # 24576
KV_UP_OUT_DIM       = NUM_HEADS * (QK_NOPE + V_HEAD)         # 32768
O_PROJ_IN_DIM       = NUM_HEADS * V_HEAD                     # 16384

NUM_ROUTED_EXPERTS  = 256
NUM_EXPERTS_PER_TOK = 8
EXPERT_INTERMEDIATE = 2048
EXPERT_WEIGHT_BYTES = 3 * HIDDEN * EXPERT_INTERMEDIATE * 2   # 84 MB / expert / layer

# DSA selection
DSA_TOPK             = 2048
DSA_SLIDING          = 512
DSA_ATTENDED         = DSA_TOPK + DSA_SLIDING                # 2560

# Cache bytes per token per layer
MLA_KV_BYTES_PER_TOKEN  = (KV_LORA_RANK + QK_ROPE) * 2       # 1152 B (FP16)
INDEXER_K_BYTES_PER_TOKEN = KV_LORA_RANK * 1                 # 512 B (FP8)


# ─────────────────────────────────────────────────────────────────────
# Primitive analytical helpers
# ─────────────────────────────────────────────────────────────────────
def hbm_ms(bytes_: int) -> float:
    return max(bytes_ / (HBM_BW_GBPS * GB) * 1000.0, HBM_FLOOR_MS)


def pcie_ms(bytes_: int) -> float:
    return max(bytes_ / (PCIE_BW_GBPS * GB) * 1000.0, PCIE_FLOOR_MS)


def io_ms(bytes_: int, mode: str) -> float:
    return pcie_ms(bytes_) if mode == "offload" else hbm_ms(bytes_)


def nvlink_ms(num_messages: int) -> float:
    return num_messages * NVLINK_LATENCY_US / 1000.0


def gemm_ms(m: int, k: int, n: int) -> float:
    """Analytical FP16 GEMM: max(compute, HBM-weight-read)."""
    flops = 2 * m * k * n
    weights = k * n * 2  # FP16 weight bytes
    acts    = (m * k + m * n) * 2
    compute_ms = (flops / 1e9) / (H100_FP16_TFLOPS * 1024 * H100_MFU) * 1000
    memory_ms  = hbm_ms(weights + acts)
    return max(compute_ms, memory_ms)


def gemm_tp_ms(m: int, k: int, n: int, tp: int, axis: str = "n") -> float:
    """TP-sharded GEMM.

    axis='n' → split output dim n by TP; each rank does [m, k] × [k, n/tp].
              Weight bytes /tp, FLOPs /tp.
    axis='k' → split inner dim k by TP; each rank does [m, k/tp] × [k/tp, n].
              Weight bytes /tp, FLOPs /tp. Requires all-reduce on output (m*n*2 B).
    """
    if axis == "n":
        return gemm_ms(m, k, n // tp)
    elif axis == "k":
        compute = gemm_ms(m, k // tp, n)
        ar_bytes = m * n * 2
        # Conservative all-reduce time: 2*(N-1)/N × bytes / NVLink BW
        ar_ms = 2 * (tp - 1) / max(tp, 1) * ar_bytes / (600 * GB) * 1000.0
        return compute + ar_ms
    raise ValueError(axis)


def moe_expert_gemm_ms(bs: int, ep: int) -> float:
    """Grouped-MoE expert GEMM at decode.

    With EP=ep, each rank holds 256/ep experts. With 8 active experts
    per token, average 8/ep active experts per rank per token. For BS
    tokens, average unique experts touched per rank:
        E[unique] = local_experts × (1 − (1 − 1/local_experts)^bs)
    where local_experts = 256 / ep.

    Each touched expert reads its 84 MB weight once (grouped GEMM):
        memory_ms = unique × 84 MB / HBM_BW
        compute_ms = BS × 3 × 2 × HIDDEN × INTERMEDIATE / peak

    NOT TP-shardable.
    """
    local = NUM_ROUTED_EXPERTS // ep
    # Probability a given expert on this rank is touched at least once:
    p_active = NUM_EXPERTS_PER_TOK / NUM_ROUTED_EXPERTS  # per-token-per-expert
    unique = local * (1 - (1 - p_active) ** bs)
    weight_bytes = int(unique * EXPERT_WEIGHT_BYTES)
    flops = bs * 3 * 2 * HIDDEN * EXPERT_INTERMEDIATE
    compute_ms = (flops / 1e9) / (H100_FP16_TFLOPS * 1024 * H100_MFU) * 1000
    return max(hbm_ms(weight_bytes), compute_ms)


# ─────────────────────────────────────────────────────────────────────
# Per-piece analytical decode-layer costs
# ─────────────────────────────────────────────────────────────────────
@dataclass
class AnalyticalLayerCost:
    """Decode-layer time decomposition (TP=1 baseline).

    All times in ms. The IO pieces (idx_io, kv_io) are placement-
    dependent; compute pieces are TP-sensitive and split here so the
    caller can apply /TP to the shardable subset.
    """
    # IO stream (block B)
    idx_io_ms: float            # F-layer indexer K cache read
    kv_io_ms: float             # MLA KV gather (2560 tokens)
    # Compute pieces — TP-shardable
    block_a_ms: float           # q_down + q_up + rope + norm + kv_down (new tok)
    block_c_ms: float           # kv_up + attn-core + o_proj + residual
    idx_comp_ms: float          # indexer matmul + topk (F-layer only)
    moe_shardable_ms: float     # norm + router_gate + softmax + topk + shared_exp + residual
    # Compute pieces — NOT TP-shardable
    moe_expert_gemm_ms: float   # EP-sharded grouped GEMM
    moe_ep_comms_ms: float      # NVLink dispatch + combine

    kind: str = "F"             # "F" or "S"

    def total_io_ms(self) -> float:
        return self.idx_io_ms + self.kv_io_ms

    def total_compute_ms(self, tp: int = 1) -> float:
        shardable = (self.block_a_ms + self.block_c_ms
                     + self.idx_comp_ms + self.moe_shardable_ms)
        # Within-layer block_a||io overlap: block_a is hidden in io if shorter
        block_a_exposed = max(0.0, self.block_a_ms / tp - self.total_io_ms())
        non_block_a_shardable = (self.block_c_ms + self.idx_comp_ms
                                 + self.moe_shardable_ms)
        return (block_a_exposed
                + non_block_a_shardable / tp
                + self.moe_expert_gemm_ms
                + self.moe_ep_comms_ms)


def analytical_layer_F(seq_len: int, bs: int, mode: str, ep: int = 8,
                       fp8: bool = False) -> AnalyticalLayerCost:
    """One F (Full) decode layer at (seq_len, bs, mode), TP=1 baseline.

    Hand-checked at BS=1 sl=200K offload:
        idx_io  = 1 × 200K × 512 / 51.5 GBps ≈ 1.896 ms
        kv_io   = 1 × 2560 × 1152 / 51.5 GBps ≈ 0.053 ms
        block_a:
          q_down  : 7168×1536 × 2 = 22.0 MB / HBM ≈ 0.0156 ms
          q_up    : 1536×24576 × 2 = 75.5 MB / HBM ≈ 0.0533 ms
          rope    : trivial   ≈ 0.015 ms (floor)
          pre_norm: trivial   ≈ 0.015 ms (floor)
          → block_a ≈ 0.0986 ms
        block_c:
          kv_up   : 2560×512×32768 (GEMM on attended) → 161 MB read ≈ 0.114 ms
                   plus weight read 512×32768×2 = 32 MB ≈ 0.023 ms; max(comp, mem) ≈ 0.144 ms
          attn    : 2560 × 32768 × 2 = 160 MB / HBM ≈ 0.114 ms
          o_proj  : 16384×7168 × 2 = 224 MB / HBM ≈ 0.158 ms
          residual: floor 0.015 ms
          → block_c ≈ 0.43 ms
        idx_comp = 0.005 ms (kernel floor) + 0.005 max(1, bs) (topk) = 0.01 ms
        moe_shardable:
          norm     : floor 0.015
          router   : 7168×256 × 2 = 3.5 MB / HBM ≈ 0.015 (floor)
          shared_e : 7168×6144 × 2 = 84 MB / HBM ≈ 0.058 ms
          softmax+topk+residual: floors ~ 0.025
          → ~0.113 ms
        moe_expert_gemm: 84 MB / HBM ≈ 0.058 ms (1 active expert per rank avg)
        moe_ep_comms   : 2 × 8 × 5 us = 0.08 ms
    """
    # FP8 path: weights, KV cache, decompressed-KV activations halve.
    # Indexer K is already FP8 in both paths.
    w_scale = 0.5 if fp8 else 1.0
    kv_bytes = (KV_LORA_RANK + QK_ROPE) * (1 if fp8 else 2)  # 576 B or 1152 B per token

    # --- IO stream ---
    idx_io = io_ms(bs * seq_len * INDEXER_K_BYTES_PER_TOKEN, mode)
    kv_io  = io_ms(bs * DSA_ATTENDED * kv_bytes, mode)

    # --- Block A: Q projection on new BS token(s) (weights × w_scale) ---
    pre_norm = hbm_ms(bs * HIDDEN * 2 * 2)            # read + write activations (FP16)
    q_down   = gemm_ms(bs, HIDDEN, Q_LORA_RANK) * w_scale + hbm_ms(0) * (1 - w_scale)
    # The cleaner expression: GEMM time is max(compute, weight HBM read).
    # At BS=1 we're memory-bound on weights, so halving weights halves time.
    # We just multiply the result by w_scale (within the BS=1 regime which is
    # what all our scenarios fall into for non-attended-token GEMMs).
    q_down = gemm_ms(bs, HIDDEN, Q_LORA_RANK) * w_scale
    q_up   = gemm_ms(bs, Q_LORA_RANK, Q_TOTAL_DIM) * w_scale
    rope   = hbm_ms(bs * NUM_HEADS * QK_ROPE * 2 * 2)
    kv_down = gemm_ms(bs, HIDDEN, KV_LORA_RANK + QK_ROPE) * w_scale
    block_a = pre_norm + q_down + q_up + rope + kv_down

    # --- Block C: kv_up_proj on attended + attn-core + o_proj ---
    # kv_up_proj reads the decompressed KV (also FP8 in FP8 path) and writes
    # the heads-expanded form. At BS=1 the GEMM is memory-bound on weights too.
    kv_up = gemm_ms(bs * DSA_ATTENDED, KV_LORA_RANK, KV_UP_OUT_DIM) * w_scale
    decompressed_kv_bytes = bs * DSA_ATTENDED * KV_UP_OUT_DIM * (1 if fp8 else 2)
    attn_core = hbm_ms(decompressed_kv_bytes)
    o_proj    = gemm_ms(bs, O_PROJ_IN_DIM, HIDDEN) * w_scale
    residual  = hbm_ms(bs * HIDDEN * 2 * 2)
    block_c   = kv_up + attn_core + o_proj + residual

    # --- Indexer compute (F only) ---
    # FP8 indexer matmul + topk; floors only
    idx_comp = 0.005 + max(0.005, 0.005 * bs)

    # --- MoE shardable pieces ---
    moe_norm     = hbm_ms(bs * HIDDEN * 2 * 2)
    router_gate  = gemm_ms(bs, HIDDEN, NUM_ROUTED_EXPERTS) * w_scale
    router_sm    = max(0.005, 0.001 * bs)
    router_topk  = max(0.005, 0.04 * bs)
    shared_exp_w = 3 * HIDDEN * EXPERT_INTERMEDIATE * (1 if fp8 else 2)
    shared_exp_gemm_t = gemm_ms(bs, HIDDEN, 3 * EXPERT_INTERMEDIATE) * w_scale
    shared_exp   = max(shared_exp_gemm_t, hbm_ms(shared_exp_w))
    moe_resid    = hbm_ms(bs * HIDDEN * 2 * 2)
    moe_shardable = (moe_norm + router_gate + router_sm + router_topk
                     + shared_exp + moe_resid)

    # --- MoE expert GEMM (EP-only). FP8 halves the weight read. ---
    moe_expert_gemm = moe_expert_gemm_ms(bs, ep) * w_scale

    # --- MoE EP NVLink comms (dispatch + combine, two passes) ---
    # Each pass: NUM_EXPERTS_PER_TOK messages of size BS × HIDDEN × 2 B
    # Latency-dominated at BS=1; bandwidth-dominated at high BS.
    msg_bytes = bs * HIDDEN * 2
    msg_count = NUM_EXPERTS_PER_TOK
    bw_ms = msg_count * msg_bytes / (600 * GB) * 1000  # NVLink ~600 GB/s
    moe_ep_comms = 2 * (nvlink_ms(msg_count) + bw_ms)

    return AnalyticalLayerCost(
        idx_io_ms=idx_io, kv_io_ms=kv_io,
        block_a_ms=block_a, block_c_ms=block_c, idx_comp_ms=idx_comp,
        moe_shardable_ms=moe_shardable,
        moe_expert_gemm_ms=moe_expert_gemm,
        moe_ep_comms_ms=moe_ep_comms,
        kind="F",
    )


def analytical_layer_S(seq_len: int, bs: int, mode: str, ep: int = 8,
                       fp8: bool = False) -> AnalyticalLayerCost:
    """One S (Shared) decode layer. Identical to F minus idx_io and idx_comp.

    The MLA-KV gather still happens — S layers attend to the F layer's
    selected top-k positions, so they still need those 2560 tokens of KV
    decompressed.
    """
    f = analytical_layer_F(seq_len, bs, mode, ep, fp8=fp8)
    return AnalyticalLayerCost(
        idx_io_ms=0.0, kv_io_ms=f.kv_io_ms,
        block_a_ms=f.block_a_ms, block_c_ms=f.block_c_ms, idx_comp_ms=0.0,
        moe_shardable_ms=f.moe_shardable_ms,
        moe_expert_gemm_ms=f.moe_expert_gemm_ms,
        moe_ep_comms_ms=f.moe_ep_comms_ms,
        kind="S",
    )


# ─────────────────────────────────────────────────────────────────────
# Memory accounting (per rank)
# ─────────────────────────────────────────────────────────────────────
# Total non-expert weights per layer (FP16):
#   q_down + q_up + kv_down + kv_up + o_proj + indexer_K_proj
#   + moe_router_gate + shared_expert
NON_EXPERT_W_PER_LAYER_B = (
    HIDDEN * Q_LORA_RANK
    + Q_LORA_RANK * NUM_HEADS * (QK_NOPE + QK_ROPE)
    + HIDDEN * KV_LORA_RANK
    + KV_LORA_RANK * NUM_HEADS * (QK_NOPE + V_HEAD)
    + NUM_HEADS * V_HEAD * HIDDEN
    + HIDDEN * KV_LORA_RANK
    + HIDDEN * NUM_ROUTED_EXPERTS
    + 3 * HIDDEN * EXPERT_INTERMEDIATE
) * 2  # FP16

# Total expert weights per layer: all 256 experts FP16
EXPERT_W_PER_LAYER_B = NUM_ROUTED_EXPERTS * 3 * HIDDEN * EXPERT_INTERMEDIATE * 2


def per_rank_memory_bytes(pp: int, tp: int, ep: int,
                           seq_len: int, bs: int,
                           n_layers: int = NUM_LAYERS,
                           fp8: bool = False) -> dict:
    """Per-rank HBM footprint with the MLA duplication tax explicit.

    - PP shards layers (KV/IDX caches and weights scale /PP).
    - TP shards non-expert weights /TP. **Does NOT shard KV/IDX**
      (MLA duplication tax) — they stay per-rank as if TP=1.
    - EP shards expert weights /EP.
    """
    import math
    layers_per_stage = math.ceil(n_layers / pp)
    # FP8 halves weights and KV cache. Indexer K is FP8 in both paths.
    w_scale = 1 if not fp8 else 2  # divide weights by this; non-FP8 = no change
    kv_bytes_per_token = (KV_LORA_RANK + QK_ROPE) * (1 if fp8 else 2)
    kv  = layers_per_stage * bs * seq_len * kv_bytes_per_token
    idx = layers_per_stage * bs * seq_len * INDEXER_K_BYTES_PER_TOKEN
    non_expert_w = layers_per_stage * NON_EXPERT_W_PER_LAYER_B // tp // w_scale
    expert_w     = layers_per_stage * EXPERT_W_PER_LAYER_B // ep // w_scale
    return {
        "layers_per_stage": layers_per_stage,
        "kv": kv, "idx": idx,
        "non_expert_w": non_expert_w,
        "expert_w": expert_w,
        "total": kv + idx + non_expert_w + expert_w,
    }
