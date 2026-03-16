#!/usr/bin/env python3
"""DeepSeek-V3 DSA Decode: Purely analytical per-layer timing breakdown.

Computes execution time for one decode step through 61 layers using
the correct DSA pipeline structure:

  Block A (Q compute)  ─┐
                        ├─ run in parallel ─→ Block C (Attention + Output)
  Block B (KV index)   ─┘

Block A: Q projection (compute the query for this decode token)
  - q_down_proj: hidden → q_lora_rank
  - q_up_proj:   q_lora_rank → H * (nope + rope)
  - RoPE on Q rope portion

Block B: KV indexing pipeline (find which KV entries to attend to)
  - Read indexer K cache from HBM (FP8, all seq_len tokens)
  - Indexer matmul: Q_compressed × indexer_K^T (score all tokens)
  - Top-k selection (pick 2048 tokens)
  - Fetch full MLA KV for selected + sliding window tokens (gather)

Block C: Attention compute (runs after both A and B complete)
  - kv_up_proj: decompress fetched KV latents → full K, V
  - Attention: Q × K^T → softmax → × V (on 2560 tokens only)
  - o_proj: attention output → hidden

All times are purely analytical — no profiling data used.
"""

# ── DeepSeek-V3 config ──────────────────────────────────────────────
NUM_LAYERS = 61
HIDDEN_SIZE = 7168
NUM_HEADS = 128
KV_LORA_RANK = 512
Q_LORA_RANK = 1536
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128

# Derived
Q_TOTAL_DIM = NUM_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)  # 128*192 = 24576
KV_UP_OUT_DIM = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)       # 128*256 = 32768
O_PROJ_IN_DIM = NUM_HEADS * V_HEAD_DIM                             # 128*128 = 16384

# MLA KV cache: per token per layer
MLA_KV_BYTES_PER_TOKEN = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2  # 1152 bytes (FP16)

# DSA config
DSA_SELECTED_TOKENS = 2048
DSA_SLIDING_WINDOW = 512
DSA_ATTENDED = DSA_SELECTED_TOKENS + DSA_SLIDING_WINDOW  # 2560

# Indexer K: FP8, kv_lora_rank dims per token
INDEXER_K_BYTES_PER_TOKEN = KV_LORA_RANK * 1  # 512 bytes (FP8)

BYTES_PER_PARAM = 2  # FP16 weights

# MoE config
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_SHARED_EXPERTS = 1
EXPERT_INTERMEDIATE_SIZE = 2048
# 3 weight matrices per expert: gate_proj, up_proj, down_proj
EXPERT_WEIGHT_BYTES = 3 * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * BYTES_PER_PARAM  # ~84 MB
SHARED_EXPERT_WEIGHT_BYTES = EXPERT_WEIGHT_BYTES  # same architecture

# Parallelism
EP = 8   # expert parallelism: each GPU holds 256/8 = 32 experts
TP = 1   # tensor parallelism (for this single-GPU analytical model)

# ── H100 SXM specs ───────────────────────────────────────────────────
H100_HBM_BW_GBS = 3350.0       # H100 SXM HBM3 peak bandwidth (GB/s)
H100_HBM_BW_EFF_GBS = 2680.0   # ~80% efficiency at large transfers
H100_FP16_TFLOPS = 989.5       # FP16 Tensor Core peak
H100_FP8_TFLOPS = 1979.0       # FP8 Tensor Core peak
H100_KERNEL_LAUNCH_US = 5.0    # Typical CUDA kernel launch overhead (μs)
NVLINK_LATENCY_US = 5.0        # NVLink per-message latency (μs)


def gemm_flops(m, k, n):
    """FLOPs for a GEMM: 2*M*K*N."""
    return 2.0 * m * k * n


def gemm_weight_bytes(k, n):
    """Weight matrix size in bytes (FP16)."""
    return k * n * BYTES_PER_PARAM


def hbm_read_time_ms(size_bytes):
    """Analytical HBM read time in ms.

    At BS=1, GEMM is memory-bound: time = max(weight_read, kernel_launch).
    H100 SXM HBM3: 3.35 TB/s peak, ~80% utilization at large sizes.
    Small transfers are dominated by kernel launch overhead (~5 μs).
    """
    bandwidth_time_ms = (size_bytes / (H100_HBM_BW_EFF_GBS * 1024**3)) * 1e3
    kernel_launch_ms = H100_KERNEL_LAUNCH_US / 1000.0
    return max(bandwidth_time_ms, kernel_launch_ms)


def gemm_time_ms(m, k, n, dtype_bytes=BYTES_PER_PARAM, peak_tflops=H100_FP16_TFLOPS, mfu=0.5):
    """Analytical GEMM time: max(compute_bound, memory_bound).

    At BS=1 (m=1), GEMMs are memory-bound (reading weight matrix).
    At larger batch sizes, compute starts to dominate.
    """
    # Compute bound: FLOPs / throughput
    flops = gemm_flops(m, k, n)
    compute_ms = (flops / 1e9) / (peak_tflops * 1024 * mfu) * 1e3

    # Memory bound: read weights + activations from HBM
    weight_bytes = k * n * dtype_bytes
    activation_bytes = (m * k + m * n) * dtype_bytes  # input + output
    total_bytes = weight_bytes + activation_bytes
    memory_ms = hbm_read_time_ms(total_bytes)

    return max(compute_ms, memory_ms)


def elementwise_time_ms(num_elements, bytes_per_element=2):
    """Time for element-wise ops (RoPE, residual add, etc.).

    Memory-bound: read + write the tensor.
    """
    total_bytes = num_elements * bytes_per_element * 2  # read + write
    return hbm_read_time_ms(total_bytes)


def analyze_dsa_decode(seq_len):
    """Analytical DSA decode timing for one layer at BS=1."""

    # ═══════════════════════════════════════════════════════════════
    # BLOCK A: Q Projection (1 decode token)
    # ═══════════════════════════════════════════════════════════════
    # At BS=1, all GEMMs are memory-bound (reading weights from HBM)

    # q_down_proj: [1, 7168] × [7168, 1536] → [1, 1536]
    q_down_ms = gemm_time_ms(1, HIDDEN_SIZE, Q_LORA_RANK)
    q_down_weight_mb = gemm_weight_bytes(HIDDEN_SIZE, Q_LORA_RANK) / (1024**2)

    # q_up_proj: [1, 1536] × [1536, 24576] → [1, 24576]
    q_up_ms = gemm_time_ms(1, Q_LORA_RANK, Q_TOTAL_DIM)
    q_up_weight_mb = gemm_weight_bytes(Q_LORA_RANK, Q_TOTAL_DIM) / (1024**2)

    # RoPE: element-wise on rope portion of Q
    rope_elements = NUM_HEADS * QK_ROPE_HEAD_DIM  # 128 * 64 = 8192
    rope_ms = elementwise_time_ms(rope_elements)

    block_a_ms = q_down_ms + q_up_ms + rope_ms

    # ═══════════════════════════════════════════════════════════════
    # BLOCK B: KV Indexing Pipeline
    # ═══════════════════════════════════════════════════════════════

    # B1: Read indexer K cache from HBM (FP8, all tokens)
    indexer_k_bytes = INDEXER_K_BYTES_PER_TOKEN * seq_len
    indexer_k_mb = indexer_k_bytes / (1024**2)
    indexer_read_ms = hbm_read_time_ms(indexer_k_bytes)

    # B2: Indexer matmul (FP8): [1, 512] × [512, seq_len] → [1, seq_len]
    #     At BS=1 this is entirely memory-bound — time is dominated by
    #     reading the indexer K cache, which we already counted above.
    indexer_flops = gemm_flops(1, KV_LORA_RANK, seq_len)
    indexer_compute_ms = (indexer_flops / 1e9) / (H100_FP8_TFLOPS * 1024) * 1e3  # negligible

    # B3: Top-k selection (2048 from seq_len)
    #     GPU parallel partial sort. ~1 ms per 1M elements on H100.
    topk_ms = seq_len / 1e6 * 1.0

    # B4: Fetch full MLA KV for attended tokens (scattered gather from KV cache)
    fetch_kv_bytes = MLA_KV_BYTES_PER_TOKEN * DSA_ATTENDED
    fetch_kv_mb = fetch_kv_bytes / (1024**2)
    fetch_kv_ms = hbm_read_time_ms(fetch_kv_bytes)

    block_b_ms = indexer_read_ms + indexer_compute_ms + topk_ms + fetch_kv_ms

    # ═══════════════════════════════════════════════════════════════
    # BLOCK C: Attention Compute (after max(A, B))
    # ═══════════════════════════════════════════════════════════════

    # C1: KV decompression (kv_up_proj on 2560 fetched tokens)
    #     [2560, 512] × [512, 32768] — this is a large GEMM, compute-bound
    kv_up_ms = gemm_time_ms(DSA_ATTENDED, KV_LORA_RANK, KV_UP_OUT_DIM)
    kv_up_flops = gemm_flops(DSA_ATTENDED, KV_LORA_RANK, KV_UP_OUT_DIM)

    # C2: Attention core (1 query vs 2560 KV entries, 128 heads)
    #     Memory-bound: reading decompressed KV from HBM
    #     Decompressed KV size: 2560 * 32768 * 2 = 160 MB
    decompressed_kv_bytes = DSA_ATTENDED * KV_UP_OUT_DIM * BYTES_PER_PARAM
    decompressed_kv_mb = decompressed_kv_bytes / (1024**2)
    attn_core_ms = hbm_read_time_ms(decompressed_kv_bytes)

    # C3: Output projection (on 1 decode token)
    #     [1, 16384] × [16384, 7168] — memory-bound at BS=1
    o_proj_ms = gemm_time_ms(1, O_PROJ_IN_DIM, HIDDEN_SIZE)
    o_proj_weight_mb = gemm_weight_bytes(O_PROJ_IN_DIM, HIDDEN_SIZE) / (1024**2)

    # C4: Residual add
    residual_ms = elementwise_time_ms(HIDDEN_SIZE)

    block_c_ms = kv_up_ms + attn_core_ms + o_proj_ms + residual_ms

    # ═══════════════════════════════════════════════════════════════
    # MoE BLOCK (sequential after attention)
    #
    # Expert weights are PREFETCHED on each GPU. With EP=8, each GPU
    # holds 256/8 = 32 experts resident in HBM. No weight loading I/O.
    #
    # At BS=1, expert GEMMs are memory-bound (reading weights from HBM).
    # A proper grouped GEMM kernel batches all active experts on this
    # GPU into one kernel launch.
    #
    # Pipeline:
    #   1. Router: gate GEMM + softmax + top-k (on each GPU)
    #   2. EP dispatch: send token to GPUs holding selected experts
    #   3. Expert GEMM: each GPU runs its local experts (1 per token avg)
    #   4. EP combine: send results back
    #   5. Shared expert: runs on every GPU (dense GEMM, always active)
    # ═══════════════════════════════════════════════════════════════

    # Norm: LayerNorm on hidden_size
    moe_norm_ms = elementwise_time_ms(HIDDEN_SIZE)

    # Router: gate GEMM [1, 7168] × [7168, 256] + softmax + topk
    router_gate_ms = gemm_time_ms(1, HIDDEN_SIZE, NUM_ROUTED_EXPERTS)
    router_softmax_ms = H100_KERNEL_LAUNCH_US / 1000.0  # tiny op, kernel launch dominated
    router_topk_ms = H100_KERNEL_LAUNCH_US / 1000.0     # tiny op, kernel launch dominated
    moe_router_ms = router_gate_ms + router_softmax_ms + router_topk_ms

    # EP dispatch + combine: send 1 token hidden_state to/from expert GPUs
    # NVLink latency-bound at these small sizes (~5 μs per message)
    ep_dispatch_ms = NUM_EXPERTS_PER_TOK * NVLINK_LATENCY_US / 1000.0
    ep_combine_ms = NUM_EXPERTS_PER_TOK * NVLINK_LATENCY_US / 1000.0

    # Expert GEMM: with EP=8, each GPU runs ~1 expert per token (8 experts / 8 GPUs)
    # Each expert = 3 GEMMs (gate+up+down), BS=1, memory-bound on weight read
    # Weight size per expert: 3 × 7168 × 2048 × 2B = ~84 MB
    experts_per_gpu = NUM_EXPERTS_PER_TOK / EP  # 1.0 on average
    expert_weight_mb = EXPERT_WEIGHT_BYTES / (1024**2)
    moe_gemm_ms = hbm_read_time_ms(int(experts_per_gpu * EXPERT_WEIGHT_BYTES))

    # Shared expert: dense GEMM on 1 token, same shape as routed expert
    # Weight size: ~84 MB, BS=1, memory-bound
    moe_shared_ms = hbm_read_time_ms(SHARED_EXPERT_WEIGHT_BYTES)

    # Residual add
    moe_residual_ms = elementwise_time_ms(HIDDEN_SIZE)

    moe_total_ms = (moe_norm_ms + moe_router_ms + ep_dispatch_ms +
                    moe_gemm_ms + ep_combine_ms + moe_shared_ms + moe_residual_ms)

    # ═══════════════════════════════════════════════════════════════
    # TOTAL per layer
    # ═══════════════════════════════════════════════════════════════
    parallel_ab_ms = max(block_a_ms, block_b_ms)
    attention_total_ms = parallel_ab_ms + block_c_ms
    layer_total_ms = attention_total_ms + moe_total_ms

    return {
        "seq_len": seq_len,
        # Block A
        "q_down_ms": q_down_ms,
        "q_up_ms": q_up_ms,
        "rope_ms": rope_ms,
        "block_a_ms": block_a_ms,
        "q_down_weight_mb": q_down_weight_mb,
        "q_up_weight_mb": q_up_weight_mb,
        # Block B
        "indexer_k_mb": indexer_k_mb,
        "indexer_read_ms": indexer_read_ms,
        "indexer_compute_ms": indexer_compute_ms,
        "topk_ms": topk_ms,
        "fetch_kv_mb": fetch_kv_mb,
        "fetch_kv_ms": fetch_kv_ms,
        "block_b_ms": block_b_ms,
        # Block C
        "kv_up_ms": kv_up_ms,
        "kv_up_gflops": kv_up_flops / 1e9,
        "decompressed_kv_mb": decompressed_kv_mb,
        "attn_core_ms": attn_core_ms,
        "o_proj_ms": o_proj_ms,
        "o_proj_weight_mb": o_proj_weight_mb,
        "residual_ms": residual_ms,
        "block_c_ms": block_c_ms,
        # MoE
        "moe_norm_ms": moe_norm_ms,
        "moe_router_ms": moe_router_ms,
        "ep_dispatch_ms": ep_dispatch_ms,
        "moe_gemm_ms": moe_gemm_ms,
        "expert_weight_mb": expert_weight_mb,
        "experts_per_gpu": experts_per_gpu,
        "ep_combine_ms": ep_combine_ms,
        "moe_shared_ms": moe_shared_ms,
        "moe_residual_ms": moe_residual_ms,
        "moe_total_ms": moe_total_ms,
        # Totals
        "parallel_ab_ms": parallel_ab_ms,
        "attention_total_ms": attention_total_ms,
        "layer_total_ms": layer_total_ms,
        "all_layers_ms": layer_total_ms * NUM_LAYERS,
    }


def print_breakdown(r):
    sl = r["seq_len"]
    sl_label = f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"

    print(f"\n{'=' * 72}")
    print(f" DSA Decode — 1 Layer — BS=1 — seq_len={sl_label} ({sl:,} tokens)")
    print(f"{'=' * 72}")

    print(f"\n  BLOCK A: Q Projection (1 token, parallel with B)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ q_down_proj  [1,7168]×[7168,1536]   weights={r['q_down_weight_mb']:.1f}MB   {r['q_down_ms']:.4f} ms")
    print(f"  │ q_up_proj    [1,1536]×[1536,24576]  weights={r['q_up_weight_mb']:.1f}MB  {r['q_up_ms']:.4f} ms")
    print(f"  │ RoPE         element-wise                           {r['rope_ms']:.4f} ms")
    print(f"  └─── Block A total: {r['block_a_ms']:.4f} ms")

    print(f"\n  BLOCK B: KV Indexing Pipeline (parallel with A)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ Read indexer K   FP8, {sl_label} tokens   {r['indexer_k_mb']:.1f} MB   {r['indexer_read_ms']:.4f} ms")
    print(f"  │ Indexer matmul   [1,512]×[512,{sl}]  FP8       {r['indexer_compute_ms']:.6f} ms")
    print(f"  │ Top-k select     {sl} → 2048                     {r['topk_ms']:.4f} ms")
    print(f"  │ Fetch MLA KV     2560 tokens (gather)  {r['fetch_kv_mb']:.1f} MB   {r['fetch_kv_ms']:.4f} ms")
    print(f"  └─── Block B total: {r['block_b_ms']:.4f} ms")

    print(f"\n  ── max(A, B) = {r['parallel_ab_ms']:.4f} ms ──")

    print(f"\n  BLOCK C: Attention + Output (after A∥B)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ kv_up_proj   [2560,512]×[512,32768]  {r['kv_up_gflops']:.1f} GFLOPs  {r['kv_up_ms']:.4f} ms")
    print(f"  │ Attn core    1 Q × 2560 KV  read {r['decompressed_kv_mb']:.0f}MB   {r['attn_core_ms']:.4f} ms")
    print(f"  │ o_proj       [1,16384]×[16384,7168]  weights={r['o_proj_weight_mb']:.0f}MB  {r['o_proj_ms']:.4f} ms")
    print(f"  │ Residual add                                        {r['residual_ms']:.4f} ms")
    print(f"  └─── Block C total: {r['block_c_ms']:.4f} ms")

    print(f"\n  MoE BLOCK (EP={EP}, weights prefetched on GPU)")
    print(f"  ┌─────────────────────────────────────────────────────────────")
    print(f"  │ Norm          {r['moe_norm_ms']:.4f} ms")
    print(f"  │ Router        gate+softmax+topk                     {r['moe_router_ms']:.4f} ms")
    print(f"  │ EP dispatch   token→expert GPUs (NVLink)             {r['ep_dispatch_ms']:.4f} ms")
    print(f"  │ Expert GEMM   {r['experts_per_gpu']:.0f}/GPU × {r['expert_weight_mb']:.0f}MB weights (HBM)    {r['moe_gemm_ms']:.4f} ms")
    print(f"  │ EP combine    results→home GPU (NVLink)              {r['ep_combine_ms']:.4f} ms")
    print(f"  │ Shared expert 1 dense × {r['expert_weight_mb']:.0f}MB weights (HBM)        {r['moe_shared_ms']:.4f} ms")
    print(f"  │ Residual                                            {r['moe_residual_ms']:.4f} ms")
    print(f"  └─── MoE total: {r['moe_total_ms']:.4f} ms")

    print(f"\n  ─── LAYER TOTAL ─────────────────────────────────────────────")
    print(f"  │ Attention path: max(A,B) + C = {r['parallel_ab_ms']:.4f} + {r['block_c_ms']:.4f} = {r['attention_total_ms']:.4f} ms")
    print(f"  │ MoE block:                                          {r['moe_total_ms']:.4f} ms")
    print(f"  │ Layer total:                                        {r['layer_total_ms']:.4f} ms")
    print(f"  │")
    print(f"  │ × {NUM_LAYERS} layers = {r['all_layers_ms']:.2f} ms ({r['all_layers_ms']/1000:.3f} s)")
    print(f"  └────────────────────────────────────────────────────────────")

    # Where does the time go?
    attn_pct = r['attention_total_ms'] / r['layer_total_ms'] * 100
    moe_pct = r['moe_total_ms'] / r['layer_total_ms'] * 100
    print(f"\n  Time breakdown: Attention {attn_pct:.1f}% | MoE {moe_pct:.1f}%")


def main():
    print("DeepSeek-V3 DSA Decode: Purely Analytical Per-Layer Breakdown")
    print(f"Model: {NUM_LAYERS} layers, hidden={HIDDEN_SIZE}, {NUM_HEADS} heads, MLA (kv_lora={KV_LORA_RANK}, q_lora={Q_LORA_RANK})")
    print(f"DSA: top-{DSA_SELECTED_TOKENS} selected + {DSA_SLIDING_WINDOW} sliding window = {DSA_ATTENDED} attended")
    print(f"Hardware: H100 SXM (analytical), HBM={H100_HBM_BW_EFF_GBS:.0f} GB/s eff, FP16={H100_FP16_TFLOPS} TFLOPS")
    print(f"EP={EP} (experts prefetched, {NUM_ROUTED_EXPERTS // EP}/GPU), BS=1")

    seq_lens = [4096, 32768, 128 * 1024, 512 * 1024, 1024 * 1024]
    results = []

    for sl in seq_lens:
        r = analyze_dsa_decode(sl)
        results.append(r)
        print_breakdown(r)

    # Summary table
    print(f"\n\n{'=' * 90}")
    print(f" SUMMARY: DSA Decode Timing (BS=1, H100 SXM analytical, {NUM_LAYERS} layers)")
    print(f"{'=' * 90}")
    print(f"{'Seq Len':>10s}  {'Block A':>10s}  {'Block B':>10s}  {'max(A,B)':>10s}  "
          f"{'Block C':>10s}  {'MoE':>10s}  {'Layer':>10s}  {'61 Layers':>12s}")
    print(f"{' ':>10s}  {'Q proj':>10s}  {'KV index':>10s}  {'parallel':>10s}  "
          f"{'Attn+O':>10s}  {'experts':>10s}  {'total':>10s}  {'total':>12s}")
    print("-" * 90)
    for r in results:
        sl = r["seq_len"]
        sl_label = f"{sl // 1024}K" if sl < 1024 * 1024 else f"{sl // (1024 * 1024)}M"
        total_s = r["all_layers_ms"] / 1000
        print(f"{sl_label:>10s}  {r['block_a_ms']:>9.3f}ms  {r['block_b_ms']:>9.3f}ms  "
              f"{r['parallel_ab_ms']:>9.3f}ms  {r['block_c_ms']:>9.3f}ms  "
              f"{r['moe_total_ms']:>9.3f}ms  {r['layer_total_ms']:>9.3f}ms  "
              f"{total_s:>10.3f}s")
    print("-" * 90)


if __name__ == "__main__":
    main()
