"""Per-layer DSA decode breakdown: combines profiled compute with analytical IO.

For each seq_len and each mode ('hbm' or 'offload'), produces a dict with
every operation's time plus block totals and the full per-layer total.
"""

from __future__ import annotations

from typing import Dict

from experiments.dsa_timing_model import (
    DSA_ATTENDED,
    EP,
    EXPERT_INTERMEDIATE_SIZE,
    EXPERT_WEIGHT_BYTES,
    H100_FP16_TFLOPS,
    H100_MFU,
    HIDDEN_SIZE,
    INDEXER_K_BYTES_PER_TOKEN,
    KV_LORA_RANK,
    KV_UP_OUT_DIM,
    MLA_KV_BYTES_PER_TOKEN,
    NUM_EXPERTS_PER_TOK,
    NUM_LAYERS,
    Q_LORA_RANK,
    Q_TOTAL_DIM,
    SHARED_EXPERT_WEIGHT_BYTES,
    ProfileTables,
    analytical_gemm_ms,
    analytical_moe_expert_gemm_bs_ms,
    analytical_moe_expert_gemm_ms,
    hbm_read_ms,
    io_read_ms,
    load_profiles,
    nvlink_ms,
    pcie_read_ms,
    profiled_or_analytical,
)


def _analytical_memory_bound_ms(weight_bytes: int) -> float:
    """Analytical fallback for a memory-bound GEMM at BS=1 = weight HBM read."""
    return hbm_read_ms(weight_bytes)


def analyze_layer(seq_len: int, mode: str, tables: ProfileTables) -> Dict[str, object]:
    """One DSA decode layer at BS=1.

    mode: 'hbm' (all-in-memory) or 'offload' (KV + indexer on CPU via PCIe).
    """
    assert mode in ("hbm", "offload")
    src: Dict[str, str] = {}  # per-op data source: profiled / analytical

    # ══════════════════════════════════════════════════════════════════
    # BLOCK A — Q projection (1 decode token), parallel with B
    # ══════════════════════════════════════════════════════════════════
    q_down_ms, src["q_down"] = profiled_or_analytical(
        tables, "attn", 1, "mla_q_down_proj",
        lambda: _analytical_memory_bound_ms(7168 * 1536 * 2),
    )
    q_up_ms, src["q_up"] = profiled_or_analytical(
        tables, "attn", 1, "mla_q_up_proj",
        lambda: _analytical_memory_bound_ms(1536 * 24576 * 2),
    )
    rope_ms, src["rope"] = profiled_or_analytical(
        tables, "attn", 1, "mla_rope",
        lambda: hbm_read_ms(128 * 64 * 2 * 2),
    )
    pre_norm_ms, src["pre_norm"] = profiled_or_analytical(
        tables, "attn", 1, "mla_block_norm",
        lambda: hbm_read_ms(HIDDEN_SIZE * 2 * 2),
    )
    block_a_ms = pre_norm_ms + q_down_ms + q_up_ms + rope_ms

    # ══════════════════════════════════════════════════════════════════
    # BLOCK B — KV indexing pipeline (parallel with A)
    # ══════════════════════════════════════════════════════════════════
    # B1: Read indexer K cache. IO-only — analytical (HBM or PCIe).
    indexer_k_bytes = INDEXER_K_BYTES_PER_TOKEN * seq_len
    indexer_read_ms = io_read_ms(indexer_k_bytes, mode)
    src["indexer_read"] = f"analytical-{mode}"

    # B2: Indexer matmul [1, 512] × [512, seq_len]. At BS=1 on FP8 this is
    #     entirely memory-bound on the K read we already counted above, so
    #     the pure compute cost is negligible (well under 1 us for seq_len
    #     up to 1M). Credit the kernel launch as the floor.
    indexer_compute_ms = 0.005  # 5 us kernel launch
    src["indexer_compute"] = "kernel-floor"

    # B3: Top-k over seq_len elements. Empirical GPU partial sort ≈ 1 us
    #     per 1K elements on H100, floored at one kernel launch.
    topk_ms = max(seq_len / 1e6, 0.005)
    src["topk"] = "analytical"

    # B4: Fetch full MLA KV for 2560 attended tokens — scattered gather.
    fetch_kv_bytes = MLA_KV_BYTES_PER_TOKEN * DSA_ATTENDED  # ~2.88 MB
    fetch_kv_ms = io_read_ms(fetch_kv_bytes, mode)
    src["fetch_kv"] = f"analytical-{mode}"

    block_b_ms = indexer_read_ms + indexer_compute_ms + topk_ms + fetch_kv_ms

    # ══════════════════════════════════════════════════════════════════
    # BLOCK C — Attention + output (after max(A,B))
    # ══════════════════════════════════════════════════════════════════
    # C1: kv_up_proj on 2560 fetched tokens — large compute-bound GEMM.
    #     [2560, 512] × [512, 32768]. Use the profile at num_tokens=2560.
    #     Analytical fallback: compute-bound at ~80% MFU of FP16 peak.
    def _kv_up_analytical():
        flops = 2 * DSA_ATTENDED * 512 * KV_UP_OUT_DIM
        tflops_eff = 989.5 * 0.5  # H100 FP16 peak × conservative MFU
        return (flops / 1e9) / (tflops_eff * 1024) * 1e3
    kv_up_ms, src["kv_up"] = profiled_or_analytical(
        tables, "attn", DSA_ATTENDED, "mla_kv_up_proj", _kv_up_analytical,
    )

    # C2: Attention core — 1 Q × 2560 KV × 128 heads. Memory-bound on the
    #     decompressed KV read (~160 MB). Analytical always.
    decompressed_kv_bytes = DSA_ATTENDED * KV_UP_OUT_DIM * 2
    attn_core_ms = hbm_read_ms(decompressed_kv_bytes)
    src["attn_core"] = "analytical"

    # C3: o_proj [1, 16384] × [16384, 7168] at BS=1.
    o_proj_ms, src["o_proj"] = profiled_or_analytical(
        tables, "attn", 1, "mla_o_proj",
        lambda: _analytical_memory_bound_ms(16384 * 7168 * 2),
    )

    # C4: Residual add.
    residual_ms, src["residual"] = profiled_or_analytical(
        tables, "attn", 1, "mla_block_residual",
        lambda: hbm_read_ms(HIDDEN_SIZE * 2 * 2),
    )

    block_c_ms = kv_up_ms + attn_core_ms + o_proj_ms + residual_ms

    # ══════════════════════════════════════════════════════════════════
    # MoE BLOCK — sequential after attention
    # ══════════════════════════════════════════════════════════════════
    moe_norm_ms, src["moe_norm"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_block_norm",
        lambda: hbm_read_ms(HIDDEN_SIZE * 2 * 2),
    )
    router_gate_ms, src["router_gate"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_router_gate",
        lambda: _analytical_memory_bound_ms(7168 * 256 * 2),
    )
    router_sm_ms, src["router_softmax"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_router_softmax",
        lambda: 0.005,
    )
    router_topk_ms, src["router_topk"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_router_topk",
        lambda: 0.005,
    )

    # EP dispatch/combine: profiled values are on a single node and are
    # close to NVLink latency × 8 messages, so profile is fine here.
    dispatch_ms, src["ep_dispatch"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_expert_dispatch",
        lambda: nvlink_ms(NUM_EXPERTS_PER_TOK),
    )
    combine_ms, src["ep_combine"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_expert_combine",
        lambda: nvlink_ms(NUM_EXPERTS_PER_TOK),
    )

    # Expert GEMM — profile is INFLATED (Python loop in profiler). Force
    # analytical: HBM read of ~1 expert × 84 MB per GPU average at EP=8.
    experts_per_gpu = NUM_EXPERTS_PER_TOK / EP  # 1.0
    expert_gemm_ms, src["expert_gemm"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_expert_gemm",
        lambda: analytical_moe_expert_gemm_ms(experts_per_gpu),
        override_always=True,
    )

    # Shared expert — one ~84 MB weight read per token at BS=1. Profile
    # is reasonable here (no multi-expert loop) but still guard against
    # >3× inflation.
    shared_ms, src["shared_expert"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_shared_expert",
        lambda: hbm_read_ms(SHARED_EXPERT_WEIGHT_BYTES),
        inflation_guard_ms=3.0,
    )
    moe_residual_ms, src["moe_residual"] = profiled_or_analytical(
        tables, "mlp", 1, "moe_block_residual",
        lambda: hbm_read_ms(HIDDEN_SIZE * 2 * 2),
    )

    moe_total_ms = (
        moe_norm_ms
        + router_gate_ms + router_sm_ms + router_topk_ms
        + dispatch_ms + expert_gemm_ms + combine_ms
        + shared_ms + moe_residual_ms
    )

    # ══════════════════════════════════════════════════════════════════
    # Totals
    # ══════════════════════════════════════════════════════════════════
    parallel_ab_ms = max(block_a_ms, block_b_ms)
    attention_total_ms = parallel_ab_ms + block_c_ms
    layer_total_ms = attention_total_ms + moe_total_ms

    return {
        "seq_len": seq_len,
        "mode": mode,
        # Block A
        "pre_norm_ms": pre_norm_ms,
        "q_down_ms": q_down_ms,
        "q_up_ms": q_up_ms,
        "rope_ms": rope_ms,
        "block_a_ms": block_a_ms,
        # Block B
        "indexer_k_mb": indexer_k_bytes / (1024 ** 2),
        "indexer_read_ms": indexer_read_ms,
        "indexer_compute_ms": indexer_compute_ms,
        "topk_ms": topk_ms,
        "fetch_kv_mb": fetch_kv_bytes / (1024 ** 2),
        "fetch_kv_ms": fetch_kv_ms,
        "block_b_ms": block_b_ms,
        # Block C
        "kv_up_ms": kv_up_ms,
        "attn_core_ms": attn_core_ms,
        "decompressed_kv_mb": decompressed_kv_bytes / (1024 ** 2),
        "o_proj_ms": o_proj_ms,
        "residual_ms": residual_ms,
        "block_c_ms": block_c_ms,
        # MoE
        "moe_norm_ms": moe_norm_ms,
        "router_gate_ms": router_gate_ms,
        "router_softmax_ms": router_sm_ms,
        "router_topk_ms": router_topk_ms,
        "ep_dispatch_ms": dispatch_ms,
        "expert_gemm_ms": expert_gemm_ms,
        "ep_combine_ms": combine_ms,
        "shared_expert_ms": shared_ms,
        "moe_residual_ms": moe_residual_ms,
        "moe_total_ms": moe_total_ms,
        # Totals
        "parallel_ab_ms": parallel_ab_ms,
        "attention_total_ms": attention_total_ms,
        "layer_total_ms": layer_total_ms,
        "all_layers_ms": layer_total_ms * NUM_LAYERS,
        # Provenance
        "source": src,
    }


def compute_max_batch(mode: str, seq_len: int,
                      total_hbm_gb: float = 80.0,
                      weight_hbm_gb: float = 60.0) -> Dict[str, float]:
    """How many concurrent sequences can each mode support on one H100?

    In offload mode the full MLA KV cache and the indexer K cache live on
    CPU, so HBM is only consumed by model weights and a small per-sequence
    working set (which we treat as negligible at this level of modelling).
    The concurrency bound is instead the PCIe transfer budget per decode
    token (layer_time_budget) — we report CPU-RAM-limited capacity as a
    nominal "unlimited by HBM" answer and print the per-seq CPU KV bytes
    so the caller can translate against whatever CPU RAM is available.

    In hbm mode the full KV cache lives on device, so concurrency is
    bounded by (total_hbm - weights) / (kv_bytes_per_seq).
    """
    free_hbm_bytes = (total_hbm_gb - weight_hbm_gb) * (1024 ** 3)
    kv_bytes_per_seq = MLA_KV_BYTES_PER_TOKEN * seq_len * NUM_LAYERS
    indexer_bytes_per_seq = INDEXER_K_BYTES_PER_TOKEN * seq_len * NUM_LAYERS

    if mode == "hbm":
        # KV + indexer K both on device
        per_seq = kv_bytes_per_seq + indexer_bytes_per_seq
        max_seqs = int(free_hbm_bytes // per_seq) if per_seq > 0 else 0
    else:
        # Offload: KV lives on CPU RAM, HBM only holds weights (+ small
        # transient gather buffer we neglect). Effectively HBM-unbounded.
        per_seq = 0
        max_seqs = -1  # sentinel for "HBM-unbounded"

    return {
        "mode": mode,
        "seq_len": seq_len,
        "hbm_free_gb": free_hbm_bytes / (1024 ** 3),
        "per_seq_kv_mb": kv_bytes_per_seq / (1024 ** 2),
        "per_seq_indexer_mb": indexer_bytes_per_seq / (1024 ** 2),
        "per_seq_hbm_mb": per_seq / (1024 ** 2),
        "max_concurrent_seqs": max_seqs,
    }


def run_sweep(seq_lens, modes=("hbm", "offload")):
    """Return a dict: (mode, seq_len) -> per-layer breakdown dict."""
    tables = load_profiles()
    results = {}
    for mode in modes:
        for sl in seq_lens:
            results[(mode, sl)] = analyze_layer(sl, mode, tables)
    return results


# ══════════════════════════════════════════════════════════════════════
# Batch-size-aware analysis
# ══════════════════════════════════════════════════════════════════════

def analyze_layer_bs(
    seq_len: int, mode: str, batch_size: int, tables: ProfileTables
) -> Dict[str, object]:
    """DSA decode one layer at arbitrary batch_size.

    Key differences from analyze_layer (BS=1):
    - Block A ops: profiled at num_tokens=batch_size, analytical fallback
      for GEMM [BS, …] if profile is saturated/extrapolated.
    - Block B IO: scales with batch_size (each sequence has its own
      indexer K + KV to fetch). Indexer matmul also batched.
    - Block C: kv_up_proj on BS*2560 tokens; attn_core reads BS× decompressed
      KV; o_proj at [BS, 16384]×[16384, 7168].
    - MoE: profiled at num_tokens=batch_size where available; expert GEMM
      always analytical (grouped-GEMM with unique-expert scaling).
    """
    assert mode in ("hbm", "offload")
    bs = batch_size

    # ─── BLOCK A ─── Q projection for BS tokens ─────────────────────
    q_down_ms, _ = profiled_or_analytical(
        tables, "attn", bs, "mla_q_down_proj",
        lambda: analytical_gemm_ms(bs, HIDDEN_SIZE, Q_LORA_RANK),
    )
    q_up_ms, _ = profiled_or_analytical(
        tables, "attn", bs, "mla_q_up_proj",
        lambda: analytical_gemm_ms(bs, Q_LORA_RANK, Q_TOTAL_DIM),
    )
    rope_ms, _ = profiled_or_analytical(
        tables, "attn", bs, "mla_rope",
        lambda: hbm_read_ms(bs * 128 * 64 * 2 * 2),
    )
    pre_norm_ms, _ = profiled_or_analytical(
        tables, "attn", bs, "mla_block_norm",
        lambda: hbm_read_ms(bs * HIDDEN_SIZE * 2 * 2),
    )
    block_a_ms = pre_norm_ms + q_down_ms + q_up_ms + rope_ms

    # ─── BLOCK B ─── KV indexing for BS sequences ───────────────────
    # Each sequence reads its own indexer K and fetches its own KV.
    indexer_k_bytes = INDEXER_K_BYTES_PER_TOKEN * seq_len * bs
    indexer_read_ms = io_read_ms(indexer_k_bytes, mode)

    indexer_compute_ms = max(0.005, 0.005 * bs)  # batched kernel
    topk_ms = max(seq_len / 1e6 * bs, 0.005)

    fetch_kv_bytes = MLA_KV_BYTES_PER_TOKEN * DSA_ATTENDED * bs
    fetch_kv_ms = io_read_ms(fetch_kv_bytes, mode)

    block_b_ms = indexer_read_ms + indexer_compute_ms + topk_ms + fetch_kv_ms

    # ─── BLOCK C ─── Attention + output ──────────────────────────────
    # kv_up_proj: [BS*2560, 512] × [512, 32768]. Profile only goes to
    # num_tokens=4096, so BS≥2 needs analytical.
    kv_up_nt = bs * DSA_ATTENDED
    kv_up_ms, _ = profiled_or_analytical(
        tables, "attn", kv_up_nt, "mla_kv_up_proj",
        lambda: analytical_gemm_ms(kv_up_nt, KV_LORA_RANK, KV_UP_OUT_DIM),
    )

    # Attn core: BS queries × 2560 KV each. BS× decompressed KV read.
    decompressed_kv_bytes = bs * DSA_ATTENDED * KV_UP_OUT_DIM * 2
    attn_core_ms = hbm_read_ms(decompressed_kv_bytes)

    # o_proj: [BS, 16384] × [16384, 7168]
    o_proj_ms, _ = profiled_or_analytical(
        tables, "attn", bs, "mla_o_proj",
        lambda: analytical_gemm_ms(bs, 16384, HIDDEN_SIZE),
    )

    residual_ms, _ = profiled_or_analytical(
        tables, "attn", bs, "mla_block_residual",
        lambda: hbm_read_ms(bs * HIDDEN_SIZE * 2 * 2),
    )
    block_c_ms = kv_up_ms + attn_core_ms + o_proj_ms + residual_ms

    # ─── MoE BLOCK ──────────────────────────────────────────────────
    moe_norm_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_block_norm",
        lambda: hbm_read_ms(bs * HIDDEN_SIZE * 2 * 2),
    )
    router_gate_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_router_gate",
        lambda: analytical_gemm_ms(bs, HIDDEN_SIZE, 256),
    )
    router_sm_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_router_softmax",
        lambda: 0.005 + 0.001 * bs,
    )
    router_topk_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_router_topk",
        lambda: max(0.005, 0.04 * bs),
    )

    dispatch_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_expert_dispatch",
        lambda: nvlink_ms(NUM_EXPERTS_PER_TOK) + 0.001 * bs,
    )
    combine_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_expert_combine",
        lambda: nvlink_ms(NUM_EXPERTS_PER_TOK) + 0.001 * bs,
    )

    # Expert GEMM: always analytical, grouped-GEMM with unique-expert scaling.
    expert_gemm_ms = analytical_moe_expert_gemm_bs_ms(bs)

    # Shared expert: profiled where available, analytical for large BS.
    shared_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_shared_expert",
        lambda: analytical_gemm_ms(bs, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE * 3),
        inflation_guard_ms=3.0,
    )
    moe_residual_ms, _ = profiled_or_analytical(
        tables, "mlp", bs, "moe_block_residual",
        lambda: hbm_read_ms(bs * HIDDEN_SIZE * 2 * 2),
    )

    moe_total_ms = (
        moe_norm_ms + router_gate_ms + router_sm_ms + router_topk_ms
        + dispatch_ms + expert_gemm_ms + combine_ms
        + shared_ms + moe_residual_ms
    )

    # ─── Totals ──────────────────────────────────────────────────────
    parallel_ab_ms = max(block_a_ms, block_b_ms)
    attention_total_ms = parallel_ab_ms + block_c_ms
    layer_total_ms = attention_total_ms + moe_total_ms

    # IO vs compute decomposition (for overlap analysis)
    io_ms = indexer_read_ms + fetch_kv_ms
    compute_ms = layer_total_ms - io_ms

    # Pipelined model: if we could fully overlap layer N's IO with layer
    # N-1's compute (double-buffering), the bottleneck per layer is
    # max(compute, io) rather than compute + io.
    pipelined_layer_ms = max(compute_ms, io_ms)

    return {
        "seq_len": seq_len,
        "mode": mode,
        "batch_size": bs,
        # Blocks
        "block_a_ms": block_a_ms,
        "block_b_ms": block_b_ms,
        "block_c_ms": block_c_ms,
        "moe_total_ms": moe_total_ms,
        # IO detail
        "indexer_read_ms": indexer_read_ms,
        "fetch_kv_ms": fetch_kv_ms,
        "io_ms": io_ms,
        "compute_ms": compute_ms,
        # Totals
        "parallel_ab_ms": parallel_ab_ms,
        "attention_total_ms": attention_total_ms,
        "layer_total_ms": layer_total_ms,
        "all_layers_ms": layer_total_ms * NUM_LAYERS,
        "tpot_ms": layer_total_ms * NUM_LAYERS / bs,
        # Pipelined (overlap IO with previous layer compute)
        "pipelined_layer_ms": pipelined_layer_ms,
        "pipelined_all_layers_ms": pipelined_layer_ms * NUM_LAYERS,
        "pipelined_tpot_ms": pipelined_layer_ms * NUM_LAYERS / bs,
        # Expert detail
        "expert_gemm_ms": expert_gemm_ms,
    }


def run_bs_sweep(seq_lens, batch_sizes, modes=("hbm", "offload")):
    """Return dict: (mode, seq_len, batch_size) -> per-layer dict."""
    tables = load_profiles()
    results = {}
    for mode in modes:
        for sl in seq_lens:
            for bs in batch_sizes:
                results[(mode, sl, bs)] = analyze_layer_bs(sl, mode, bs, tables)
    return results
