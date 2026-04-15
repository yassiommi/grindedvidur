"""Per-layer DSA decode breakdown: combines profiled compute with analytical IO.

For each seq_len and each mode ('hbm' or 'offload'), produces a dict with
every operation's time plus block totals and the full per-layer total.
"""

from __future__ import annotations

from typing import Dict

from experiments.dsa_timing_model import (
    DSA_ATTENDED,
    EP,
    EXPERT_WEIGHT_BYTES,
    HIDDEN_SIZE,
    INDEXER_K_BYTES_PER_TOKEN,
    KV_UP_OUT_DIM,
    MLA_KV_BYTES_PER_TOKEN,
    NUM_EXPERTS_PER_TOK,
    NUM_LAYERS,
    SHARED_EXPERT_WEIGHT_BYTES,
    ProfileTables,
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
