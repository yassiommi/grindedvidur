#!/usr/bin/env python3
"""Why does V4 still need on-disk KV offloading despite its aggressive compression?

This experiment answers the question quantitatively by analysing three orthogonal
factors:

  1. STATE CACHE vs COMPRESSED HISTORY separation
     State cache (HBM-resident, bounded ≤2 GB): CSA/HCA uncompressed tails + SWA window.
     Compressed history (grows linearly with S): CSA+HCA entries. At 1M ctx this is ~15 GB
     per request. Disk offloading is for the history, not the state cache.

  2. HBM CAPACITY AT SCALE
     Even though each request's per-GPU footprint is manageable, a production cluster
     serving N concurrent long-context sessions will exhaust HBM once:
       N × compressed_kv_per_gpu + model_weights_per_gpu > HBM_per_gpu

  3. HEAD-DIM AMPLIFICATION (why the tail/SWA are expensive per token)
     V4's head_dim=512 means each UNCOMPRESSED token costs 32 768 bytes —
     8× more than V3's attention heads and 8× more than a hypothetical V4 with head_dim=64.
     This makes the state-cache "expensive per token", forcing careful management even
     when the token count is small.

  4. MULTI-TURN SESSION ACCUMULATION
     In a long conversation (many turns, each adding tokens), the compressed history grows
     with every turn. After ~50 turns at 4 096 tokens/turn, a single session accumulates
     >3 GB of compressed KV. Without disk offloading this persists in HBM between turns.

All hardware and model geometry from InferLens configs.

Outputs (example_outputs/experiments/deepseek_v4_disk_necessity/):
  deepseek_v4_disk_necessity.json
  plot_a_state_vs_history.png       — bounded state cache vs linear compressed history
  plot_b_hbm_capacity.png           — concurrent request limit by cluster / context
  plot_c_head_dim_amplification.png — per-token cost vs hypothetical smaller head_dim
  plot_d_multiturn.png              — compressed KV accumulation over conversation turns
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from vidur.config.device_sku_config import A100DeviceSKUConfig
from vidur.config.model_config import (
    DeepSeekV3ModelConfig,
    DeepSeekV4ProModelConfig,
    Llama3_70BModelConfig,
)

RESULTS_DIR = os.path.join(
    _ROOT, "example_outputs", "experiments", "deepseek_v4_disk_necessity"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── InferLens config instances ────────────────────────────────────────────────
_V4   = DeepSeekV4ProModelConfig()
_V3   = DeepSeekV3ModelConfig()
_L3   = Llama3_70BModelConfig()
_A100 = A100DeviceSKUConfig()

# ── Hardware constants ────────────────────────────────────────────────────────
HBM_GB_PER_GPU  = _A100.total_memory_gb          # 80 GB
TP              = 8

# V4 model weight footprint per GPU: active_params × 2 bytes (FP16) / TP
# active_params ≈ 49B (1.6T total, 6/384 routed + 1 shared expert per token)
# dense layers (embedding + norms) ≈ 1B, per-expert ≈ 4B; 49B is the published figure
V4_ACTIVE_PARAMS_B  = 49.0e9
V3_ACTIVE_PARAMS_B  = 37.0e9
L3_PARAMS_B         = 70.0e9

V4_WEIGHTS_PER_GPU_GB = V4_ACTIVE_PARAMS_B * 2 / TP / 1e9   # FP16
V3_WEIGHTS_PER_GPU_GB = V3_ACTIVE_PARAMS_B * 2 / TP / 1e9
L3_WEIGHTS_PER_GPU_GB = L3_PARAMS_B        * 2 / TP / 1e9

OVERHEAD_GB = 3.0   # activations + framework + scratch

HBM_AVAIL_FOR_KV = {
    "V4":    HBM_GB_PER_GPU - V4_WEIGHTS_PER_GPU_GB - OVERHEAD_GB,
    "V3":    HBM_GB_PER_GPU - V3_WEIGHTS_PER_GPU_GB - OVERHEAD_GB,
    "Llama": HBM_GB_PER_GPU - L3_WEIGHTS_PER_GPU_GB - OVERHEAD_GB,
}


# ── Core KV geometry (corrected from vLLM DeepSeek-V4 blog post) ─────────────
# Each compressed entry = 512-dim shared KV latent + 128-dim DSA indexer, bf16
# SWA tokens keep only the shared KV latent (no indexer)
KV_ENTRY_BYTES  = (_V4.kv_entry_dim + _V4.indexer_dim) * 2   # 1280 bytes
SWA_ENTRY_BYTES = _V4.kv_entry_dim * 2                        # 1024 bytes
N_ALL_LAYERS    = _V4.n_csa_layers + _V4.n_hca_layers         # 61 (30 c4a + 31 c128a)


def v4_kv_breakdown(seq_len: int) -> dict:
    """Per-request KV breakdown for V4 at seq_len tokens."""
    n_c4a  = _V4.n_csa_layers   # 30 c4a layers (stride-4)
    n_c128a = _V4.n_hca_layers  # 31 c128a layers (stride-128)

    c4a_comp   = (seq_len // _V4.csa_chunk_size)  * KV_ENTRY_BYTES * n_c4a
    c128a_comp = (seq_len // _V4.hca_chunk_size) * KV_ENTRY_BYTES * n_c128a
    # 128-token SWA window embedded in every layer (c4a and c128a)
    swa_state  = min(seq_len, _V4.swa_window_size) * SWA_ENTRY_BYTES * N_ALL_LAYERS

    compressed  = c4a_comp + c128a_comp   # grows linearly → disk candidate
    state_cache = swa_state               # HBM-resident, bounded at 128 tok × 61 layers

    return {
        "csa_compressed": c4a_comp,
        "hca_compressed": c128a_comp,
        "csa_tail":       0,
        "hca_tail":       0,
        "swa_state":      swa_state,
        "state_cache":    state_cache,   # HBM-resident, bounded
        "compressed":     compressed,    # grows linearly with S → disk candidate
        "total":          state_cache + compressed,
    }


def v3_kv_bytes(seq_len: int) -> int:
    # MLA (1152 B) + DSA indexer (256 B) = 1408 B/tok/layer
    kv_per = (_V3.kv_lora_rank + _V3.qk_rope_head_dim) * 2 + 256  # 1408 B/tok/layer
    return kv_per * seq_len * _V3.num_layers


def l3_kv_bytes(seq_len: int) -> int:
    kv_per = 2 * (_L3.embedding_dim // _L3.num_q_heads) * _L3.num_kv_heads * 2  # 4096 B/tok/layer
    return kv_per * seq_len * _L3.num_layers


# ── State cache maximum (saturates at S ≥ 128 tokens) ────────────────────────
# SWA window is 128 tokens per layer across all 61 layers; no uncompressed tails
MAX_SWA_STATE   = _V4.swa_window_size * SWA_ENTRY_BYTES * N_ALL_LAYERS  # 128 × 1024 × 61 ≈ 7.9 MB
MAX_STATE_CACHE = MAX_SWA_STATE


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis A — State cache vs compressed history separation
# ═══════════════════════════════════════════════════════════════════════════════

CONTEXTS = [1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536,
            131_072, 262_144, 524_288, 1_000_000]


def compute_separation(contexts: List[int]) -> List[dict]:
    rows = []
    for S in contexts:
        bd = v4_kv_breakdown(S)
        rows.append({
            "seq_len":           S,
            "state_cache_gb":    bd["state_cache"]  / 1e9,
            "compressed_gb":     bd["compressed"]   / 1e9,
            "total_gb":          bd["total"]        / 1e9,
            "state_pct":         100 * bd["state_cache"] / max(bd["total"], 1),
            "compressed_pct":    100 * bd["compressed"]  / max(bd["total"], 1),
            "v3_kv_gb":          v3_kv_bytes(S) / 1e9,
            "l3_kv_gb":          l3_kv_bytes(S) / 1e9,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis B — HBM capacity: concurrent requests vs cluster size
# ═══════════════════════════════════════════════════════════════════════════════

GPU_COUNTS = [8, 16, 32, 64, 128]   # TP=8 always → 1, 2, 4, 8, 16 nodes
CTX_FOR_CAPACITY = [131_072, 262_144, 524_288, 1_000_000]


def max_concurrent_requests(model: str, seq_len: int, total_gpus: int) -> int:
    """Max simultaneous requests whose compressed KV fits in cluster HBM."""
    # With TP=8, KV is sharded across 8 GPUs → per-GPU KV footprint ÷ 8
    kv_per_gpu_gb = (
        v4_kv_breakdown(seq_len)["compressed"] / TP / 1e9 if model == "V4"
        else v3_kv_bytes(seq_len) / TP / 1e9 if model == "V3"
        else l3_kv_bytes(seq_len) / TP / 1e9
    )
    avail = HBM_AVAIL_FOR_KV[model]
    if kv_per_gpu_gb <= 0:
        return 10_000
    return max(1, int(avail / kv_per_gpu_gb))


def compute_capacity(gpu_counts: List[int], contexts: List[int]) -> List[dict]:
    rows = []
    for S in contexts:
        for n_gpu in gpu_counts:
            # Each TP=8 group is one model replica; n_gpu / 8 replicas per cluster
            replicas = n_gpu // TP
            for model in ["V4", "V3", "Llama"]:
                cap_per_replica = max_concurrent_requests(model, S, n_gpu)
                rows.append({
                    "seq_len":           S,
                    "total_gpus":        n_gpu,
                    "replicas":          replicas,
                    "model":             model,
                    "per_replica_cap":   cap_per_replica,
                    "cluster_cap":       cap_per_replica * replicas,
                })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis C — KV latent dimension sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
# V4 stores a 512-dim shared KV latent per compressed entry and per SWA token.
# This analysis shows how the state-cache and compressed history scale with
# the latent dimension (hypothetical vs actual 512).

KV_LATENT_DIMS = [64, 128, 256, 512]   # hypothetical vs actual (512)


def state_cache_at_saturation(kv_latent_dim: int) -> dict:
    """Saturated state cache bytes if V4 had a different KV latent dimension."""
    swa_bytes_per_tok = kv_latent_dim * 2   # K+V, no indexer in SWA
    swa_state = _V4.swa_window_size * swa_bytes_per_tok * N_ALL_LAYERS
    return {
        "kv_latent_dim":   kv_latent_dim,
        "swa_bytes_per_tok": swa_bytes_per_tok,
        "swa_state_mb":    swa_state / 1e6,
        "total_state_mb":  swa_state / 1e6,
    }


def compressed_per_token_per_layer_avg(kv_latent_dim: int) -> float:
    """Average compressed KV bytes/context-token/layer for c4a+c128a layers."""
    entry_bytes = (kv_latent_dim + _V4.indexer_dim) * 2   # with DSA indexer
    n_csa, n_hca = _V4.n_csa_layers, _V4.n_hca_layers
    c4a_bpt  = entry_bytes / _V4.csa_chunk_size    # bytes/token/layer for c4a
    c128a_bpt = entry_bytes / _V4.hca_chunk_size   # bytes/token/layer for c128a
    return (c4a_bpt * n_csa + c128a_bpt * n_hca) / (n_csa + n_hca)


def compute_head_dim_analysis(kv_latent_dims: List[int]) -> List[dict]:
    rows = []
    for kld in kv_latent_dims:
        sat = state_cache_at_saturation(kld)
        avg_comp = compressed_per_token_per_layer_avg(kld)
        entry_bytes = (kld + _V4.indexer_dim) * 2
        comp_1m = (
            (1_000_000 // _V4.csa_chunk_size)  * entry_bytes * _V4.n_csa_layers
            + (1_000_000 // _V4.hca_chunk_size) * entry_bytes * _V4.n_hca_layers
        )
        rows.append({
            "kv_latent_dim":         kld,
            "swa_bytes_per_tok":     sat["swa_bytes_per_tok"],
            "state_cache_sat_mb":    sat["total_state_mb"],
            "state_cache_sat_gb":    sat["total_state_mb"] / 1e3,
            "avg_comp_bytes_tok_lyr": avg_comp,
            "compressed_1m_gb":      comp_1m / 1e9,
            "is_actual":             kld == _V4.kv_entry_dim,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis D — Multi-turn session accumulation
# ═══════════════════════════════════════════════════════════════════════════════

TOKENS_PER_TURN = 2_048     # moderate turn length
MAX_TURNS       = 200


def compute_multiturn(tokens_per_turn: int, max_turns: int) -> List[dict]:
    rows = []
    for turn in range(1, max_turns + 1):
        S = turn * tokens_per_turn
        if S > 1_000_000:
            break
        bd = v4_kv_breakdown(S)
        v3 = v3_kv_bytes(S)
        l3 = l3_kv_bytes(S)
        # Check if compressed KV fits in 1 GPU's available HBM
        fits_hbm = bd["compressed"] / TP / 1e9 <= HBM_AVAIL_FOR_KV["V4"]
        rows.append({
            "turn":              turn,
            "seq_len":           S,
            "state_cache_gb":    bd["state_cache"]  / 1e9,
            "compressed_gb":     bd["compressed"]   / 1e9,
            "total_v4_gb":       bd["total"]        / 1e9,
            "v3_total_gb":       v3 / 1e9,
            "l3_total_gb":       l3 / 1e9,
            "compressed_fits_hbm": fits_hbm,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Print tables
# ═══════════════════════════════════════════════════════════════════════════════

def _ctx(s: int) -> str:
    return f"{s // 1_000_000}M" if s >= 1_000_000 else f"{s // 1024}K"


def print_tables(results: dict) -> None:
    print()
    print("=" * 80)
    print("  DeepSeek-V4-Pro: Why On-Disk KV Cache? Capacity Analysis")
    print("=" * 80)

    print()
    print("  A — State Cache vs Compressed History")
    print(f"  {'Context':<8}  {'State (HBM)':>12}  {'Compressed':>12}  "
          f"{'State%':>8}  {'V3 MLA':>10}  {'Llama MHA':>10}")
    print("  " + "-" * 70)
    for r in results["separation"]:
        print(f"  {_ctx(r['seq_len']):<8}  "
              f"{r['state_cache_gb']:>11.3f}G  "
              f"{r['compressed_gb']:>11.3f}G  "
              f"{r['state_pct']:>7.1f}%  "
              f"{r['v3_kv_gb']:>9.3f}G  "
              f"{r['l3_kv_gb']:>9.3f}G")

    print()
    print("  B — Max Concurrent V4 Requests (per GPU, single TP=8 replica)")
    print(f"  {'Context':<8}  {'KV/GPU (GB)':>12}  "
          f"{'Avail HBM (GB)':>15}  {'Max requests':>13}")
    print("  " + "-" * 55)
    seen = set()
    for r in results["capacity"]:
        if r["model"] == "V4" and r["total_gpus"] == TP:
            key = r["seq_len"]
            if key not in seen:
                seen.add(key)
                kv_per_gpu = v4_kv_breakdown(r["seq_len"])["compressed"] / TP / 1e9
                print(f"  {_ctx(r['seq_len']):<8}  "
                      f"{kv_per_gpu:>11.3f}G  "
                      f"{HBM_AVAIL_FOR_KV['V4']:>14.1f}G  "
                      f"{r['per_replica_cap']:>13,}")

    print()
    print("  C — KV Latent Dim Sensitivity (state cache at saturation, ≥128-token session)")
    print(f"  {'kv_latent_dim':>14}  {'SWA B/tok':>10}  {'avg comp B/tok/lyr':>20}  "
          f"{'State cache MB':>15}  {'Compressed@1M GB':>17}")
    print("  " + "-" * 84)
    for r in results["head_dim_analysis"]:
        marker = " ← actual V4" if r["is_actual"] else ""
        print(f"  {r['kv_latent_dim']:>14}  "
              f"{r['swa_bytes_per_tok']:>10,}  "
              f"{r['avg_comp_bytes_tok_lyr']:>20.1f}  "
              f"{r['state_cache_sat_mb']:>15.2f}  "
              f"{r['compressed_1m_gb']:>16.3f}{marker}")

    print()
    print("  D — Multi-Turn Session (2048 tokens/turn)")
    print(f"  {'Turn':>5}  {'Context':>8}  {'State (HBM)':>12}  "
          f"{'Compressed':>12}  {'V3 total':>10}  {'HBM OK?':>8}")
    print("  " + "-" * 60)
    show_turns = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    for r in results["multiturn"]:
        if r["turn"] in show_turns:
            ok = "✓" if r["compressed_fits_hbm"] else "DISK!"
            print(f"  {r['turn']:>5}  "
                  f"{_ctx(r['seq_len']):>8}  "
                  f"{r['state_cache_gb']:>11.3f}G  "
                  f"{r['compressed_gb']:>11.3f}G  "
                  f"{r['v3_total_gb']:>9.3f}G  "
                  f"{ok:>8}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_a_separation(results: dict) -> None:
    rows = results["separation"]
    seqs  = [r["seq_len"] for r in rows]
    xlabs = [_ctx(s) for s in seqs]
    x     = np.arange(len(seqs))

    state_g = [r["state_cache_gb"] for r in rows]
    comp_g  = [r["compressed_gb"]  for r in rows]
    v3_g    = [r["v3_kv_gb"]       for r in rows]
    l3_g    = [r["l3_kv_gb"]       for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.25

    ax.bar(x - w, state_g, w, color="#1abc9c", label="V4 state cache (HBM, bounded)", zorder=3)
    ax.bar(x,     comp_g,  w, color="#e74c3c", label="V4 compressed history (→ disk)", zorder=3)
    ax.bar(x + w, v3_g,    w, color="#3498db", label="V3 MLA total KV", zorder=3, alpha=0.7)

    # Horizontal line at max state cache
    max_sc = MAX_STATE_CACHE / 1e9
    ax.axhline(max_sc, color="#16a085", ls="--", lw=1.5, zorder=4)
    ax.text(len(seqs) - 0.5, max_sc * 1.05, f"State cache max ≈ {max_sc:.2f} GB",
            ha="right", va="bottom", fontsize=8.5, color="#16a085")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2g}"))
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, fontsize=9)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("KV Cache (GB, log scale)")
    ax.set_title(
        "A — V4 State Cache (HBM, bounded) vs Compressed History (disk candidate)\n"
        f"State cache saturates at {MAX_STATE_CACHE/1e6:.1f} MB; compressed history grows to ~9.6 GB at 1M tokens",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "plot_a_state_vs_history.png")


def plot_b_capacity(results: dict) -> None:
    cap_rows = results["capacity"]
    contexts = sorted(set(r["seq_len"] for r in cap_rows if r["model"] == "V4"))
    gpu_counts = sorted(set(r["total_gpus"] for r in cap_rows))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: V4 concurrent requests per TP-8 replica vs context length
    ax = axes[0]
    caps = {S: None for S in contexts}
    for r in cap_rows:
        if r["model"] == "V4" and r["total_gpus"] == TP:
            caps[r["seq_len"]] = r["per_replica_cap"]

    ctxs = [S for S in contexts if caps[S] is not None]
    vals = [caps[S] for S in ctxs]
    xlabs = [_ctx(S) for S in ctxs]
    colors = ["#27ae60" if v >= 100 else "#e67e22" if v >= 10 else "#e74c3c" for v in vals]
    ax.bar(range(len(ctxs)), vals, color=colors, zorder=3)
    ax.axhline(100, color="#e67e22", ls="--", lw=1.2, label="100 concurrent requests")
    ax.axhline(10,  color="#e74c3c", ls=":",  lw=1.2, label="10 concurrent requests")
    ax.set_xticks(range(len(ctxs)))
    ax.set_xticklabels(xlabs, rotation=30, ha="right", fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylabel("Max concurrent V4 requests per replica")
    ax.set_xlabel("Context Length")
    ax.set_title(
        f"B-left — Concurrent request cap\n(single TP={TP} replica, {HBM_GB_PER_GPU} GB HBM/GPU)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, (v, S) in enumerate(zip(vals, ctxs)):
        ax.text(i, v * 1.4, f"{v:,}", ha="center", va="bottom", fontsize=8)

    # Right: V4 vs V3 vs Llama at 1M context, across cluster sizes
    ax = axes[1]
    target_ctx = 1_000_000
    lw = {"V4": 2.5, "V3": 2, "Llama": 2}
    col = {"V4": "#e67e22", "V3": "#3498db", "Llama": "#9b59b6"}
    ls  = {"V4": "-", "V3": "--", "Llama": "-."}

    for model in ["V4", "V3", "Llama"]:
        cluster_caps = []
        for n_gpu in gpu_counts:
            for r in cap_rows:
                if r["model"] == model and r["seq_len"] == target_ctx and r["total_gpus"] == n_gpu:
                    cluster_caps.append(r["cluster_cap"])
                    break
        ax.plot([f"{n}xA100" for n in gpu_counts], cluster_caps,
                marker="o", lw=lw[model], color=col[model], ls=ls[model],
                label=f"{model}", ms=7, zorder=3)

    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Max concurrent sessions (1M-token ctx)")
    ax.set_title(
        "B-right — Cluster capacity @ 1M context\n(all compressed KV in HBM)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "plot_b_hbm_capacity.png")


def plot_c_head_dim(results: dict) -> None:
    rows = results["head_dim_analysis"]
    klds = [r["kv_latent_dim"]       for r in rows]
    sc   = [r["state_cache_sat_gb"]  for r in rows]
    sc_mb= [r["state_cache_sat_mb"]  for r in rows]
    comp = [r["compressed_1m_gb"]    for r in rows]
    avg  = [r["avg_comp_bytes_tok_lyr"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: state cache (MB) and compressed KV at 1M (GB) — different axes
    ax = axes[0]
    ax2r = ax.twinx()
    x  = np.arange(len(klds))
    w  = 0.3
    b1 = ax.bar(x - w/2, sc_mb, w, color="#1abc9c", label="State cache MB (left)", zorder=3)
    b2 = ax2r.bar(x + w/2, comp, w, color="#e74c3c", label="Compressed @ 1M ctx GB (right)", zorder=3)

    # Highlight actual V4
    actual_idx = next(i for i, r in enumerate(rows) if r["is_actual"])
    b1[actual_idx].set_edgecolor("black"); b1[actual_idx].set_linewidth(2)
    b2[actual_idx].set_edgecolor("black"); b2[actual_idx].set_linewidth(2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"kv_latent={k}" for k in klds], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("State cache (MB, saturated)")
    ax2r.set_ylabel("Compressed history @ 1M ctx (GB)")
    ax.set_title(
        "C-left — KV cost vs hypothetical KV latent dim\n(actual V4 = 512; boxed = actual)",
        fontsize=10,
    )
    lines = [b1, b2]
    labels = ["State cache (MB)", "Compressed@1M (GB)"]
    ax.legend(lines, labels, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right: compressed bytes/tok/layer
    ax3 = axes[1]
    colors = ["#e74c3c" if k == _V4.kv_entry_dim else "#95a5a6" for k in klds]
    ax3.bar(range(len(klds)), avg, color=colors, zorder=3)
    ax3.set_xticks(range(len(klds)))
    ax3.set_xticklabels([f"kv_latent={k}" for k in klds], rotation=15, ha="right", fontsize=9)
    ax3.set_ylabel("Avg compressed B/token/layer (c4a+c128a)")
    ax3.set_title(
        "C-right — Compressed KV bytes per token per layer\n(lower = better compression)",
        fontsize=10,
    )
    ax3.grid(axis="y", alpha=0.3)
    for i, a in enumerate(avg):
        ax3.text(i, a + 0.2, f"{a:.1f}", ha="center", fontsize=9)

    # Reference: V3 MLA+indexer
    v3_bpt = (_V3.kv_lora_rank + _V3.qk_rope_head_dim) * 2 + 256  # 1408
    ax3.axhline(v3_bpt, color="#3498db", ls="--", lw=1.5, label=f"V3 MLA+idx: {v3_bpt} B/tok/layer")
    ax3.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, "plot_c_head_dim_amplification.png")


def plot_d_multiturn(results: dict) -> None:
    rows = results["multiturn"]
    turns    = [r["turn"]           for r in rows]
    state_g  = [r["state_cache_gb"] for r in rows]
    comp_g   = [r["compressed_gb"]  for r in rows]
    v3_g     = [r["v3_total_gb"]    for r in rows]
    fits     = [r["compressed_fits_hbm"] for r in rows]

    # Find crossover turn
    crossover = next((r["turn"] for r in rows if not r["compressed_fits_hbm"]), None)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(turns, state_g, color="#1abc9c", alpha=0.4, label="State cache (HBM)")
    ax.fill_between(turns, state_g, [s + c for s, c in zip(state_g, comp_g)],
                    color="#e74c3c", alpha=0.4, label="Compressed history (→ disk)")
    ax.plot(turns, [s + c for s, c in zip(state_g, comp_g)],
            color="#c0392b", lw=2, label="V4 total KV")
    ax.plot(turns, v3_g, "b--", lw=1.8, label="V3 MLA total KV")

    avail_hbm = HBM_AVAIL_FOR_KV["V4"]
    ax.axhline(avail_hbm * TP, color="#8e44ad", ls=":", lw=1.5,
               label=f"Full cluster HBM available for KV ({avail_hbm * TP:.0f} GB, TP={TP})")

    if crossover:
        ax.axvline(crossover, color="#e74c3c", ls="--", lw=1.2)
        ax.text(crossover + 1, avail_hbm * TP * 0.7,
                f"Compressed KV\nexceeds 1 GPU's HBM\nat turn {crossover}\n"
                f"({_ctx(crossover * TOKENS_PER_TURN)} ctx)",
                fontsize=8.5, color="#c0392b",
                bbox=dict(facecolor="white", edgecolor="#c0392b", boxstyle="round,pad=0.3"))

    ax.set_xlabel(f"Conversation Turn ({TOKENS_PER_TURN // 1024}K tokens/turn)")
    ax.set_ylabel("KV Cache (GB)")
    ax.set_title(
        f"D — Multi-Turn Session KV Accumulation ({TOKENS_PER_TURN // 1024}K tokens/turn)\n"
        "State cache (bounded, HBM) vs compressed history (grows → disk)",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, "plot_d_multiturn.png")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Running DeepSeek-V4-Pro Disk Necessity Analysis …")
    print(f"  V4 model weights/GPU: {V4_WEIGHTS_PER_GPU_GB:.2f} GB  |  "
          f"HBM available for KV: {HBM_AVAIL_FOR_KV['V4']:.1f} GB")
    print(f"  Max state cache (saturated): {MAX_STATE_CACHE / 1e6:.1f} MB  |  "
          f"KV_ENTRY_BYTES: {KV_ENTRY_BYTES} B/entry  SWA_ENTRY_BYTES: {SWA_ENTRY_BYTES} B/tok")

    results = {
        "config": {
            "model":              _V4.get_name(),
            "device":             "A100",
            "hbm_gb":             HBM_GB_PER_GPU,
            "tp":                 TP,
            "v4_weights_gb":      V4_WEIGHTS_PER_GPU_GB,
            "hbm_avail_for_kv":   HBM_AVAIL_FOR_KV["V4"],
            "max_state_cache_mb": MAX_STATE_CACHE / 1e6,
            "kv_entry_bytes":     KV_ENTRY_BYTES,
            "swa_entry_bytes":    SWA_ENTRY_BYTES,
            "n_csa_layers":       _V4.n_csa_layers,
            "n_hca_layers":       _V4.n_hca_layers,
            "swa_window_size":    _V4.swa_window_size,
            "tokens_per_turn":    TOKENS_PER_TURN,
        },
        "separation":        compute_separation(CONTEXTS),
        "capacity":          compute_capacity(GPU_COUNTS, CTX_FOR_CAPACITY),
        "head_dim_analysis": compute_head_dim_analysis(KV_LATENT_DIMS),
        "multiturn":         compute_multiturn(TOKENS_PER_TURN, MAX_TURNS),
    }

    json_path = os.path.join(RESULTS_DIR, "deepseek_v4_disk_necessity.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved: {json_path}")

    print_tables(results)

    print("Generating plots …")
    plot_a_separation(results)
    plot_b_capacity(results)
    plot_c_head_dim(results)
    plot_d_multiturn(results)
    print("Done.")


if __name__ == "__main__":
    main()
