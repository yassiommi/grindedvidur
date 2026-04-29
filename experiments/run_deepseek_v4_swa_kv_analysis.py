#!/usr/bin/env python3
"""DeepSeek-V4-Pro SWA KV Cache Analysis: layout, on-disk storage, and cache-hit latency.

Models the heterogeneous KV cache described in the DeepSeek V4 inference framework:

  "KV Cache Structure and Management: A customized KV cache layout manages
   heterogeneous KV entries (CSA/HCA, indexer, Sliding Window Attention (SWA),
   unready-for-compression tokens). A 'state cache' is used for SWA and
   uncompressed tail tokens."

  "On-Disk KV Cache Storage: Three strategies for SWA KV caching
   (Full SWA Caching, Periodic Checkpointing, Zero SWA Caching) offer
   different trade-offs between storage and computation."

V4-Pro layer split (estimated; exact breakdown not published):
  61 total = 8 SWA (sliding-window local) + 28 CSA (moderate global) + 25 HCA (heavy global)

All timing is purely analytical — no GPU profiling data used.
Uses InferLens model configs and device SKU configs for all parameters.

Outputs (example_outputs/experiments/deepseek_v4_swa_kv_analysis/):
  deepseek_v4_swa_kv_analysis.json   — full results for all sweeps
  plot_a_kv_storage_breakdown.png    — per-component storage stacked bars
  plot_b_disk_storage_vs_context.png — on-disk storage lines vs context
  plot_c_cache_hit_latency.png       — latency breakdown by strategy
  plot_d_pareto_storage_vs_latency.png — Pareto at 131K tokens
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

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
    _ROOT, "example_outputs", "experiments", "deepseek_v4_swa_kv_analysis"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Instantiate InferLens configs once ──────────────────────────────────────
_V4 = DeepSeekV4ProModelConfig()
_V3 = DeepSeekV3ModelConfig()
_L3 = Llama3_70BModelConfig()
_A100 = A100DeviceSKUConfig()

# ── Hardware constants (from A100DeviceSKUConfig) ────────────────────────────
BW_EFFICIENCY   = 0.80
NVME_DISK_BW_GBS = 7.0                           # enterprise NVMe SSD
HBM_BW_GBS      = _A100.memory_bandwidth_gb_per_s  # 2039
FP16_TFLOPS     = _A100.fp16_tflops               # 312
PREFILL_MFU     = 0.45
TP              = 8

DISK_EFF_BPS = NVME_DISK_BW_GBS * BW_EFFICIENCY * 1e9
HBM_EFF_BPS  = HBM_BW_GBS       * BW_EFFICIENCY * 1e9

# ── Sweep configuration ──────────────────────────────────────────────────────
CONTEXT_LENGTHS = [4_096, 8_192, 16_384, 32_768, 65_536,
                   131_072, 262_144, 524_288, 1_000_000]
LATENCY_CONTEXTS = [8_192, 32_768, 131_072, 524_288, 1_000_000]
PARETO_CONTEXT   = 131_072

# Checkpoint C values for Pareto sweep — deliberately non-divisors of 4096
# so we get genuinely intermediate disk/latency trade-off points
PARETO_C_VALUES = [200, 500, 800, 1100, 1500, 2000, 2700, 3300, 3800, 4500]

# Main strategy set for plots A/B/C
MAIN_STRATEGIES = [
    ("Full",         None),    # store full SWA window
    ("Periodic-500", 500),
    ("Periodic-1500",1500),
    ("Zero",         None),    # store no SWA
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SwaStrategy:
    """Describes one on-disk SWA caching strategy."""
    label: str                          # display name
    checkpoint_interval: Optional[int]  # C tokens; None means Full or Zero
    is_full: bool   = False
    is_zero: bool   = False

    @classmethod
    def full(cls) -> "SwaStrategy":
        return cls("Full SWA Caching", None, is_full=True)

    @classmethod
    def zero(cls) -> "SwaStrategy":
        return cls("Zero SWA Caching", None, is_zero=True)

    @classmethod
    def periodic(cls, c: int) -> "SwaStrategy":
        return cls(f"Periodic (C={c})", c)


@dataclass
class V4LayerKv:
    """KV storage for a single V4 layer at a given sequence length."""
    layer_type: str          # "CSA", "HCA", or "SWA"
    seq_len: int
    n_compressed: int        # floor(S / chunk) compressed entries
    n_tail: int              # S % chunk — full-resolution tail in state cache
    n_state: int             # tokens held in HBM state cache
    kv_full_bytes: int       # bytes per uncompressed token (32 768 for V4)

    @property
    def compressed_bytes(self) -> int:
        return self.n_compressed * self.kv_full_bytes

    @property
    def tail_bytes(self) -> int:
        return self.n_tail * self.kv_full_bytes

    @property
    def state_bytes(self) -> int:
        return self.n_state * self.kv_full_bytes


@dataclass
class V4KvLayout:
    """Complete KV cache layout for a V4-Pro request at one context length."""
    seq_len: int
    kv_full_bytes: int           # 32 768 — bytes per uncompressed token

    # Per-layer lists
    csa_layers: List[V4LayerKv]
    hca_layers: List[V4LayerKv]
    swa_layers: List[V4LayerKv]

    # Aggregated bytes
    csa_compressed_bytes: int
    csa_tail_bytes: int
    hca_compressed_bytes: int
    hca_tail_bytes: int
    swa_state_bytes: int         # always-HBM SWA window

    @property
    def state_cache_bytes(self) -> int:
        """Total HBM state cache: CSA tail + HCA tail + SWA window."""
        return self.csa_tail_bytes + self.hca_tail_bytes + self.swa_state_bytes

    @property
    def full_disk_bytes(self) -> int:
        """On-disk bytes under Full SWA strategy."""
        return self.csa_compressed_bytes + self.hca_compressed_bytes + self.swa_state_bytes

    @property
    def zero_disk_bytes(self) -> int:
        """On-disk bytes under Zero SWA strategy (no SWA stored)."""
        return self.csa_compressed_bytes + self.hca_compressed_bytes


@dataclass
class StrategyResult:
    """Cache-hit metrics for one strategy at one context length."""
    strategy_label: str
    seq_len: int
    disk_bytes: int
    swa_disk_tokens: int        # SWA tokens stored on disk (per SWA layer)
    swa_recompute_tokens: int   # SWA tokens to recompute at cache-hit
    disk_load_ms: float
    swa_recompute_ms: float
    first_decode_read_ms: float

    @property
    def total_latency_ms(self) -> float:
        return self.disk_load_ms + self.swa_recompute_ms + self.first_decode_read_ms


@dataclass
class BaselineResult:
    """On-disk KV storage for a non-V4 model."""
    model_name: str
    seq_len: int
    kv_bytes_per_tok_per_layer: int
    num_layers: int

    @property
    def total_disk_bytes(self) -> int:
        return self.kv_bytes_per_tok_per_layer * self.seq_len * self.num_layers


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical core — KV layout
# ═══════════════════════════════════════════════════════════════════════════════

def _kv_full_bytes() -> int:
    """Full-resolution KV bytes per token: 2 * num_kv_heads * head_dim * 2 (FP16)."""
    return 2 * _V4.num_kv_heads * _V4.head_dim * 2  # = 32 768


def _build_layer(layer_type: str, seq_len: int, kv_full: int) -> V4LayerKv:
    csa_chunk = _V4.csa_chunk_size  # 64
    hca_chunk = _V4.hca_chunk_size  # 1024
    swa_win   = _V4.swa_window_size  # 4096

    if layer_type == "CSA":
        n_comp = seq_len // csa_chunk
        n_tail = seq_len %  csa_chunk
        n_state = n_tail
    elif layer_type == "HCA":
        n_comp = seq_len // hca_chunk
        n_tail = seq_len %  hca_chunk
        n_state = n_tail
    else:  # SWA
        n_comp  = 0
        n_tail  = 0
        n_state = min(seq_len, swa_win)

    return V4LayerKv(
        layer_type=layer_type,
        seq_len=seq_len,
        n_compressed=n_comp,
        n_tail=n_tail,
        n_state=n_state,
        kv_full_bytes=kv_full,
    )


def compute_v4_kv_layout(seq_len: int) -> V4KvLayout:
    """Build the full heterogeneous KV layout for V4-Pro at the given context length."""
    kv_full   = _kv_full_bytes()
    n_csa     = _V4.n_csa_layers   # 28
    n_hca     = _V4.n_hca_layers   # 25
    n_swa     = _V4.n_swa_layers   # 8

    csa_layers = [_build_layer("CSA", seq_len, kv_full) for _ in range(n_csa)]
    hca_layers = [_build_layer("HCA", seq_len, kv_full) for _ in range(n_hca)]
    swa_layers = [_build_layer("SWA", seq_len, kv_full) for _ in range(n_swa)]

    csa_comp  = sum(l.compressed_bytes for l in csa_layers)
    csa_tail  = sum(l.tail_bytes       for l in csa_layers)
    hca_comp  = sum(l.compressed_bytes for l in hca_layers)
    hca_tail  = sum(l.tail_bytes       for l in hca_layers)
    swa_state = sum(l.state_bytes      for l in swa_layers)

    return V4KvLayout(
        seq_len=seq_len,
        kv_full_bytes=kv_full,
        csa_layers=csa_layers,
        hca_layers=hca_layers,
        swa_layers=swa_layers,
        csa_compressed_bytes=csa_comp,
        csa_tail_bytes=csa_tail,
        hca_compressed_bytes=hca_comp,
        hca_tail_bytes=hca_tail,
        swa_state_bytes=swa_state,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical core — SWA strategy
# ═══════════════════════════════════════════════════════════════════════════════

def _strategy_swa_disk_tokens(strategy: SwaStrategy, layout: V4KvLayout) -> Tuple[int, int]:
    """Return (swa_disk_tokens_per_layer, swa_recompute_tokens).

    swa_disk_tokens_per_layer: how many SWA tokens are persisted per SWA layer.
    swa_recompute_tokens: tokens that must be recomputed on a cache hit.
    """
    swa_win = _V4.swa_window_size
    actual_win = min(layout.seq_len, swa_win)

    if strategy.is_full:
        return actual_win, 0
    if strategy.is_zero:
        return 0, actual_win
    # Periodic(C)
    c = strategy.checkpoint_interval
    n_ckpts  = actual_win // c
    disk_tok = n_ckpts * c
    recomp   = actual_win - disk_tok
    return disk_tok, recomp


def _swa_recompute_ms(n_recompute: int) -> float:
    """Analytical prefill time (ms) to recompute n_recompute tokens through SWA layers.

    FLOPs = (attention + MoE MLP) × n_swa_layers
    Attention: 4 × n² × head_dim × num_q_heads  (QK + SV, causal window ≈ n)
    MoE MLP:   num_experts_per_tok × 3 × expert_dim × embedding_dim × n × 2
    """
    if n_recompute == 0:
        return 0.0
    n   = n_recompute
    n_s = _V4.n_swa_layers
    attn_flops = 4 * n * n * _V4.head_dim * _V4.num_q_heads * n_s
    mlp_flops  = (_V4.num_experts_per_tok * 3 * _V4.moe_intermediate_size
                  * _V4.embedding_dim * n * 2 * n_s)
    total_flops = attn_flops + mlp_flops
    return total_flops / (FP16_TFLOPS * PREFILL_MFU * 1e12) * 1e3


def compute_strategy_result(
    strategy: SwaStrategy, layout: V4KvLayout
) -> StrategyResult:
    """Compute on-disk bytes and cache-hit latency for one strategy."""
    kv_full  = layout.kv_full_bytes
    n_swa    = _V4.n_swa_layers

    swa_disk_tok, swa_recomp_tok = _strategy_swa_disk_tokens(strategy, layout)

    disk_bytes = (layout.csa_compressed_bytes
                  + layout.hca_compressed_bytes
                  + swa_disk_tok * kv_full * n_swa)

    disk_load_ms        = disk_bytes / DISK_EFF_BPS * 1e3
    swa_recompute_ms    = _swa_recompute_ms(swa_recomp_tok)
    first_decode_read_ms = layout.state_cache_bytes / HBM_EFF_BPS / TP * 1e3

    return StrategyResult(
        strategy_label=strategy.label,
        seq_len=layout.seq_len,
        disk_bytes=disk_bytes,
        swa_disk_tokens=swa_disk_tok,
        swa_recompute_tokens=swa_recomp_tok,
        disk_load_ms=disk_load_ms,
        swa_recompute_ms=swa_recompute_ms,
        first_decode_read_ms=first_decode_read_ms,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Baseline models (V3 MLA, Llama-3-70B MHA)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_baseline(model_name: str, seq_len: int) -> BaselineResult:
    if model_name == "V3":
        kv = (_V3.kv_lora_rank + _V3.qk_rope_head_dim) * 2  # 1152
        n  = _V3.num_layers
    else:  # Llama
        kv = 2 * _L3.num_kv_heads * (_L3.embedding_dim // _L3.num_q_heads) * 2  # 4096
        n  = _L3.num_layers
    return BaselineResult(model_name=model_name, seq_len=seq_len,
                          kv_bytes_per_tok_per_layer=kv, num_layers=n)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_results — ties everything together
# ═══════════════════════════════════════════════════════════════════════════════

def compute_results() -> dict:
    """Run all sweeps and return a serialisable results dict."""

    main_strats = [
        SwaStrategy.full(),
        SwaStrategy.periodic(500),
        SwaStrategy.periodic(1500),
        SwaStrategy.zero(),
    ]

    # ── Context sweep ────────────────────────────────────────────────────────
    layouts, strategy_rows, v3_rows, l3_rows = [], [], [], []

    for S in CONTEXT_LENGTHS:
        layout = compute_v4_kv_layout(S)
        layouts.append({
            "seq_len":              S,
            "kv_full_bytes":        layout.kv_full_bytes,
            "csa_compressed_gb":    layout.csa_compressed_bytes / 1e9,
            "csa_tail_mb":          layout.csa_tail_bytes / 1e6,
            "hca_compressed_gb":    layout.hca_compressed_bytes / 1e9,
            "hca_tail_mb":          layout.hca_tail_bytes / 1e6,
            "swa_state_gb":         layout.swa_state_bytes / 1e9,
            "state_cache_gb":       layout.state_cache_bytes / 1e9,
            "full_disk_gb":         layout.full_disk_bytes / 1e9,
            "zero_disk_gb":         layout.zero_disk_bytes / 1e9,
        })

        for strat in main_strats:
            r = compute_strategy_result(strat, layout)
            strategy_rows.append({
                "seq_len":               S,
                "strategy":              strat.label,
                "disk_gb":               r.disk_bytes / 1e9,
                "swa_disk_tokens":       r.swa_disk_tokens,
                "swa_recompute_tokens":  r.swa_recompute_tokens,
                "disk_load_ms":          r.disk_load_ms,
                "swa_recompute_ms":      r.swa_recompute_ms,
                "first_decode_read_ms":  r.first_decode_read_ms,
                "total_latency_ms":      r.total_latency_ms,
            })

        v3_rows.append({"seq_len": S,
                        "disk_gb": compute_baseline("V3", S).total_disk_bytes / 1e9})
        l3_rows.append({"seq_len": S,
                        "disk_gb": compute_baseline("Llama", S).total_disk_bytes / 1e9})

    # ── Pareto sweep at PARETO_CONTEXT ───────────────────────────────────────
    pareto_layout = compute_v4_kv_layout(PARETO_CONTEXT)
    pareto_rows = []
    pareto_strats = (
        [SwaStrategy.full()]
        + [SwaStrategy.periodic(c) for c in PARETO_C_VALUES]
        + [SwaStrategy.zero()]
    )
    for strat in pareto_strats:
        r = compute_strategy_result(strat, pareto_layout)
        pareto_rows.append({
            "label":                 strat.label,
            "checkpoint_interval":   strat.checkpoint_interval,
            "is_full":               strat.is_full,
            "is_zero":               strat.is_zero,
            "disk_gb":               r.disk_bytes / 1e9,
            "swa_recompute_tokens":  r.swa_recompute_tokens,
            "disk_load_ms":          r.disk_load_ms,
            "swa_recompute_ms":      r.swa_recompute_ms,
            "first_decode_read_ms":  r.first_decode_read_ms,
            "total_latency_ms":      r.total_latency_ms,
        })

    return {
        "config": {
            "model":           _V4.get_name(),
            "device":          "A100",
            "n_layers_total":  _V4.num_layers,
            "n_swa_layers":    _V4.n_swa_layers,
            "n_csa_layers":    _V4.n_csa_layers,
            "n_hca_layers":    _V4.n_hca_layers,
            "num_kv_heads":    _V4.num_kv_heads,
            "head_dim":        _V4.head_dim,
            "csa_chunk_size":  _V4.csa_chunk_size,
            "hca_chunk_size":  _V4.hca_chunk_size,
            "swa_window_size": _V4.swa_window_size,
            "kv_full_bytes":   _kv_full_bytes(),
            "hbm_bw_gbs":      HBM_BW_GBS,
            "disk_bw_gbs":     NVME_DISK_BW_GBS,
            "bw_efficiency":   BW_EFFICIENCY,
            "tp":              TP,
            "prefill_mfu":     PREFILL_MFU,
            "fp16_tflops":     FP16_TFLOPS,
        },
        "layouts":          layouts,
        "strategy_results": strategy_rows,
        "v3_baselines":     v3_rows,
        "l3_baselines":     l3_rows,
        "pareto_sweep":     pareto_rows,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# print_tables
# ═══════════════════════════════════════════════════════════════════════════════

def _ctx_label(s: int) -> str:
    return f"{s // 1_000_000}M" if s >= 1_000_000 else f"{s // 1024}K"


def print_tables(results: dict) -> None:
    cfg = results["config"]
    print()
    print("=" * 90)
    print(f"  DeepSeek-V4-Pro SWA KV Cache Analysis  |  {cfg['model']}")
    print(f"  Device: {cfg['device']}  |  "
          f"HBM {cfg['hbm_bw_gbs']} GB/s  |  "
          f"NVMe {cfg['disk_bw_gbs']} GB/s  |  "
          f"TP={cfg['tp']}  |  MFU={cfg['prefill_mfu']}")
    print(f"  Layer split: {cfg['n_swa_layers']} SWA + "
          f"{cfg['n_csa_layers']} CSA (chunk={cfg['csa_chunk_size']}) + "
          f"{cfg['n_hca_layers']} HCA (chunk={cfg['hca_chunk_size']})  |  "
          f"SWA window={cfg['swa_window_size']} tokens  |  "
          f"kv_full={cfg['kv_full_bytes']} B/tok")
    print("=" * 90)

    # Table 1: KV layout breakdown
    print()
    print("  TABLE 1 — KV Layout Breakdown")
    print(f"  {'Context':<8}  {'CSA comp':>10}  {'CSA tail':>10}  "
          f"{'HCA comp':>10}  {'HCA tail':>10}  {'SWA state':>10}  {'State cache':>12}  {'Full disk':>10}")
    print("  " + "-" * 88)
    for row in results["layouts"]:
        print(f"  {_ctx_label(row['seq_len']):<8}  "
              f"{row['csa_compressed_gb']:>9.3f}G  "
              f"{row['csa_tail_mb']:>9.1f}M  "
              f"{row['hca_compressed_gb']:>9.3f}G  "
              f"{row['hca_tail_mb']:>9.1f}M  "
              f"{row['swa_state_gb']:>9.3f}G  "
              f"{row['state_cache_gb']:>11.3f}G  "
              f"{row['full_disk_gb']:>9.3f}G")

    # Table 2: On-disk storage comparison
    print()
    print("  TABLE 2 — On-Disk Storage (GB) by Strategy + Baselines")
    strat_labels = ["Full SWA Caching", "Periodic (C=500)", "Periodic (C=1500)", "Zero SWA Caching"]
    print(f"  {'Context':<8}  {'Full':>8}  {'P-C=500':>8}  {'P-C=1500':>9}  "
          f"{'Zero':>8}  {'V3 MLA':>8}  {'Llama MHA':>10}")
    print("  " + "-" * 74)
    by_ctx: dict = {}
    for row in results["strategy_results"]:
        by_ctx.setdefault(row["seq_len"], {})[row["strategy"]] = row["disk_gb"]
    for v3, l3 in zip(results["v3_baselines"], results["l3_baselines"]):
        S = v3["seq_len"]
        d = by_ctx.get(S, {})
        print(f"  {_ctx_label(S):<8}  "
              f"{d.get('Full SWA Caching', 0):>8.3f}  "
              f"{d.get('Periodic (C=500)', 0):>8.3f}  "
              f"{d.get('Periodic (C=1500)', 0):>9.3f}  "
              f"{d.get('Zero SWA Caching', 0):>8.3f}  "
              f"{v3['disk_gb']:>8.3f}  "
              f"{l3['disk_gb']:>10.3f}")

    # Table 3: Cache-hit latency at selected contexts
    print()
    print("  TABLE 3 — Cache-Hit Latency (ms) at Selected Contexts")
    print(f"  {'Context':<8}  {'Strategy':<22}  "
          f"{'Disk ms':>9}  {'Recomp ms':>10}  {'Decode ms':>10}  {'Total ms':>10}")
    print("  " + "-" * 80)
    sel = set(LATENCY_CONTEXTS)
    for row in results["strategy_results"]:
        if row["seq_len"] in sel:
            print(f"  {_ctx_label(row['seq_len']):<8}  "
                  f"{row['strategy']:<22}  "
                  f"{row['disk_load_ms']:>9.1f}  "
                  f"{row['swa_recompute_ms']:>10.2f}  "
                  f"{row['first_decode_read_ms']:>10.2f}  "
                  f"{row['total_latency_ms']:>10.1f}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# plot_results — four publication-quality plots
# ═══════════════════════════════════════════════════════════════════════════════

_STRAT_COLORS = {
    "Full SWA Caching":   "#e74c3c",
    "Periodic (C=500)":   "#e67e22",
    "Periodic (C=1500)":  "#f1c40f",
    "Zero SWA Caching":   "#2ecc71",
}
_STRAT_LABELS_SHORT = {
    "Full SWA Caching":   "Full",
    "Periodic (C=500)":   "Periodic-500",
    "Periodic (C=1500)":  "Periodic-1500",
    "Zero SWA Caching":   "Zero",
}


def _gb(b: int) -> float:
    return b / 1e9


def _plot_a_kv_storage_breakdown(results: dict, ax: plt.Axes) -> None:
    """Stacked bars: HBM state-cache components + on-disk components at each context."""
    seqs = [r["seq_len"] for r in results["layouts"]]
    labels = [_ctx_label(s) for s in seqs]
    x = np.arange(len(seqs))
    w = 0.38

    # Left bars: HBM state cache (CSA tail + HCA tail + SWA window)
    csa_t  = np.array([r["csa_tail_mb"] / 1e3      for r in results["layouts"]])
    hca_t  = np.array([r["hca_tail_mb"] / 1e3      for r in results["layouts"]])
    swa_s  = np.array([r["swa_state_gb"]            for r in results["layouts"]])

    # Right bars: on-disk (Full strategy) = CSA comp + HCA comp + SWA full
    csa_c  = np.array([r["csa_compressed_gb"]       for r in results["layouts"]])
    hca_c  = np.array([r["hca_compressed_gb"]       for r in results["layouts"]])
    swa_d  = np.array([r["swa_state_gb"]            for r in results["layouts"]])  # Full = swa_state

    # HBM left
    ax.bar(x - w/2, csa_t, w, color="#3498db", label="CSA tail (HBM)")
    ax.bar(x - w/2, hca_t, w, bottom=csa_t, color="#9b59b6", label="HCA tail (HBM)")
    ax.bar(x - w/2, swa_s, w, bottom=csa_t + hca_t, color="#1abc9c", label="SWA window (HBM)")

    # Disk right
    bot = np.zeros(len(seqs))
    ax.bar(x + w/2, csa_c, w, color="#2980b9", label="CSA compressed (disk)", alpha=0.85)
    bot += csa_c
    ax.bar(x + w/2, hca_c, w, bottom=bot, color="#8e44ad", label="HCA compressed (disk)", alpha=0.85)
    bot += hca_c
    ax.bar(x + w/2, swa_d, w, bottom=bot, color="#16a085", label="SWA full copy (disk)", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Storage (GB, log scale)")
    ax.set_title("A — KV Storage Breakdown\n(left: HBM state cache, right: on-disk Full strategy)")
    ax.legend(fontsize=7, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2g}"))


def _plot_b_disk_storage_vs_context(results: dict, ax: plt.Axes) -> None:
    """Line chart: on-disk GB vs context for all strategies + V3 + Llama baselines."""
    seqs   = [r["seq_len"] for r in results["layouts"]]
    labels = [_ctx_label(s) for s in seqs]

    by_strat: dict = {}
    for row in results["strategy_results"]:
        by_strat.setdefault(row["strategy"], []).append(row["disk_gb"])

    ls_map = {
        "Full SWA Caching":  "-",
        "Periodic (C=500)":  "--",
        "Periodic (C=1500)": "-.",
        "Zero SWA Caching":  ":",
    }
    for slab, vals in by_strat.items():
        ax.plot(range(len(seqs)), vals,
                label=_STRAT_LABELS_SHORT.get(slab, slab),
                color=_STRAT_COLORS.get(slab, "#888"),
                ls=ls_map.get(slab, "-"), marker="o", ms=4)

    v3_vals = [r["disk_gb"] for r in results["v3_baselines"]]
    l3_vals = [r["disk_gb"] for r in results["l3_baselines"]]
    ax.plot(range(len(seqs)), v3_vals, "k--", marker="s", ms=4, label="V3 (MLA)")
    ax.plot(range(len(seqs)), l3_vals, "k:",  marker="^", ms=4, label="Llama-3-70B (MHA)")

    ax.set_yscale("log")
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("On-Disk Storage (GB, log scale)")
    ax.set_title("B — On-Disk Storage vs Context Length")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2g}"))


def _plot_c_cache_hit_latency(results: dict, ax: plt.Axes) -> None:
    """Grouped bar chart: latency breakdown (disk + recompute + decode) per strategy × context."""
    sel_ctxs = LATENCY_CONTEXTS
    strat_order = ["Full SWA Caching", "Periodic (C=500)", "Periodic (C=1500)", "Zero SWA Caching"]

    # Gather data[ctx][strat] = (disk_ms, recomp_ms, decode_ms)
    data: dict = {}
    for row in results["strategy_results"]:
        if row["seq_len"] in sel_ctxs:
            data.setdefault(row["seq_len"], {})[row["strategy"]] = (
                row["disk_load_ms"], row["swa_recompute_ms"], row["first_decode_read_ms"]
            )

    n_ctx   = len(sel_ctxs)
    n_strat = len(strat_order)
    grp_w   = 0.8
    bar_w   = grp_w / n_strat
    ctx_labels = [_ctx_label(s) for s in sel_ctxs]

    for si, slab in enumerate(strat_order):
        color = _STRAT_COLORS[slab]
        disk_vals  = []
        recomp_vals = []
        decode_vals = []
        for S in sel_ctxs:
            d = data.get(S, {}).get(slab, (0, 0, 0))
            disk_vals.append(d[0])
            recomp_vals.append(d[1])
            decode_vals.append(d[2])
        xs = np.arange(n_ctx) + si * bar_w - grp_w / 2 + bar_w / 2
        disk_arr   = np.array(disk_vals)
        recomp_arr = np.array(recomp_vals)
        decode_arr = np.array(decode_vals)
        ax.bar(xs, disk_arr,   bar_w, color=color,  alpha=0.9, label=f"{_STRAT_LABELS_SHORT[slab]} disk")
        ax.bar(xs, recomp_arr, bar_w, bottom=disk_arr, color=color, alpha=0.55,
               label=f"{_STRAT_LABELS_SHORT[slab]} recomp", hatch="//")
        ax.bar(xs, decode_arr, bar_w, bottom=disk_arr + recomp_arr, color=color, alpha=0.35,
               label=f"{_STRAT_LABELS_SHORT[slab]} decode", hatch="..")

    ax.set_xticks(np.arange(n_ctx))
    ax.set_xticklabels(ctx_labels, fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("C — Cache-Hit Latency Breakdown by Strategy")
    # Compact legend: one entry per pattern type
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#555", alpha=0.9, label="Disk load"),
        Patch(facecolor="#555", alpha=0.55, hatch="//", label="SWA recompute"),
        Patch(facecolor="#555", alpha=0.35, hatch="..", label="First decode HBM read"),
    ] + [
        Patch(facecolor=_STRAT_COLORS[s], label=_STRAT_LABELS_SHORT[s])
        for s in strat_order
    ]
    ax.legend(handles=legend_elems, fontsize=7, ncol=2)


def _plot_d_pareto(results: dict, ax: plt.Axes) -> None:
    """Pareto scatter at PARETO_CONTEXT: disk GB vs total latency ms."""
    rows = results["pareto_sweep"]

    xs = [r["disk_gb"]        for r in rows]
    ys = [r["total_latency_ms"] for r in rows]

    colors = []
    edgecolors = []
    sizes = []
    for r in rows:
        if r["is_full"]:
            colors.append("#e74c3c"); edgecolors.append("darkred"); sizes.append(120)
        elif r["is_zero"]:
            colors.append("#2ecc71"); edgecolors.append("darkgreen"); sizes.append(120)
        else:
            colors.append("#f39c12"); edgecolors.append("#b7770d"); sizes.append(60)

    ax.scatter(xs, ys, c=colors, edgecolors=edgecolors, s=sizes, zorder=3)

    # Annotate Full and Zero
    for r in rows:
        if r["is_full"] or r["is_zero"]:
            ax.annotate(
                r["label"].replace(" SWA Caching", ""),
                (r["disk_gb"], r["total_latency_ms"]),
                textcoords="offset points", xytext=(6, 3), fontsize=8,
            )

    # Annotate a few Periodic points
    for r in rows:
        if not r["is_full"] and not r["is_zero"] and r["checkpoint_interval"] in (500, 1500, 3800):
            ax.annotate(
                f"C={r['checkpoint_interval']}",
                (r["disk_gb"], r["total_latency_ms"]),
                textcoords="offset points", xytext=(4, 3), fontsize=7, color="#555",
            )

    ax.set_xlabel("On-Disk Storage (GB)")
    ax.set_ylabel("Total Cache-Hit Latency (ms)")
    ax.set_title(f"D — Pareto: Disk vs Latency @ {_ctx_label(PARETO_CONTEXT)} ctx")
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markeredgecolor="darkred", markersize=10, label="Full SWA"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f39c12",
               markeredgecolor="#b7770d", markersize=8, label="Periodic(C)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markeredgecolor="darkgreen", markersize=10, label="Zero SWA"),
    ]
    ax.legend(handles=legend_elems, fontsize=8)


def plot_results(results: dict) -> None:
    """Generate and save all four plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "DeepSeek-V4-Pro SWA KV Cache Analysis\n"
        "(Analytical roofline model · A100 · TP=8 · InferLens)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    _plot_a_kv_storage_breakdown(results, axes[0, 0])
    _plot_b_disk_storage_vs_context(results, axes[0, 1])
    _plot_c_cache_hit_latency(results, axes[1, 0])
    _plot_d_pareto(results, axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for idx, name in enumerate(
        ["plot_a_kv_storage_breakdown",
         "plot_b_disk_storage_vs_context",
         "plot_c_cache_hit_latency",
         "plot_d_pareto_storage_vs_latency"]
    ):
        r, c = divmod(idx, 2)
        single_fig, single_ax = plt.subplots(figsize=(8, 5))
        # Rebuild individual plot
        [_plot_a_kv_storage_breakdown,
         _plot_b_disk_storage_vs_context,
         _plot_c_cache_hit_latency,
         _plot_d_pareto][idx](results, single_ax)
        single_fig.tight_layout()
        single_path = os.path.join(RESULTS_DIR, f"{name}.png")
        single_fig.savefig(single_path, dpi=140, bbox_inches="tight")
        plt.close(single_fig)
        print(f"  Saved: {single_path}")

    composite_path = os.path.join(RESULTS_DIR, "plots_composite.png")
    fig.savefig(composite_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {composite_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Running DeepSeek-V4-Pro SWA KV Cache Analysis …")
    results = compute_results()

    json_path = os.path.join(RESULTS_DIR, "deepseek_v4_swa_kv_analysis.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved: {json_path}")

    print_tables(results)

    print("Generating plots …")
    plot_results(results)
    print("Done.")


if __name__ == "__main__":
    main()
