"""Compare limited-LRU vs unlimited (no-eviction) KV cache in agentic inference.

Quantifies the **compute cost of thrashing** by asking: how much extra prefill
work does the server do because cache blocks were evicted and must be recomputed?

For every (concurrent-sessions × cache-size) configuration we run *two*
simulations on the identical request stream:

  • LIMITED   — standard LRU eviction when cache is full
  • UNLIMITED — oracle cache; blocks are never evicted (infinite capacity)

The gap between them is the performance penalty of operating under capacity.

Latency model
─────────────
  TTFT  ∝  effective_prefill_tokens  =  total_prefill  −  cached_tokens

  Only tokens *not* served from cache need to be (re-)computed during prefill.
  The TTFT multiplier for request i is:

      ttft_mult_i  =  eff_limited_i  /  eff_unlimited_i   (≥ 1)

  When ttft_mult = 4 the server spends 4× as long on prefill as it would with
  an unlimited cache.

Throughput model
────────────────
  Throughput  ∝  1 / total_effective_prefill

      throughput_ratio  =  total_eff_unlimited / total_eff_limited   (≤ 1)

  throughput_ratio = 0.25 means the limited cache delivers only ¼ of the
  throughput an unlimited cache would.

Plots generated
───────────────
  unlimited_overhead_heatmap.png    — compute overhead & throughput ratio vs
                                      (concurrent sessions × cache size)
  unlimited_detail_comparison.png   — 4-panel per-request view for the worst
                                      homogeneous case (8 conc, 400 blks)
  unlimited_ttft_cdf.png            — CDF of per-request TTFT multiplier for
                                      several representative configurations
  unlimited_cost_of_thrashing.png   — overhead vs working-set/cache ratio,
                                      showing the thrashing cliff
  unlimited_agent_mix_cost.png      — compute overhead by agent-mix for the
                                      heterogeneous configurations
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_ROOT, "report_figures", "kv_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from vidur.entities.prefix_cache_manager import PrefixCacheManager
from vidur.entities.request import Request

from experiment_thrashing import (
    # homogeneous workload
    generate_agentic_requests,
    SEED, BLOCK_SIZE, DECODE_TOKENS,
    CONCURRENT_SESSIONS_SWEEP, CACHE_BLOCKS_SWEEP,
    NUM_SESSIONS, STEPS_PER_SESSION, SUSTAINED_REPLACEMENTS,
    BLOCKS_PER_FULL_SESSION,
    # heterogeneous workload
    AgentTypeConfig,
    AGENT_SHORT, AGENT_MEDIUM, AGENT_LONG, AGENT_HIGH_VAR,
    HETERO_MIXES, HETERO_CONCURRENT, HETERO_CACHE,
    HETERO_NUM_SESSIONS, HETERO_SUSTAINED,
    generate_hetero_requests,
    _type_color,
)

# Effectively unlimited — far beyond any working set in these simulations
# (120 sessions × 470 max blocks/session ≈ 56 400 blocks)
UNLIMITED_BLOCKS = 500_000


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """Per-simulation metrics collected from one cache run."""
    label: str
    unlimited: bool
    concurrent_sessions: int
    cache_max_blocks: int

    total_requests: int
    total_prefill_tokens: int       # raw context size summed over all requests
    total_effective_prefill: int    # total_prefill − cached  =  tokens to compute
    total_cached_tokens: int
    total_blocks_evicted: int

    token_hit_rate: float
    request_hit_rate: float

    # Per-request parallel arrays
    prefill_tokens: List[int]       # raw prefill length for request i
    effective_prefill: List[int]    # tokens actually computed for request i
    hit_fractions: List[float]      # token_hit_fraction for request i

    @property
    def compute_saving_pct(self) -> float:
        return 100.0 * self.total_cached_tokens / max(1, self.total_prefill_tokens)

    @property
    def mean_effective_prefill(self) -> float:
        return self.total_effective_prefill / max(1, self.total_requests)


@dataclass
class GapMetrics:
    """Performance gap between a limited-cache run and the unlimited baseline."""
    config_label: str
    concurrent: int
    cache_blocks: int
    ws_ratio: float           # working_set_blocks / cache_blocks

    # Throughput / compute
    compute_overhead: float   # eff_limited / eff_unlimited  (≥ 1)
    throughput_ratio: float   # eff_unlimited / eff_limited  (≤ 1)

    # TTFT distribution
    mean_ttft_mult: float
    p50_ttft_mult: float
    p95_ttft_mult: float
    max_ttft_mult: float

    # Hit-rate gap
    hit_rate_limited: float
    hit_rate_unlimited: float
    hit_rate_gap_pp: float    # percentage points

    # Absolute waste
    extra_tokens_computed: int
    blocks_evicted: int


# ── Core simulation ───────────────────────────────────────────────────────────

def run_cache_simulation(
    requests: List[Tuple[Request, Tuple[int, ...]]],
    cache_max_blocks: int,
    label: str,
    concurrent_sessions: int,
    unlimited: bool,
) -> SimResult:
    """Run a pre-generated request stream through a cache of given capacity."""
    cache = PrefixCacheManager(max_blocks=cache_max_blocks, block_size=BLOCK_SIZE)

    prefill_list: List[int] = []
    effective_list: List[int] = []
    hit_frac_list: List[float] = []
    total_cached = 0

    for orig_req, token_ids in requests:
        req = Request(
            arrived_at=orig_req.arrived_at,
            num_prefill_tokens=len(token_ids),
            num_decode_tokens=DECODE_TOKENS,
        )
        cached_tokens = cache.match_prefix(token_ids)
        if cached_tokens > 0:
            req.apply_prefix_cache_hit(cached_tokens)
        cache.on_request_complete(token_ids)

        n = len(token_ids)
        eff = max(0, n - cached_tokens)
        hf = cached_tokens / n if n > 0 else 0.0

        prefill_list.append(n)
        effective_list.append(eff)
        hit_frac_list.append(hf)
        total_cached += cached_tokens

    stats = cache.stats
    return SimResult(
        label=label,
        unlimited=unlimited,
        concurrent_sessions=concurrent_sessions,
        cache_max_blocks=cache_max_blocks,
        total_requests=len(requests),
        total_prefill_tokens=sum(prefill_list),
        total_effective_prefill=sum(effective_list),
        total_cached_tokens=total_cached,
        total_blocks_evicted=stats.total_blocks_evicted,
        token_hit_rate=stats.token_hit_rate,
        request_hit_rate=stats.hit_rate,
        prefill_tokens=prefill_list,
        effective_prefill=effective_list,
        hit_fractions=hit_frac_list,
    )


def simulate_and_compare(
    concurrent_sessions: int,
    cache_max_blocks: int,
    num_sessions: int = NUM_SESSIONS,
    steps_per_session: int = STEPS_PER_SESSION,
    sustained_replacements: int = SUSTAINED_REPLACEMENTS,
) -> Tuple[SimResult, SimResult]:
    """Generate a homogeneous workload then run it through both cache types."""
    requests = generate_agentic_requests(
        num_sessions=num_sessions,
        steps_per_session=steps_per_session,
        concurrent_sessions=concurrent_sessions,
        sustained_replacements=sustained_replacements,
        seed=SEED,
    )
    lbl_base = f"{concurrent_sessions} conc"
    limited = run_cache_simulation(
        requests, cache_max_blocks,
        label=f"{lbl_base}, {cache_max_blocks} blks",
        concurrent_sessions=concurrent_sessions,
        unlimited=False,
    )
    unlim = run_cache_simulation(
        requests, UNLIMITED_BLOCKS,
        label=f"{lbl_base}, unlimited",
        concurrent_sessions=concurrent_sessions,
        unlimited=True,
    )
    return limited, unlim


def simulate_hetero_and_compare(
    agent_configs: List[Tuple[AgentTypeConfig, float]],
    concurrent_sessions: int,
    cache_max_blocks: int,
    num_sessions: int = HETERO_NUM_SESSIONS,
    sustained_replacements: int = HETERO_SUSTAINED,
) -> Tuple[SimResult, SimResult]:
    """Generate a heterogeneous workload then run it through both cache types."""
    requests, _ = generate_hetero_requests(
        agent_configs=agent_configs,
        concurrent_sessions=concurrent_sessions,
        num_sessions=num_sessions,
        sustained_replacements=sustained_replacements,
        seed=SEED,
    )
    mix_tag = "+".join(f"{int(f * 100)}%{c.name}" for c, f in agent_configs)
    limited = run_cache_simulation(
        requests, cache_max_blocks,
        label=f"{mix_tag}, {cache_max_blocks} blks",
        concurrent_sessions=concurrent_sessions,
        unlimited=False,
    )
    unlim = run_cache_simulation(
        requests, UNLIMITED_BLOCKS,
        label=f"{mix_tag}, unlimited",
        concurrent_sessions=concurrent_sessions,
        unlimited=True,
    )
    return limited, unlim


def compute_gap(
    limited: SimResult,
    unlimited: SimResult,
    config_label: str,
) -> GapMetrics:
    """Derive all gap metrics from a (limited, unlimited) pair."""
    eff_unl = max(1, unlimited.total_effective_prefill)
    overhead = limited.total_effective_prefill / eff_unl

    # Per-request TTFT multiplier; clamp denominator to avoid /0
    ttft_mults = np.array([
        lim / max(1, unl)
        for lim, unl in zip(limited.effective_prefill, unlimited.effective_prefill)
    ])

    ws = limited.concurrent_sessions * BLOCKS_PER_FULL_SESSION

    return GapMetrics(
        config_label=config_label,
        concurrent=limited.concurrent_sessions,
        cache_blocks=limited.cache_max_blocks,
        ws_ratio=ws / max(1, limited.cache_max_blocks),
        compute_overhead=overhead,
        throughput_ratio=1.0 / overhead,
        mean_ttft_mult=float(np.mean(ttft_mults)),
        p50_ttft_mult=float(np.percentile(ttft_mults, 50)),
        p95_ttft_mult=float(np.percentile(ttft_mults, 95)),
        max_ttft_mult=float(np.max(ttft_mults)),
        hit_rate_limited=limited.token_hit_rate,
        hit_rate_unlimited=unlimited.token_hit_rate,
        hit_rate_gap_pp=(unlimited.token_hit_rate - limited.token_hit_rate) * 100,
        extra_tokens_computed=max(
            0, limited.total_effective_prefill - unlimited.total_effective_prefill),
        blocks_evicted=limited.total_blocks_evicted,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_overhead_heatmap(
    gaps: Dict[Tuple[int, int], GapMetrics],
    concurrent_vals: List[int],
    cache_vals: List[int],
):
    """Side-by-side heatmaps: compute overhead (×) and throughput ratio (%)."""
    overhead_mat = []
    throughput_mat = []

    for cs in concurrent_vals:
        o_row, t_row = [], []
        for cb in cache_vals:
            g = gaps.get((cs, cb))
            o_row.append(g.compute_overhead if g else 1.0)
            t_row.append(g.throughput_ratio * 100 if g else 100.0)
        overhead_mat.append(o_row)
        throughput_mat.append(t_row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Compute overhead (higher = worse, red)
    im1 = ax1.imshow(overhead_mat, cmap="RdYlGn_r", aspect="auto",
                     vmin=1.0, vmax=6.0, origin="lower")
    ax1.set_xticks(range(len(cache_vals)))
    ax1.set_xticklabels([str(c) for c in cache_vals])
    ax1.set_yticks(range(len(concurrent_vals)))
    ax1.set_yticklabels([str(c) for c in concurrent_vals])
    ax1.set_xlabel("Cache Size (blocks)")
    ax1.set_ylabel("Concurrent Sessions")
    ax1.set_title("Compute Overhead vs Unlimited Cache (×)\n"
                  "How many more tokens to compute than with unlimited cache",
                  fontsize=10, fontweight="bold")
    for i, cs in enumerate(concurrent_vals):
        for j, cb in enumerate(cache_vals):
            val = overhead_mat[i][j]
            color = "white" if val > 3.5 else "black"
            ax1.text(j, i, f"{val:.1f}×", ha="center", va="center",
                     color=color, fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=ax1, label="Compute overhead (×)")

    # Throughput ratio (higher = better, green)
    im2 = ax2.imshow(throughput_mat, cmap="RdYlGn", aspect="auto",
                     vmin=15, vmax=100, origin="lower")
    ax2.set_xticks(range(len(cache_vals)))
    ax2.set_xticklabels([str(c) for c in cache_vals])
    ax2.set_yticks(range(len(concurrent_vals)))
    ax2.set_yticklabels([str(c) for c in concurrent_vals])
    ax2.set_xlabel("Cache Size (blocks)")
    ax2.set_ylabel("Concurrent Sessions")
    ax2.set_title("Throughput vs Unlimited Cache (%)\n"
                  "Fraction of unlimited-cache throughput delivered",
                  fontsize=10, fontweight="bold")
    for i, cs in enumerate(concurrent_vals):
        for j, cb in enumerate(cache_vals):
            val = throughput_mat[i][j]
            color = "white" if val < 35 else "black"
            ax2.text(j, i, f"{val:.0f}%", ha="center", va="center",
                     color=color, fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=ax2, label="Throughput (% of unlimited)")

    fig.suptitle("Cost of Cache Thrashing: Limited vs Unlimited KV Cache",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "unlimited_overhead_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved unlimited_overhead_heatmap.png")


def plot_detail_comparison(
    limited: SimResult,
    unlimited: SimResult,
    label: str,
    filename: str,
):
    """4-panel per-request view comparing limited vs unlimited for one config."""
    n = limited.total_requests
    xs = list(range(n))
    WINDOW = 15

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

    # ── Panel 1: token hit fraction ───────────────────────────────────────
    ax = axes[0]
    ax.bar(xs, [hf * 100 for hf in limited.hit_fractions],
           width=1.0, color="#F44336", alpha=0.45, label="Limited")
    ax.bar(xs, [hf * 100 for hf in unlimited.hit_fractions],
           width=1.0, color="#4CAF50", alpha=0.45, label="Unlimited")

    def _windowed(vals, w=WINDOW):
        out = []
        for i in range(len(vals)):
            sl = vals[max(0, i - w + 1):i + 1]
            out.append(sum(sl) / len(sl))
        return out

    ax.plot(xs, _windowed([hf * 100 for hf in limited.hit_fractions]),
            color="#C62828", linewidth=2, label=f"Limited {WINDOW}-req avg")
    ax.plot(xs, _windowed([hf * 100 for hf in unlimited.hit_fractions]),
            color="#1B5E20", linewidth=2, label=f"Unlimited {WINDOW}-req avg")
    ax.set_ylabel("Token Hit Fraction (%)")
    ax.set_title(f"Limited vs Unlimited Cache — {label}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: effective prefill (actual compute load) ──────────────────
    ax = axes[1]
    ax.fill_between(xs, limited.effective_prefill,
                    color="#F44336", alpha=0.5, label="Limited (tokens to compute)")
    ax.fill_between(xs, unlimited.effective_prefill,
                    color="#4CAF50", alpha=0.6, label="Unlimited (tokens to compute)")
    ax.set_ylabel("Effective Prefill (tokens)")
    ax.set_title("Tokens That Must Be Re-Computed (cache miss = full recompute)",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: per-request TTFT multiplier ─────────────────────────────
    ax = axes[2]
    ttft_mults = [
        lim / max(1, unl)
        for lim, unl in zip(limited.effective_prefill, unlimited.effective_prefill)
    ]
    colors = ["#F44336" if m > 2 else "#FF9800" if m > 1.2 else "#4CAF50"
              for m in ttft_mults]
    ax.bar(xs, ttft_mults, width=1.0, color=colors, alpha=0.7)
    ax.plot(xs, _windowed(ttft_mults), color="black", linewidth=2,
            label=f"{WINDOW}-req moving avg")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1,
               label="1× = unlimited speed")
    ax.set_ylabel("TTFT Multiplier (×)")
    ax.set_title("Per-Request Latency Inflation vs Unlimited Cache",
                 fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(ttft_mults) * 1.1 + 0.5)
    ax.grid(True, alpha=0.2)

    # ── Panel 4: cumulative compute overhead ──────────────────────────────
    ax = axes[3]
    cum_lim = np.cumsum(limited.effective_prefill)
    cum_unl = np.cumsum(unlimited.effective_prefill)
    cum_overhead = cum_lim / np.maximum(1, cum_unl)
    ax.plot(xs, cum_overhead, color="#F44336", linewidth=2,
            label="Cumulative compute overhead (×)")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1,
               label="1× baseline (unlimited)")
    ax.fill_between(xs, 1.0, cum_overhead,
                    where=cum_overhead >= 1.0,
                    color="#F44336", alpha=0.2, label="Wasted compute")
    ax.set_ylabel("Cumulative Overhead (×)")
    ax.set_xlabel("Request Index")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_ttft_cdf(
    gap_list: List[Tuple[GapMetrics, SimResult, SimResult]],
):
    """CDF of per-request TTFT multiplier for several configurations."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(gap_list)))
    for (gap, limited, unlim), color in zip(gap_list, colors):
        ttft_mults = np.array([
            lim / max(1, unl)
            for lim, unl in zip(limited.effective_prefill, unlim.effective_prefill)
        ])
        ttft_sorted = np.sort(ttft_mults)
        cdf = np.arange(1, len(ttft_sorted) + 1) / len(ttft_sorted) * 100
        ax.plot(ttft_sorted, cdf, color=color, linewidth=2,
                label=f"{gap.config_label}  (p95={gap.p95_ttft_mult:.1f}×)")

    ax.axvline(1.0, color="black", linestyle=":", linewidth=1.5,
               label="1× = same as unlimited")
    ax.set_xlabel("TTFT Multiplier vs Unlimited Cache (×)")
    ax.set_ylabel("CDF (%)")
    ax.set_title("Distribution of Per-Request Latency Inflation\n"
                 "Limited KV Cache vs Unlimited — Agentic Workloads",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 101)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "unlimited_ttft_cdf.png"), dpi=150)
    plt.close(fig)
    print("  Saved unlimited_ttft_cdf.png")


def plot_cost_of_thrashing(all_gaps: List[GapMetrics]):
    """Scatter: compute overhead vs working-set/cache ratio — the thrashing cliff."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    conc_vals = sorted(set(g.concurrent for g in all_gaps))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(conc_vals)))
    color_map = dict(zip(conc_vals, colors))

    for g in all_gaps:
        c = color_map[g.concurrent]
        ax1.scatter(g.ws_ratio, g.compute_overhead,
                    color=c, s=90, zorder=3,
                    edgecolors="white", linewidths=0.5)
        ax2.scatter(g.ws_ratio, g.throughput_ratio * 100,
                    color=c, s=90, zorder=3,
                    edgecolors="white", linewidths=0.5)

    # Legend by concurrent count
    for cs, c in color_map.items():
        ax1.scatter([], [], color=c, s=60, label=f"{cs} concurrent")

    ax1.axvline(1.0, color="black", linestyle="--", linewidth=1.2,
                label="WS = Cache capacity")
    ax1.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax1.set_xlabel("Working Set / Cache Capacity")
    ax1.set_ylabel("Compute Overhead (×)")
    ax1.set_title("Compute Overhead vs Cache Overcommit Ratio",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0.9)

    ax2.axvline(1.0, color="black", linestyle="--", linewidth=1.2,
                label="WS = Cache capacity")
    ax2.axhline(100, color="grey", linestyle=":", linewidth=1,
                label="100% = unlimited baseline")
    for cs, c in color_map.items():
        ax2.scatter([], [], color=c, s=60, label=f"{cs} concurrent")
    ax2.set_xlabel("Working Set / Cache Capacity")
    ax2.set_ylabel("Throughput vs Unlimited (%)")
    ax2.set_title("Throughput Retained vs Cache Overcommit Ratio",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("The Thrashing Cliff: Performance Collapses When "
                 "Working Set Exceeds Cache",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "unlimited_cost_of_thrashing.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved unlimited_cost_of_thrashing.png")


def plot_agent_mix_cost(
    hetero_gaps: Dict[str, Dict[int, GapMetrics]],
    cache_vals: List[int],
):
    """Grouped bar chart: compute overhead per agent mix at multiple cache sizes."""
    mix_names = list(hetero_gaps.keys())
    n_mixes = len(mix_names)
    n_sizes = len(cache_vals)

    x = np.arange(n_mixes)
    width = 0.7 / n_sizes
    bar_colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_sizes))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10))

    for j, (cb, color) in enumerate(zip(cache_vals, bar_colors)):
        overheads = []
        tputs = []
        for mn in mix_names:
            g = hetero_gaps[mn].get(cb)
            overheads.append(g.compute_overhead if g else 1.0)
            tputs.append(g.throughput_ratio * 100 if g else 100.0)
        offset = (j - n_sizes / 2 + 0.5) * width
        ax1.bar(x + offset, overheads, width, label=f"{cb} blocks",
                color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, tputs, width, label=f"{cb} blocks",
                color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax1.axhline(1.0, color="black", linestyle=":", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(mix_names, rotation=15, ha="right")
    ax1.set_ylabel("Compute Overhead vs Unlimited (×)")
    ax1.set_title("Compute Overhead by Agent Mix — "
                  f"{HETERO_CONCURRENT} Concurrent Sessions",
                  fontsize=11, fontweight="bold")
    ax1.legend(title="Cache size", fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(bottom=0.9)

    ax2.axhline(100, color="black", linestyle=":", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(mix_names, rotation=15, ha="right")
    ax2.set_ylabel("Throughput vs Unlimited (%)")
    ax2.set_title("Throughput Retained by Agent Mix",
                  fontsize=11, fontweight="bold")
    ax2.legend(title="Cache size", fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "unlimited_agent_mix_cost.png"),
                dpi=150)
    plt.close(fig)
    print("  Saved unlimited_agent_mix_cost.png")


# ── Console analysis ──────────────────────────────────────────────────────────

def print_gap_analysis(gap: GapMetrics):
    wasted_pct = (gap.extra_tokens_computed /
                  max(1, gap.extra_tokens_computed +
                      int(gap.compute_overhead and
                          gap.extra_tokens_computed / max(0.001, gap.compute_overhead - 1)))) * 100 \
        if gap.compute_overhead > 1 else 0.0
    # Simpler: wasted = 1 - 1/overhead
    wasted_frac = (1 - 1 / max(1.0, gap.compute_overhead)) * 100
    print(
        f"  {gap.config_label:40s} | "
        f"ws/cap={gap.ws_ratio:.1f}x | "
        f"overhead={gap.compute_overhead:.2f}x | "
        f"tput={gap.throughput_ratio * 100:.0f}% | "
        f"TTFT p50={gap.p50_ttft_mult:.1f}x p95={gap.p95_ttft_mult:.1f}x | "
        f"hit_gap={gap.hit_rate_gap_pp:.0f}pp | "
        f"wasted={wasted_frac:.0f}%compute"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 80)
    print("  UNLIMITED CACHE EXPERIMENT — COST OF THRASHING")
    print("=" * 80)
    print(f"\n  Unlimited proxy: {UNLIMITED_BLOCKS:,} blocks (never evicts)")
    print(f"  Latency model:  TTFT ∝ effective_prefill_tokens")
    print(f"  Throughput model: tput ∝ 1 / total_effective_prefill\n")

    # ── 1. Full homogeneous sweep ─────────────────────────────────────────
    print(f"{'─'*80}")
    print("  HOMOGENEOUS SWEEP: concurrent sessions × cache sizes")
    print(f"{'─'*80}")
    print(f"  {'Config':40s} | ws/cap | overhead | tput% | "
          f"TTFT p50 p95 | hit_gap | wasted")
    print(f"  {'─'*78}")

    all_gaps: Dict[Tuple[int, int], GapMetrics] = {}
    # Also keep raw results for the CDF and detail plots
    all_raw: Dict[Tuple[int, int], Tuple[SimResult, SimResult]] = {}

    for cs in CONCURRENT_SESSIONS_SWEEP:
        for cb in CACHE_BLOCKS_SWEEP:
            limited, unlim = simulate_and_compare(
                concurrent_sessions=cs, cache_max_blocks=cb)
            gap = compute_gap(
                limited, unlim,
                config_label=f"{cs} conc, {cb} blks")
            all_gaps[(cs, cb)] = gap
            all_raw[(cs, cb)] = (limited, unlim)
            print_gap_analysis(gap)

    # ── 2. Heatmap ────────────────────────────────────────────────────────
    print(f"\n  Generating heatmap...")
    plot_overhead_heatmap(
        all_gaps,
        concurrent_vals=CONCURRENT_SESSIONS_SWEEP,
        cache_vals=CACHE_BLOCKS_SWEEP,
    )

    # ── 3. Detail comparison for worst case ──────────────────────────────
    print("  Generating detail comparison (8 conc, 400 blks)...")
    lim_detail, unl_detail = all_raw[(8, 400)]
    plot_detail_comparison(
        lim_detail, unl_detail,
        label="8 concurrent, 400-block cache (3.4× overcommit)",
        filename="unlimited_detail_comparison.png",
    )

    # ── 4. TTFT CDF ───────────────────────────────────────────────────────
    print("  Generating TTFT CDF...")
    cdf_configs = [(2, 1200), (4, 800), (8, 600), (8, 400), (12, 400)]
    cdf_data = [
        (all_gaps[(cs, cb)], *all_raw[(cs, cb)])
        for cs, cb in cdf_configs
        if (cs, cb) in all_gaps
    ]
    plot_ttft_cdf(cdf_data)

    # ── 5. Cost-of-thrashing scatter ──────────────────────────────────────
    print("  Generating cost-of-thrashing scatter...")
    plot_cost_of_thrashing(list(all_gaps.values()))

    # ── 6. Heterogeneous: sweep all mixes × key cache sizes ──────────────
    print(f"\n{'─'*80}")
    print("  HETEROGENEOUS SWEEP: agent mixes × cache sizes")
    print(f"{'─'*80}")
    print(f"  {'Config':50s} | ws/cap | overhead | tput% | "
          f"TTFT p50 p95 | hit_gap")
    print(f"  {'─'*78}")

    HETERO_CACHE_SWEEP = [400, 800, 1200]
    hetero_gaps: Dict[str, Dict[int, GapMetrics]] = {}

    for mix_name, mix_cfgs in HETERO_MIXES:
        hetero_gaps[mix_name] = {}
        for cb in HETERO_CACHE_SWEEP:
            limited, unlim = simulate_hetero_and_compare(
                agent_configs=mix_cfgs,
                concurrent_sessions=HETERO_CONCURRENT,
                cache_max_blocks=cb,
            )
            gap = compute_gap(limited, unlim,
                              config_label=f"{mix_name}, {cb} blks")
            hetero_gaps[mix_name][cb] = gap
            print_gap_analysis(gap)

    # ── 7. Agent mix cost bar chart ───────────────────────────────────────
    print("\n  Generating agent mix cost chart...")
    plot_agent_mix_cost(hetero_gaps, HETERO_CACHE_SWEEP)

    # ── 8. Summary table ─────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  SUMMARY: COMPUTE WASTE DUE TO THRASHING (homogeneous, medium agents)")
    print(f"{'─'*80}")
    print(f"  {'Config':30s}  {'Overhead':>10}  {'Tput %':>8}  "
          f"{'TTFT p95':>10}  {'Hit gap':>9}  {'% waste':>8}")
    print(f"  {'─'*78}")
    for cs in CONCURRENT_SESSIONS_SWEEP:
        for cb in CACHE_BLOCKS_SWEEP:
            g = all_gaps.get((cs, cb))
            if g:
                waste_pct = (1 - 1 / max(1.0, g.compute_overhead)) * 100
                print(f"  {g.config_label:30s}  "
                      f"{g.compute_overhead:>9.2f}×  "
                      f"{g.throughput_ratio * 100:>7.0f}%  "
                      f"{g.p95_ttft_mult:>9.1f}×  "
                      f"{g.hit_rate_gap_pp:>8.0f}pp  "
                      f"{waste_pct:>7.0f}%")

    print(f"\n{'─'*80}")
    print("  SUMMARY: COMPUTE WASTE BY AGENT MIX (at 800 blocks, 8 concurrent)")
    print(f"{'─'*80}")
    print(f"  {'Mix':15s}  {'Overhead':>10}  {'Tput %':>8}  "
          f"{'TTFT p95':>10}  {'Hit gap':>9}  {'% waste':>8}")
    print(f"  {'─'*62}")
    for mix_name in hetero_gaps:
        g = hetero_gaps[mix_name].get(800)
        if g:
            waste_pct = (1 - 1 / max(1.0, g.compute_overhead)) * 100
            print(f"  {mix_name:15s}  "
                  f"{g.compute_overhead:>9.2f}×  "
                  f"{g.throughput_ratio * 100:>7.0f}%  "
                  f"{g.p95_ttft_mult:>9.1f}×  "
                  f"{g.hit_rate_gap_pp:>8.0f}pp  "
                  f"{waste_pct:>7.0f}%")


if __name__ == "__main__":
    main()
