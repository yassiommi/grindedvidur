"""Experiment: PCIe KV cache reloading vs recomputation during thrashing.

When KV cache thrashing occurs (working set > HBM capacity), evicted blocks
are normally lost and must be recomputed from scratch via prefill. This
experiment models a tiered memory hierarchy:

    Tier 1 (HBM)  : Small, fast — the GPU's on-chip KV cache
    Tier 2 (DRAM) : Larger, accessible via PCIe — host memory backup
                    (conceptually similar for NVMe disk, just slower)

When a prefix lookup misses in HBM but hits in DRAM, the KV cache is
reloaded via PCIe instead of recomputed. This trades PCIe bandwidth for
saved GPU compute cycles.

Three outcomes per prefix lookup:
    1. HBM hit        → 0 extra cost (tokens already on GPU)
    2. HBM miss,      → PCIe reload cost (cheaper than recompute)
       DRAM hit
    3. Full miss      → recompute cost (full prefill required)

We compare two cost models over the same workload:
    Baseline : recompute all tokens not found in HBM
    Tiered   : PCIe-reload DRAM-resident tokens + recompute only true misses

Sweeps:
    - Concurrent agent sessions × HBM cache size (same as thrashing exp.)
    - DRAM tier capacity (multiples of HBM)
    - PCIe bandwidth (Gen3 / Gen4 / Gen5 / CXL-like)
    - Storage tiers (DRAM via PCIe vs NVMe disk)
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

# Reuse the exact workload from the thrashing experiment for a fair comparison.
from experiment_thrashing import (
    BLOCK_SIZE,
    SEED,
    DECODE_TOKENS,
    SYSTEM_PROMPT_TOKENS,
    TASK_PROMPT_TOKENS,
    TOKENS_PER_STEP,
    STEPS_PER_SESSION,
    NUM_SESSIONS,
    MAX_SESSION_TOKENS,
    BLOCKS_PER_FULL_SESSION,
    SUSTAINED_REPLACEMENTS,
    CONCURRENT_SESSIONS_SWEEP,
    CACHE_BLOCKS_SWEEP,
    generate_agentic_requests,
)


# ── Hardware / cost-model configuration ─────────────────────────────────────

@dataclass
class HardwareConfig:
    """Hardware parameters used by the cost model.

    The experiment cares about two per-token cost values:
      - prefill_ms_per_token : what it costs to recompute one token of KV
      - pcie_ms_per_token    : what it costs to reload one token of KV via PCIe

    Both are linear approximations (the prefill attention is technically
    O(n^2), but for marginal-cost comparisons a per-token rate is the cleaner
    apples-to-apples metric — the baseline cost in the same model also uses
    per-token prefix-cache savings).
    """
    name: str
    kv_bytes_per_token: int        # total KV cache bytes per token (all layers)
    prefill_ms_per_token: float    # time to compute one token's KV via prefill
    pcie_bandwidth_gb_s: float     # raw PCIe bandwidth (GB/s)
    bw_efficiency: float = 0.8     # realised / peak bandwidth ratio

    @property
    def effective_pcie_bw_bytes_s(self) -> float:
        return self.pcie_bandwidth_gb_s * self.bw_efficiency * 1e9

    @property
    def pcie_ms_per_token(self) -> float:
        """Time to load one token's KV cache via PCIe (ms)."""
        if self.effective_pcie_bw_bytes_s <= 0:
            return float("inf")
        return (self.kv_bytes_per_token / self.effective_pcie_bw_bytes_s) * 1e3

    @property
    def speedup_ratio(self) -> float:
        """How many times faster PCIe reload is vs recomputation."""
        p = self.pcie_ms_per_token
        return self.prefill_ms_per_token / p if p > 0 else float("inf")


# Reference configurations.
#
# Llama-2-7B — 32 layers, 32 KV heads, head_dim=128, fp16:
#     KV/token = 2(K+V) × 32 × 32 × 128 × 2 = 524,288 B  ≈  512 KB/token
#     Prefill  ≈ 14 GFLOPs/token  (2 × 7B)
#       on A100 (312 TFLOPS peak, ~175 eff.):  14e9 / 175e12 ≈ 0.080 ms
#       on H100 (~500 eff. TFLOPS):            14e9 / 500e12 ≈ 0.028 ms
HW_A100_7B = HardwareConfig(
    name="A100 + Llama-2-7B",
    kv_bytes_per_token=524_288,
    prefill_ms_per_token=0.080,
    pcie_bandwidth_gb_s=31.5,      # PCIe Gen4 x16
)

HW_H100_7B = HardwareConfig(
    name="H100 + Llama-2-7B",
    kv_bytes_per_token=524_288,
    prefill_ms_per_token=0.028,
    pcie_bandwidth_gb_s=64.0,      # PCIe Gen5 x16
)

# Llama-2-70B (GQA) — 80 layers, 8 KV heads, head_dim=128, fp16:
#     KV/token = 2 × 80 × 8 × 128 × 2 = 327,680 B  ≈  320 KB/token
#     Prefill  ≈ 140 GFLOPs/token  (with TP you divide, but compute stays
#                                    proportional — using TP=1 equivalent)
HW_A100_70B = HardwareConfig(
    name="A100 + Llama-2-70B (GQA)",
    kv_bytes_per_token=327_680,
    prefill_ms_per_token=0.800,
    pcie_bandwidth_gb_s=31.5,
)

DEFAULT_HW = HW_A100_7B

# Tier-2 capacity (as multiples of HBM cache).
DRAM_MULTIPLIERS = [2, 5, 10, 50]

# PCIe bandwidths to sweep (GB/s). Includes NVMe-SSD bandwidth as a "disk tier".
PCIE_BW_SWEEP: List[Tuple[str, float]] = [
    ("NVMe SSD (7 GB/s)", 7.0),
    ("PCIe Gen3 x16 (16 GB/s)", 16.0),
    ("PCIe Gen4 x16 (31.5 GB/s)", 31.5),
    ("PCIe Gen5 x16 (64 GB/s)", 64.0),
    ("CXL/future (128 GB/s)", 128.0),
]


# ── Trace data class ────────────────────────────────────────────────────────

@dataclass
class PCIeReloadTrace:
    """Per-request trace for a tiered-cache simulation."""
    request_idx: int
    session_id: int
    step_in_session: int
    arrived_at: float
    num_prefill_tokens: int

    # Tier breakdown (tokens)
    hbm_cached_tokens: int        # free (on GPU already)
    dram_cached_tokens: int       # raw DRAM lookup result
    pcie_reload_tokens: int       # in DRAM but not HBM → PCIe transfer
    recompute_tokens: int         # not in either tier → full prefill

    # Cost comparison (ms)
    baseline_cost_ms: float       # recompute all non-HBM-cached tokens
    tiered_cost_ms: float         # PCIe reload + recompute remainder
    savings_ms: float
    savings_pct: float

    # Cumulative
    cum_baseline_ms: float
    cum_tiered_ms: float
    cum_savings_ms: float
    cum_savings_pct: float

    # Token fractions (of num_prefill_tokens)
    hbm_hit_frac: float
    pcie_frac: float
    miss_frac: float

    # Cache state
    hbm_cached_blocks: int
    hbm_max_blocks: int
    hbm_utilization: float
    dram_cached_blocks: int
    dram_max_blocks: int
    dram_utilization: float

    # Working-set estimate (same heuristic as experiment_thrashing.py)
    estimated_working_set_blocks: int


# ── Simulation core ─────────────────────────────────────────────────────────

def _replay_assignments(
    concurrent_sessions: int,
    num_sessions: int,
    steps_per_session: int,
    sustained_replacements: int,
) -> List[Tuple[int, int]]:
    """Re-derive (session_id, step) for every request produced by
    generate_agentic_requests(), by replaying the same three-phase logic.

    This must stay in lockstep with experiment_thrashing.generate_agentic_requests.
    """
    active: List[Tuple[int, int]] = []
    next_sid = 0
    repl_remaining = sustained_replacements
    assignments: List[Tuple[int, int]] = []

    def _take() -> Optional[int]:
        nonlocal next_sid
        if next_sid < num_sessions:
            sid = next_sid
            next_sid += 1
            return sid
        return None

    # Phase 1: ramp-up
    ramp_tick = 2
    ticks_since = ramp_tick
    while len(active) < concurrent_sessions:
        if ticks_since >= ramp_tick:
            sid = _take()
            if sid is None:
                break
            active.append((sid, 0))
            ticks_since = 0
        next_active: List[Tuple[int, int]] = []
        for sid, step in active:
            assignments.append((sid, step))
            if step + 1 <= steps_per_session:
                next_active.append((sid, step + 1))
        active = next_active
        ticks_since += 1

    # Phase 2: sustained
    while active and repl_remaining >= 0:
        next_active = []
        for sid, step in active:
            assignments.append((sid, step))
            if step + 1 <= steps_per_session:
                next_active.append((sid, step + 1))
            else:
                if repl_remaining > 0:
                    new_sid = _take()
                    if new_sid is not None:
                        next_active.append((new_sid, 0))
                        repl_remaining -= 1
                    else:
                        repl_remaining = -1
                else:
                    repl_remaining = -1
        active = next_active

    # Phase 3: drain
    while active:
        next_active = []
        for sid, step in active:
            assignments.append((sid, step))
            if step + 1 <= steps_per_session:
                next_active.append((sid, step + 1))
        active = next_active

    return assignments


def simulate_pcie_reload(
    concurrent_sessions: int,
    hbm_cache_blocks: int,
    dram_cache_blocks: int,
    hw: HardwareConfig = DEFAULT_HW,
    num_sessions: int = NUM_SESSIONS,
    steps_per_session: int = STEPS_PER_SESSION,
    sustained_replacements: int = SUSTAINED_REPLACEMENTS,
) -> List[PCIeReloadTrace]:
    """Run one tiered-cache simulation.

    Two independent PrefixCacheManagers are maintained over the same workload:
      • hbm_cache  — small (tier 1)
      • dram_cache — large (tier 2)
    Both see identical insertions on request completion, so the DRAM cache is
    effectively a superset of the HBM cache whenever it has the capacity.

    On each request we look up both; the HBM result tells us what's free,
    the (effective) DRAM result tells us how many additional tokens can be
    reloaded over PCIe rather than recomputed.
    """
    requests = generate_agentic_requests(
        num_sessions=num_sessions,
        steps_per_session=steps_per_session,
        concurrent_sessions=concurrent_sessions,
        sustained_replacements=sustained_replacements,
        seed=SEED,
    )
    assignments = _replay_assignments(
        concurrent_sessions=concurrent_sessions,
        num_sessions=num_sessions,
        steps_per_session=steps_per_session,
        sustained_replacements=sustained_replacements,
    )

    hbm_cache = PrefixCacheManager(
        max_blocks=hbm_cache_blocks, block_size=BLOCK_SIZE)
    dram_cache = PrefixCacheManager(
        max_blocks=dram_cache_blocks, block_size=BLOCK_SIZE)

    traces: List[PCIeReloadTrace] = []
    cum_baseline = 0.0
    cum_tiered = 0.0

    prefill_cost = hw.prefill_ms_per_token
    pcie_cost = hw.pcie_ms_per_token

    for i, ((req, token_ids), (sid, step)) in enumerate(
            zip(requests, assignments)):
        total_tokens = len(token_ids)

        hbm_cached = hbm_cache.match_prefix(token_ids)
        dram_cached = dram_cache.match_prefix(token_ids)

        # A DRAM hit is only useful up to at least as many tokens as HBM.
        # Both caches are prefix-aligned and block-aligned, and DRAM is
        # almost always a superset, so dram_cached >= hbm_cached. Guard
        # with max() in case LRU corner cases cause divergence.
        effective_cached = max(hbm_cached, dram_cached)
        pcie_reload_tokens = effective_cached - hbm_cached
        recompute_tokens = total_tokens - effective_cached

        # Baseline: no DRAM tier, recompute everything that's not in HBM.
        baseline_recompute = total_tokens - hbm_cached
        baseline_cost = baseline_recompute * prefill_cost

        # Tiered: PCIe-load the DRAM-only portion, recompute the rest.
        tiered_cost = (pcie_reload_tokens * pcie_cost
                       + recompute_tokens * prefill_cost)

        savings = baseline_cost - tiered_cost
        savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        cum_baseline += baseline_cost
        cum_tiered += tiered_cost
        cum_savings = cum_baseline - cum_tiered
        cum_savings_pct = (cum_savings / cum_baseline * 100
                           if cum_baseline > 0 else 0.0)

        if total_tokens > 0:
            hbm_frac = hbm_cached / total_tokens
            pcie_frac = pcie_reload_tokens / total_tokens
            miss_frac = recompute_tokens / total_tokens
        else:
            hbm_frac = pcie_frac = miss_frac = 0.0

        # Working-set estimate (same heuristic as the thrashing experiment).
        tick_start = max(0, i - (i % max(1, concurrent_sessions)))
        tick_end = min(len(assignments), tick_start + concurrent_sessions)
        ws_blocks = 0
        for j in range(tick_start, tick_end):
            _, s = assignments[j]
            ctx = SYSTEM_PROMPT_TOKENS + TASK_PROMPT_TOKENS + s * TOKENS_PER_STEP
            ws_blocks += math.ceil(ctx / BLOCK_SIZE)

        if hbm_cached > 0:
            req.apply_prefix_cache_hit(hbm_cached)

        hbm_cache.on_request_complete(token_ids)
        dram_cache.on_request_complete(token_ids)

        traces.append(PCIeReloadTrace(
            request_idx=i,
            session_id=sid,
            step_in_session=step,
            arrived_at=req.arrived_at,
            num_prefill_tokens=total_tokens,
            hbm_cached_tokens=hbm_cached,
            dram_cached_tokens=dram_cached,
            pcie_reload_tokens=pcie_reload_tokens,
            recompute_tokens=recompute_tokens,
            baseline_cost_ms=baseline_cost,
            tiered_cost_ms=tiered_cost,
            savings_ms=savings,
            savings_pct=savings_pct,
            cum_baseline_ms=cum_baseline,
            cum_tiered_ms=cum_tiered,
            cum_savings_ms=cum_savings,
            cum_savings_pct=cum_savings_pct,
            hbm_hit_frac=hbm_frac,
            pcie_frac=pcie_frac,
            miss_frac=miss_frac,
            hbm_cached_blocks=hbm_cache.num_cached_blocks,
            hbm_max_blocks=hbm_cache_blocks,
            hbm_utilization=hbm_cache.cache_utilization,
            dram_cached_blocks=dram_cache.num_cached_blocks,
            dram_max_blocks=dram_cache_blocks,
            dram_utilization=dram_cache.cache_utilization,
            estimated_working_set_blocks=ws_blocks,
        ))

    return traces


# ── Summary helpers ─────────────────────────────────────────────────────────

@dataclass
class TierSummary:
    """Aggregate stats for one simulation run."""
    num_requests: int
    total_tokens: int
    hbm_hit_tokens: int
    pcie_reload_tokens: int
    recompute_tokens: int
    baseline_total_ms: float
    tiered_total_ms: float
    savings_total_ms: float
    savings_pct: float
    avg_baseline_ms_per_req: float
    avg_tiered_ms_per_req: float

    @property
    def hbm_hit_frac(self) -> float:
        return self.hbm_hit_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def pcie_frac(self) -> float:
        return self.pcie_reload_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def miss_frac(self) -> float:
        return self.recompute_tokens / self.total_tokens if self.total_tokens else 0.0


def summarize(traces: List[PCIeReloadTrace]) -> TierSummary:
    n = len(traces)
    if n == 0:
        return TierSummary(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_tokens = sum(t.num_prefill_tokens for t in traces)
    hbm_tok = sum(t.hbm_cached_tokens for t in traces)
    pcie_tok = sum(t.pcie_reload_tokens for t in traces)
    reco_tok = sum(t.recompute_tokens for t in traces)
    base_total = traces[-1].cum_baseline_ms
    tier_total = traces[-1].cum_tiered_ms
    savings = base_total - tier_total
    pct = (savings / base_total * 100) if base_total > 0 else 0.0
    return TierSummary(
        num_requests=n,
        total_tokens=total_tokens,
        hbm_hit_tokens=hbm_tok,
        pcie_reload_tokens=pcie_tok,
        recompute_tokens=reco_tok,
        baseline_total_ms=base_total,
        tiered_total_ms=tier_total,
        savings_total_ms=savings,
        savings_pct=pct,
        avg_baseline_ms_per_req=base_total / n,
        avg_tiered_ms_per_req=tier_total / n,
    )


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_pcie_reload_detail(
    traces: List[PCIeReloadTrace],
    hw: HardwareConfig,
    label: str,
    filename: str,
):
    """5-panel deep-dive for a single configuration."""
    if not traces:
        return
    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    xs = [t.request_idx for t in traces]

    # ── Panel 1: Token-tier breakdown (stacked area) ────────────────────────
    ax = axes[0]
    hbm_pct = [t.hbm_hit_frac * 100 for t in traces]
    pcie_pct = [t.pcie_frac * 100 for t in traces]
    miss_pct = [t.miss_frac * 100 for t in traces]
    ax.stackplot(
        xs, hbm_pct, pcie_pct, miss_pct,
        labels=["HBM hit (free)",
                "PCIe reload (DRAM → HBM)",
                "Recompute (prefill)"],
        colors=["#4CAF50", "#2196F3", "#F44336"],
        alpha=0.85,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Token disposition (%)")
    ax.set_title(
        f"PCIe KV Reload vs Recompute — {label}\n"
        f"{hw.name} | prefill {hw.prefill_ms_per_token:.3f} ms/tok  "
        f"vs  PCIe {hw.pcie_ms_per_token:.3f} ms/tok "
        f"({hw.speedup_ratio:.1f}× cheaper)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Per-request cost (baseline vs tiered) ──────────────────────
    ax = axes[1]
    ax.bar(xs, [t.baseline_cost_ms for t in traces],
           width=1.0, color="#F44336", alpha=0.55,
           label="Baseline (recompute)")
    ax.bar(xs, [t.tiered_cost_ms for t in traces],
           width=1.0, color="#2196F3", alpha=0.85,
           label="Tiered (PCIe reload + recompute)")
    ax.set_ylabel("Cost per request (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Per-request savings (ms) with moving average ──────────────
    ax = axes[2]
    savings = [t.savings_ms for t in traces]
    ax.bar(xs, savings, width=1.0, color="#4CAF50", alpha=0.6, label="Savings")
    WINDOW = 15
    windowed = []
    for i in range(len(traces)):
        start = max(0, i - WINDOW + 1)
        w = savings[start:i + 1]
        windowed.append(sum(w) / len(w))
    ax.plot(xs, windowed, color="black", linewidth=2,
            label=f"{WINDOW}-req moving avg")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Savings per request (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 4: Cumulative cost — baseline vs tiered ──────────────────────
    ax = axes[3]
    ax.plot(xs, [t.cum_baseline_ms / 1000.0 for t in traces],
            color="#F44336", linewidth=2,
            label="Baseline (cumulative, s)")
    ax.plot(xs, [t.cum_tiered_ms / 1000.0 for t in traces],
            color="#2196F3", linewidth=2,
            label="Tiered (cumulative, s)")
    ax.fill_between(
        xs,
        [t.cum_tiered_ms / 1000.0 for t in traces],
        [t.cum_baseline_ms / 1000.0 for t in traces],
        color="#4CAF50", alpha=0.2, label="Saved compute")
    ax.set_ylabel("Cumulative compute time (s)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 5: Cumulative savings % + working set vs capacity ─────────────
    ax = axes[4]
    ax.plot(xs, [t.cum_savings_pct for t in traces],
            color="#4CAF50", linewidth=2, label="Cumulative savings (%)")
    ax.set_ylabel("Cumulative savings (%)")
    ax.set_ylim(-2, 102)
    ax.set_xlabel("Request index")
    ax.grid(True, alpha=0.2)

    # Shade thrashing regions (right y-axis)
    ax2 = ax.twinx()
    ax2.plot(xs, [t.estimated_working_set_blocks for t in traces],
             color="#F44336", linewidth=1.2, linestyle="--",
             alpha=0.7, label="Working set (blocks)")
    ax2.axhline(traces[0].hbm_max_blocks, color="black", linestyle=":",
                linewidth=1, label=f"HBM capacity ({traces[0].hbm_max_blocks})")
    ax2.set_ylabel("Blocks")
    thrash_start = None
    for t in traces:
        if (t.estimated_working_set_blocks > t.hbm_max_blocks
                and thrash_start is None):
            thrash_start = t.request_idx
        elif (t.estimated_working_set_blocks <= t.hbm_max_blocks
              and thrash_start is not None):
            ax.axvspan(thrash_start, t.request_idx, alpha=0.08, color="red")
            thrash_start = None
    if thrash_start is not None:
        ax.axvspan(thrash_start, xs[-1], alpha=0.08, color="red")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_savings_heatmap(
    summaries: Dict[Tuple[int, int], TierSummary],
    hw: HardwareConfig,
    filename: str = "pcie_reload_savings_heatmap.png",
):
    """Heatmap: overall compute-time savings (%) vs (concurrent, HBM size)."""
    concurrent_vals = sorted({k[0] for k in summaries})
    cache_vals = sorted({k[1] for k in summaries})

    matrix: List[List[float]] = []
    for cs in concurrent_vals:
        row = []
        for cb in cache_vals:
            s = summaries.get((cs, cb))
            row.append(s.savings_pct if s else 0.0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100,
                   origin="lower")
    ax.set_xticks(range(len(cache_vals)))
    ax.set_xticklabels([str(c) for c in cache_vals])
    ax.set_yticks(range(len(concurrent_vals)))
    ax.set_yticklabels([str(c) for c in concurrent_vals])
    ax.set_xlabel("HBM Cache Size (blocks)")
    ax.set_ylabel("Concurrent Agent Sessions")
    ax.set_title(
        f"Prefill-Compute Savings From PCIe Reload (%)\n"
        f"{hw.name} | DRAM tier = 10× HBM",
        fontsize=12, fontweight="bold",
    )
    for i, cs in enumerate(concurrent_vals):
        for j, cb in enumerate(cache_vals):
            val = matrix[i][j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    # Mark configurations whose working set fits in HBM (no thrashing, no win).
    for i, cs in enumerate(concurrent_vals):
        ws = cs * BLOCKS_PER_FULL_SESSION
        for j, cb in enumerate(cache_vals):
            if cb >= ws:
                ax.plot(j, i, "ko", markersize=10, markerfacecolor="none",
                        markeredgewidth=2)
                break

    fig.colorbar(im, ax=ax, label="Savings (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_dram_size_sweep(
    results: Dict[int, List[PCIeReloadTrace]],
    hw: HardwareConfig,
    label: str,
    filename: str = "pcie_reload_dram_sweep.png",
):
    """Shows how cumulative savings % evolves as the DRAM tier grows."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    n_lines = max(1, len(results))
    colors = plt.cm.viridis([i / n_lines for i in range(n_lines)])

    sorted_keys = sorted(results.keys())
    for mult, color in zip(sorted_keys, colors):
        traces = results[mult]
        if not traces:
            continue
        xs = [t.request_idx for t in traces]
        ax1.plot(xs, [t.cum_savings_pct for t in traces],
                 color=color, linewidth=2,
                 label=f"DRAM = {mult}× HBM")
        # Stacked breakdown of token dispositions (line form)
        ax2.plot(xs, [t.pcie_frac * 100 for t in traces],
                 color=color, linewidth=1.8,
                 label=f"DRAM = {mult}× HBM")

    ax1.set_ylabel("Cumulative compute savings (%)")
    ax1.set_title(
        f"DRAM Tier Size Sweep — {label}\n"
        f"{hw.name} | PCIe @ {hw.pcie_bandwidth_gb_s} GB/s "
        f"({hw.speedup_ratio:.1f}× vs recompute)",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Tokens served via PCIe reload (%)")
    ax2.set_xlabel("Request index")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_pcie_bandwidth_sweep(
    summaries: List[Tuple[str, float, TierSummary]],
    hw: HardwareConfig,
    filename: str = "pcie_reload_bandwidth_sweep.png",
):
    """Bar chart: how the tier-2 bandwidth affects savings and absolute cost."""
    names = [n for n, _, _ in summaries]
    bws = [b for _, b, _ in summaries]
    savings = [s.savings_pct for _, _, s in summaries]
    baseline_ms = [s.baseline_total_ms for _, _, s in summaries]
    tiered_ms = [s.tiered_total_ms for _, _, s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    xs = list(range(len(names)))
    bars = ax1.bar(xs, savings, color="#4CAF50", alpha=0.85)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Compute savings (%)")
    ax1.set_title("Tier-2 Bandwidth vs Prefill-Compute Savings",
                  fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(100, max(savings) + 10) if savings else 100)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, v, bw in zip(bars, savings, bws):
        kv_ms = hw.kv_bytes_per_token / (bw * hw.bw_efficiency * 1e9) * 1e3
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1,
                 f"{v:.1f}%\n({kv_ms:.3f} ms/tok)",
                 ha="center", va="bottom", fontsize=8)

    width = 0.38
    ax2.bar([x - width / 2 for x in xs],
            [b / 1000 for b in baseline_ms],
            width=width, color="#F44336", alpha=0.75, label="Baseline (recompute)")
    ax2.bar([x + width / 2 for x in xs],
            [t / 1000 for t in tiered_ms],
            width=width, color="#2196F3", alpha=0.9, label="Tiered (PCIe reload)")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Total prefill compute (s)")
    ax2.set_title("Absolute Compute Time (full workload)",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


# ── Console analysis ────────────────────────────────────────────────────────

def print_tier_summary(summary: TierSummary, label: str) -> None:
    print(f"\n  ── {label} ──")
    print(f"  Requests: {summary.num_requests}, "
          f"Total tokens: {summary.total_tokens:,}")
    print(f"  Tier breakdown (by token):")
    print(f"    HBM hit    : {summary.hbm_hit_frac:6.1%}  "
          f"({summary.hbm_hit_tokens:>8,} tokens)")
    print(f"    PCIe reload: {summary.pcie_frac:6.1%}  "
          f"({summary.pcie_reload_tokens:>8,} tokens)")
    print(f"    Recompute  : {summary.miss_frac:6.1%}  "
          f"({summary.recompute_tokens:>8,} tokens)")
    print(f"  Compute cost:")
    print(f"    Baseline : {summary.baseline_total_ms:>10,.1f} ms "
          f"({summary.avg_baseline_ms_per_req:6.2f} ms/req)")
    print(f"    Tiered   : {summary.tiered_total_ms:>10,.1f} ms "
          f"({summary.avg_tiered_ms_per_req:6.2f} ms/req)")
    print(f"    Savings  : {summary.savings_total_ms:>10,.1f} ms "
          f"({summary.savings_pct:5.1f}%)")


def print_hardware_info(hw: HardwareConfig) -> None:
    kv_kb = hw.kv_bytes_per_token / 1024.0
    print(f"\n  Hardware: {hw.name}")
    print(f"    KV cache        : {kv_kb:,.0f} KB/token "
          f"({hw.kv_bytes_per_token:,} bytes)")
    print(f"    Prefill cost    : {hw.prefill_ms_per_token:.4f} ms/token")
    print(f"    PCIe bandwidth  : {hw.pcie_bandwidth_gb_s} GB/s raw "
          f"(× {hw.bw_efficiency:.1f} eff. = "
          f"{hw.pcie_bandwidth_gb_s * hw.bw_efficiency:.1f} GB/s)")
    print(f"    PCIe reload     : {hw.pcie_ms_per_token:.4f} ms/token")
    print(f"    Reload speedup  : {hw.speedup_ratio:.1f}× cheaper than recompute")


# ── Results JSON dump ───────────────────────────────────────────────────────

def dump_results_json(
    path: str,
    hw: HardwareConfig,
    summaries: Dict[Tuple[int, int], TierSummary],
    dram_sweep: Dict[int, TierSummary],
    pcie_sweep: List[Tuple[str, float, TierSummary]],
    hw_comparison: List[Tuple[HardwareConfig, TierSummary]],
) -> None:
    """Persist summary results to JSON for the report."""
    def _s(summary: TierSummary) -> dict:
        return {
            "num_requests": summary.num_requests,
            "total_tokens": summary.total_tokens,
            "hbm_hit_tokens": summary.hbm_hit_tokens,
            "pcie_reload_tokens": summary.pcie_reload_tokens,
            "recompute_tokens": summary.recompute_tokens,
            "hbm_hit_frac": summary.hbm_hit_frac,
            "pcie_frac": summary.pcie_frac,
            "miss_frac": summary.miss_frac,
            "baseline_total_ms": summary.baseline_total_ms,
            "tiered_total_ms": summary.tiered_total_ms,
            "savings_total_ms": summary.savings_total_ms,
            "savings_pct": summary.savings_pct,
            "avg_baseline_ms_per_req": summary.avg_baseline_ms_per_req,
            "avg_tiered_ms_per_req": summary.avg_tiered_ms_per_req,
        }

    data = {
        "hardware": {
            "name": hw.name,
            "kv_bytes_per_token": hw.kv_bytes_per_token,
            "prefill_ms_per_token": hw.prefill_ms_per_token,
            "pcie_bandwidth_gb_s": hw.pcie_bandwidth_gb_s,
            "pcie_ms_per_token": hw.pcie_ms_per_token,
            "speedup_ratio": hw.speedup_ratio,
        },
        "concurrent_x_hbm_sweep": {
            f"{cs}x{cb}": _s(s) for (cs, cb), s in summaries.items()
        },
        "dram_multiplier_sweep": {
            str(m): _s(s) for m, s in dram_sweep.items()
        },
        "pcie_bandwidth_sweep": [
            {"name": n, "bandwidth_gb_s": bw, **_s(s)}
            for n, bw, s in pcie_sweep
        ],
        "hardware_comparison": [
            {
                "name": hc.name,
                "prefill_ms_per_token": hc.prefill_ms_per_token,
                "pcie_ms_per_token": hc.pcie_ms_per_token,
                "speedup_ratio": hc.speedup_ratio,
                **_s(s),
            }
            for hc, s in hw_comparison
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved results JSON to {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 74)
    print("  PCIe KV CACHE RELOADING — AVOIDING RECOMPUTATION DURING THRASHING")
    print("=" * 74)

    hw = DEFAULT_HW
    print_hardware_info(hw)

    print(f"\n  Workload (shared with experiment_thrashing.py):")
    print(f"    System prompt   : {SYSTEM_PROMPT_TOKENS} tokens (shared)")
    print(f"    Task prompt     : {TASK_PROMPT_TOKENS} tokens (per session)")
    print(f"    Per step        : ~{TOKENS_PER_STEP} tokens")
    print(f"    Steps/session   : {STEPS_PER_SESSION}")
    print(f"    Max context     : {MAX_SESSION_TOKENS} tokens "
          f"= {BLOCKS_PER_FULL_SESSION} blocks/session")
    print(f"    Total sessions  : {NUM_SESSIONS}")

    # ── 1. Detailed single-config analysis ──────────────────────────────────
    detail_concurrent = 8
    detail_hbm = 400
    detail_dram = detail_hbm * 10

    print(f"\n{'─' * 74}")
    print(f"  DETAILED ANALYSIS: {detail_concurrent} concurrent, "
          f"{detail_hbm} HBM blocks, {detail_dram} DRAM blocks")
    print(f"{'─' * 74}")

    traces_detail = simulate_pcie_reload(
        concurrent_sessions=detail_concurrent,
        hbm_cache_blocks=detail_hbm,
        dram_cache_blocks=detail_dram,
        hw=hw,
    )
    print_tier_summary(
        summarize(traces_detail),
        f"{detail_concurrent} concurrent, HBM={detail_hbm}, DRAM={detail_dram}",
    )
    plot_pcie_reload_detail(
        traces_detail, hw,
        label=f"{detail_concurrent} concurrent, HBM={detail_hbm}, "
              f"DRAM={detail_dram} (10×)",
        filename="pcie_reload_detail.png",
    )

    # ── 2. Sweep: concurrent × HBM size (DRAM = 10× HBM) ────────────────────
    print(f"\n{'─' * 74}")
    print("  SWEEP: concurrent sessions × HBM cache size "
          "(DRAM tier = 10× HBM)")
    print(f"{'─' * 74}")

    sweep_summaries: Dict[Tuple[int, int], TierSummary] = {}
    sweep_by_cs_at_400: Dict[int, List[PCIeReloadTrace]] = {}

    for cs in CONCURRENT_SESSIONS_SWEEP:
        for cb in CACHE_BLOCKS_SWEEP:
            dram = cb * 10
            print(f"  Simulating cs={cs}, HBM={cb}, DRAM={dram}...", end=" ")
            traces = simulate_pcie_reload(
                concurrent_sessions=cs,
                hbm_cache_blocks=cb,
                dram_cache_blocks=dram,
                hw=hw,
            )
            summary = summarize(traces)
            sweep_summaries[(cs, cb)] = summary
            print(f"savings={summary.savings_pct:5.1f}% "
                  f"(PCIe {summary.pcie_frac:5.1%} of tokens)")
            if cb == 400:
                sweep_by_cs_at_400[cs] = traces

    plot_savings_heatmap(sweep_summaries, hw)
    plot_concurrent_savings_overlay(sweep_by_cs_at_400, hw, hbm_blocks=400)

    # ── 3. DRAM tier size sweep (at a thrashing config) ─────────────────────
    print(f"\n{'─' * 74}")
    print(f"  DRAM TIER SIZE SWEEP: {detail_concurrent} concurrent, "
          f"HBM={detail_hbm} blocks")
    print(f"{'─' * 74}")

    dram_sweep_results: Dict[int, List[PCIeReloadTrace]] = {}
    dram_sweep_summaries: Dict[int, TierSummary] = {}
    for mult in DRAM_MULTIPLIERS:
        dram_blocks = detail_hbm * mult
        print(f"  Simulating DRAM = {mult}× HBM "
              f"({dram_blocks} blocks)...", end=" ")
        traces = simulate_pcie_reload(
            concurrent_sessions=detail_concurrent,
            hbm_cache_blocks=detail_hbm,
            dram_cache_blocks=dram_blocks,
            hw=hw,
        )
        summary = summarize(traces)
        dram_sweep_results[mult] = traces
        dram_sweep_summaries[mult] = summary
        print(f"savings={summary.savings_pct:5.1f}% "
              f"(PCIe {summary.pcie_frac:5.1%})")

    plot_dram_size_sweep(
        dram_sweep_results, hw,
        label=f"{detail_concurrent} concurrent, HBM={detail_hbm} blocks",
    )

    # ── 4. PCIe bandwidth sweep (including disk NVMe) ──────────────────────
    print(f"\n{'─' * 74}")
    print(f"  TIER-2 BANDWIDTH SWEEP: {detail_concurrent} concurrent, "
          f"HBM={detail_hbm}, DRAM={detail_dram}")
    print(f"{'─' * 74}")
    print("  (Lower bandwidths model NVMe disk as the backing tier; "
          "higher bandwidths model CXL / future fabrics.)")

    pcie_sweep_summaries: List[Tuple[str, float, TierSummary]] = []
    for name, bw in PCIE_BW_SWEEP:
        hw_variant = HardwareConfig(
            name=f"{hw.name} ({name})",
            kv_bytes_per_token=hw.kv_bytes_per_token,
            prefill_ms_per_token=hw.prefill_ms_per_token,
            pcie_bandwidth_gb_s=bw,
            bw_efficiency=hw.bw_efficiency,
        )
        traces = simulate_pcie_reload(
            concurrent_sessions=detail_concurrent,
            hbm_cache_blocks=detail_hbm,
            dram_cache_blocks=detail_dram,
            hw=hw_variant,
        )
        summary = summarize(traces)
        pcie_sweep_summaries.append((name, bw, summary))
        print(f"    {name:28s}: {hw_variant.pcie_ms_per_token:.4f} ms/tok "
              f"({hw_variant.speedup_ratio:5.1f}× vs recompute)  "
              f"→ savings={summary.savings_pct:5.1f}%")

    plot_pcie_bandwidth_sweep(pcie_sweep_summaries, hw)

    # ── 5. Hardware config comparison ───────────────────────────────────────
    print(f"\n{'─' * 74}")
    print("  HARDWARE CONFIG COMPARISON "
          f"({detail_concurrent} concurrent, HBM={detail_hbm}, DRAM={detail_dram})")
    print(f"{'─' * 74}")

    hw_comparison: List[Tuple[HardwareConfig, TierSummary]] = []
    for config in [HW_A100_7B, HW_H100_7B, HW_A100_70B]:
        traces = simulate_pcie_reload(
            concurrent_sessions=detail_concurrent,
            hbm_cache_blocks=detail_hbm,
            dram_cache_blocks=detail_dram,
            hw=config,
        )
        summary = summarize(traces)
        hw_comparison.append((config, summary))
        print(f"    {config.name:32s}: "
              f"prefill={config.prefill_ms_per_token:.3f} ms/tok, "
              f"PCIe={config.pcie_ms_per_token:.3f} ms/tok, "
              f"speedup={config.speedup_ratio:4.1f}×, "
              f"savings={summary.savings_pct:5.1f}%")

    plot_hardware_comparison(hw_comparison)

    # ── 6. Dump results JSON for the report ────────────────────────────────
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_pcie_kv_reload_results.json",
    )
    dump_results_json(
        json_path,
        hw=hw,
        summaries=sweep_summaries,
        dram_sweep=dram_sweep_summaries,
        pcie_sweep=pcie_sweep_summaries,
        hw_comparison=hw_comparison,
    )

    # ── 7. Final takeaways (on-screen) ──────────────────────────────────────
    print(f"\n{'═' * 74}")
    print("  TAKEAWAYS")
    print(f"{'═' * 74}")

    # Find the worst-thrashing configuration and show its savings.
    worst_key = max(
        sweep_summaries.keys(),
        key=lambda k: sweep_summaries[k].miss_frac * 0.0
                      + sweep_summaries[k].pcie_frac,  # max PCIe activity
    )
    best_savings_key = max(
        sweep_summaries.keys(),
        key=lambda k: sweep_summaries[k].savings_pct,
    )
    worst = sweep_summaries[worst_key]
    best = sweep_summaries[best_savings_key]
    print(f"  • Most tokens reloaded via PCIe: "
          f"{worst_key[0]} concurrent × {worst_key[1]} HBM blocks → "
          f"{worst.pcie_frac:.0%} of tokens, "
          f"saving {worst.savings_pct:.0f}% compute")
    print(f"  • Biggest compute savings       : "
          f"{best_savings_key[0]} concurrent × {best_savings_key[1]} HBM blocks → "
          f"{best.savings_pct:.0f}% "
          f"({best.savings_total_ms / 1000:.1f} s of prefill avoided)")
    print(f"  • PCIe reload is "
          f"{hw.speedup_ratio:.1f}× cheaper per token than recompute "
          f"on {hw.name}.")


def plot_concurrent_savings_overlay(
    results: Dict[int, List[PCIeReloadTrace]],
    hw: HardwareConfig,
    hbm_blocks: int,
    filename: str = "pcie_reload_concurrent_sweep.png",
):
    """Cumulative savings % over time for each concurrency level at a fixed
    HBM size (and DRAM = 10x HBM)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    concs = sorted(results.keys())
    colors = plt.cm.coolwarm([i / max(1, len(concs) - 1) for i in range(len(concs))])

    for cs, color in zip(concs, colors):
        traces = results[cs]
        if not traces:
            continue
        xs = [t.request_idx for t in traces]
        ax1.plot(xs, [t.cum_savings_pct for t in traces],
                 color=color, linewidth=1.8,
                 label=f"{cs} concurrent")
        ax2.plot(xs, [t.pcie_frac * 100 for t in traces],
                 color=color, linewidth=1.4,
                 label=f"{cs} concurrent", alpha=0.9)

    ax1.set_ylabel("Cumulative compute savings (%)")
    ax1.set_title(
        f"PCIe Reload Savings by Concurrency "
        f"(HBM={hbm_blocks} blocks, DRAM=10× HBM)",
        fontsize=11, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Tokens reloaded via PCIe (%)")
    ax2.set_xlabel("Request index")
    ax2.legend(loc="upper right", fontsize=9, ncol=2)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_hardware_comparison(
    hw_summaries: List[Tuple[HardwareConfig, TierSummary]],
    filename: str = "pcie_reload_hardware_comparison.png",
):
    """Compare savings % across different hardware configurations."""
    if not hw_summaries:
        return
    names = [hw.name for hw, _ in hw_summaries]
    savings = [s.savings_pct for _, s in hw_summaries]
    speedups = [hw.speedup_ratio for hw, _ in hw_summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    xs = list(range(len(names)))

    bars = ax1.bar(xs, savings, color="#4CAF50", alpha=0.85)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax1.set_ylabel("Compute savings (%)")
    ax1.set_title("Savings Across Hardware Configs",
                  fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_ylim(0, max(100, max(savings) + 10))
    for bar, v in zip(bars, savings):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.8,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    bars = ax2.bar(xs, speedups, color="#2196F3", alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Speedup ratio (recompute_ms / pcie_ms)")
    ax2.set_title("Cost Ratio: Recompute vs PCIe Reload",
                  fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.08,
                 f"{v:.1f}×", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


if __name__ == "__main__":
    main()
