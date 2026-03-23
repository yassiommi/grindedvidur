"""Plot KV cache hit rate over time and characterize I/O patterns.

Runs two simulation modes side by side:
  1. Synthetic: uniform prefix groups (PrefixTokenGenerator)
  2. Trace-based: ShareGPT-style conversations (ShareGPTTokenIdProvider)

Generates plots showing how trace-based token IDs produce realistic
fluctuations in cache hit rate compared to the idealized synthetic case.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

# Allow running from experiments/ or repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
PLOT_DIR = os.path.join(REPO_ROOT, "report_figures", "kv_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from vidur.config.config import PrefixCacheConfig
from vidur.entities.prefix_cache_manager import PrefixCacheManager
from vidur.entities.prefix_token_generator import PrefixTokenGenerator
from vidur.entities.trace_token_provider import ShareGPTTokenIdProvider
from vidur.entities.request import Request

# ── Parameters ───────────────────────────────────────────────────────────────
NUM_REQUESTS = 200
PREFILL_TOKENS = 512
DECODE_TOKENS = 128
BLOCK_SIZE = 16
NUM_PREFIX_GROUPS = 5
CACHE_MAX_BLOCKS = 400
SEED = 42
SHARED_FRACTIONS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

BLOCKS_PER_REQUEST = PREFILL_TOKENS // BLOCK_SIZE  # 32


@dataclass
class RequestTrace:
    """Per-request snapshot of cache state."""
    request_idx: int
    arrived_at: float
    is_hit: bool
    tokens_cached: int
    tokens_prefill: int
    cum_hits: int
    cum_lookups: int
    cum_hit_rate: float
    cum_token_hits: int
    cum_token_lookups: int
    cum_token_hit_rate: float
    cached_blocks: int
    utilization: float
    blocks_inserted: int
    blocks_evicted: int
    evictions_this_req: int


def _run_cache_simulation(requests: List[Request]) -> List[RequestTrace]:
    """Run cache simulation on requests that already have token_ids assigned."""
    cache = PrefixCacheManager(max_blocks=CACHE_MAX_BLOCKS, block_size=BLOCK_SIZE)
    traces = []

    for i, req in enumerate(requests):
        evictions_before = cache.stats.total_evictions
        blocks_evicted_before = cache.stats.total_blocks_evicted
        blocks_inserted_before = cache.stats.total_blocks_inserted

        cached_tokens = cache.match_prefix(req.token_ids)
        is_hit = cached_tokens > 0
        if is_hit:
            req.apply_prefix_cache_hit(cached_tokens)

        cache.on_request_complete(req.token_ids)

        stats = cache.stats
        traces.append(RequestTrace(
            request_idx=i,
            arrived_at=req.arrived_at,
            is_hit=is_hit,
            tokens_cached=cached_tokens,
            tokens_prefill=PREFILL_TOKENS,
            cum_hits=stats.total_hits,
            cum_lookups=stats.total_lookups,
            cum_hit_rate=stats.hit_rate,
            cum_token_hits=stats.total_tokens_hit,
            cum_token_lookups=stats.total_tokens_looked_up,
            cum_token_hit_rate=stats.token_hit_rate,
            cached_blocks=cache.num_cached_blocks,
            utilization=cache.cache_utilization,
            blocks_inserted=stats.total_blocks_inserted - blocks_inserted_before,
            blocks_evicted=stats.total_blocks_evicted - blocks_evicted_before,
            evictions_this_req=stats.total_evictions - evictions_before,
        ))

    return traces


def _make_requests() -> List[Request]:
    return [
        Request(arrived_at=float(i) * 0.1,
                num_prefill_tokens=PREFILL_TOKENS,
                num_decode_tokens=DECODE_TOKENS)
        for i in range(NUM_REQUESTS)
    ]


def simulate_synthetic(shared_fraction: float) -> List[RequestTrace]:
    """Run simulation with synthetic prefix groups."""
    requests = _make_requests()
    config = PrefixCacheConfig(
        enabled=True,
        num_shared_prefixes=NUM_PREFIX_GROUPS,
        shared_prefix_length_fraction=shared_fraction,
        seed=SEED,
    )
    gen = PrefixTokenGenerator(config)
    gen.assign_token_ids(requests)
    return _run_cache_simulation(requests)


def simulate_trace(
    multi_turn: bool = True,
    interleave: bool = True,
    max_active: int = 5,
) -> List[RequestTrace]:
    """Run simulation with ShareGPT-style trace token IDs."""
    requests = _make_requests()
    provider = ShareGPTTokenIdProvider(
        seed=SEED,
        multi_turn=multi_turn,
        interleave_conversations=interleave,
        max_active_conversations=max_active,
    )
    provider.assign_token_ids(requests)
    return _run_cache_simulation(requests)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_synthetic(all_traces: dict):
    """Generate plots for synthetic simulation (original plots)."""
    fracs = list(all_traces.keys())
    colors = plt.cm.viridis([i / (len(fracs) - 1) if len(fracs) > 1 else 0
                             for i in range(len(fracs))])

    # Figure 1: Cumulative hit rate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for frac, color in zip(fracs, colors):
        traces = all_traces[frac]
        xs = [t.request_idx for t in traces]
        ax1.plot(xs, [t.cum_hit_rate * 100 for t in traces],
                 color=color, label=f"{frac:.0%} shared", linewidth=1.5)
        ax2.plot(xs, [t.cum_token_hit_rate * 100 for t in traces],
                 color=color, label=f"{frac:.0%} shared", linewidth=1.5)
    ax1.set_ylabel("Cumulative Request Hit Rate (%)")
    ax1.set_title("Cache Hit Rate Over Time (Synthetic)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(-2, 102); ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Cumulative Token Hit Rate (%)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(-2, 102); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_hit_rate_over_time.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_hit_rate_over_time.png")

    # Figure 2: I/O pattern
    select_fracs = [f for f in [0.0, 0.3, 0.7, 0.9] if f in all_traces]
    fig, axes = plt.subplots(len(select_fracs), 1,
                             figsize=(12, 3 * len(select_fracs)), sharex=True)
    if len(select_fracs) == 1: axes = [axes]
    for ax, frac in zip(axes, select_fracs):
        traces = all_traces[frac]
        xs = [t.request_idx for t in traces]
        ins = [t.blocks_inserted for t in traces]
        evi = [-t.blocks_evicted for t in traces]
        ax.bar(xs, ins, width=1.0, color="#2196F3", alpha=0.7, label="Blocks inserted")
        ax.bar(xs, evi, width=1.0, color="#F44336", alpha=0.7, label="Blocks evicted")
        miss_xs = [t.request_idx for t in traces if not t.is_hit]
        ymax = max(ins) + 2 if ins else 35
        ax.scatter(miss_xs, [ymax] * len(miss_xs), marker='x', color='red',
                   s=15, zorder=5, label="Miss")
        ax.set_ylabel("Blocks")
        ax.set_title(f"I/O Pattern — {frac:.0%} Shared Prefix (Synthetic)", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2); ax.axhline(0, color='black', linewidth=0.5)
    axes[-1].set_xlabel("Request Index")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_io_pattern.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_io_pattern.png")

    # Figure 3: Cache utilization
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for frac, color in zip(fracs, colors):
        traces = all_traces[frac]
        xs = [t.request_idx for t in traces]
        ax.plot(xs, [t.cached_blocks for t in traces],
                color=color, label=f"{frac:.0%} shared", linewidth=1.5)
    ax.axhline(CACHE_MAX_BLOCKS, color='red', linestyle='--', linewidth=1,
               label=f"Capacity ({CACHE_MAX_BLOCKS} blocks)")
    ax.set_ylabel("Cached Blocks"); ax.set_xlabel("Request Index")
    ax.set_title("Cache Block Utilization Over Time (Synthetic)")
    ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_cache_utilization.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_cache_utilization.png")

    # Figure 4: R/W volume
    fig, axes = plt.subplots(len(select_fracs), 1,
                             figsize=(12, 3 * len(select_fracs)), sharex=True)
    if len(select_fracs) == 1: axes = [axes]
    for ax, frac in zip(axes, select_fracs):
        traces = all_traces[frac]
        xs = [t.request_idx for t in traces]
        reads = [t.tokens_cached for t in traces]
        writes = [t.blocks_inserted * BLOCK_SIZE for t in traces]
        evicts = [t.blocks_evicted * BLOCK_SIZE for t in traces]
        ax.fill_between(xs, reads, alpha=0.4, color="#4CAF50", label="Read (cache hit tokens)")
        ax.fill_between(xs, writes, alpha=0.4, color="#2196F3", label="Write (tokens inserted)")
        ax.fill_between(xs, evicts, alpha=0.4, color="#F44336", label="Evict (tokens discarded)")
        ax.plot(xs, reads, color="#4CAF50", linewidth=1)
        ax.plot(xs, writes, color="#2196F3", linewidth=1)
        ax.plot(xs, evicts, color="#F44336", linewidth=1)
        ax.set_ylabel("Tokens")
        ax.set_title(f"Read / Write / Evict Volume — {frac:.0%} Shared (Synthetic)", fontsize=10)
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Request Index")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_rw_volume.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_rw_volume.png")

    # Figure 5: Windowed hit rate
    WINDOW = 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for frac, color in zip(fracs, colors):
        traces = all_traces[frac]
        windowed_req, windowed_tok = [], []
        for i in range(len(traces)):
            start = max(0, i - WINDOW + 1)
            window = traces[start:i + 1]
            windowed_req.append(sum(1 for t in window if t.is_hit) / len(window) * 100)
            windowed_tok.append(sum(t.tokens_cached for t in window) /
                                sum(t.tokens_prefill for t in window) * 100)
        xs = [t.request_idx for t in traces]
        ax1.plot(xs, windowed_req, color=color, label=f"{frac:.0%} shared",
                 linewidth=1.2, alpha=0.8)
        ax2.plot(xs, windowed_tok, color=color, label=f"{frac:.0%} shared",
                 linewidth=1.2, alpha=0.8)
    ax1.set_ylabel(f"Request Hit Rate (%, {WINDOW}-req window)")
    ax1.set_title(f"Instantaneous Cache Hit Rate (Synthetic, window={WINDOW})")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(-2, 102); ax1.grid(True, alpha=0.3)
    ax2.set_ylabel(f"Token Hit Rate (%, {WINDOW}-req window)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(-2, 102); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_windowed_hit_rate.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_windowed_hit_rate.png")


def plot_trace(trace_results: dict):
    """Generate plots for trace-based simulation."""
    configs = list(trace_results.keys())
    colors = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800"]

    # ── Figure 6: Cumulative hit rate comparison across trace configs ────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for (label, traces), color in zip(trace_results.items(), colors):
        xs = [t.request_idx for t in traces]
        ax1.plot(xs, [t.cum_hit_rate * 100 for t in traces],
                 color=color, label=label, linewidth=1.5)
        ax2.plot(xs, [t.cum_token_hit_rate * 100 for t in traces],
                 color=color, label=label, linewidth=1.5)
    ax1.set_ylabel("Cumulative Request Hit Rate (%)")
    ax1.set_title("Cache Hit Rate Over Time (ShareGPT Trace)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(-2, 102); ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("Cumulative Token Hit Rate (%)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(-2, 102); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "trace_hit_rate_over_time.png"), dpi=150)
    plt.close(fig)
    print("  Saved trace_hit_rate_over_time.png")

    # ── Figure 7: Windowed hit rate (trace) ──────────────────────────────
    WINDOW = 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for (label, traces), color in zip(trace_results.items(), colors):
        windowed_req, windowed_tok = [], []
        for i in range(len(traces)):
            start = max(0, i - WINDOW + 1)
            window = traces[start:i + 1]
            windowed_req.append(sum(1 for t in window if t.is_hit) / len(window) * 100)
            windowed_tok.append(sum(t.tokens_cached for t in window) /
                                sum(t.tokens_prefill for t in window) * 100)
        xs = [t.request_idx for t in traces]
        ax1.plot(xs, windowed_req, color=color, label=label,
                 linewidth=1.2, alpha=0.8)
        ax2.plot(xs, windowed_tok, color=color, label=label,
                 linewidth=1.2, alpha=0.8)
    ax1.set_ylabel(f"Request Hit Rate (%, {WINDOW}-req window)")
    ax1.set_title(f"Instantaneous Cache Hit Rate (ShareGPT, window={WINDOW})")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(-2, 102); ax1.grid(True, alpha=0.3)
    ax2.set_ylabel(f"Token Hit Rate (%, {WINDOW}-req window)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(-2, 102); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "trace_windowed_hit_rate.png"), dpi=150)
    plt.close(fig)
    print("  Saved trace_windowed_hit_rate.png")

    # ── Figure 8: I/O pattern for each trace config ──────────────────────
    fig, axes = plt.subplots(len(configs), 1,
                             figsize=(12, 3 * len(configs)), sharex=True)
    if len(configs) == 1: axes = [axes]
    for ax, ((label, traces), color) in zip(axes, zip(trace_results.items(), colors)):
        xs = [t.request_idx for t in traces]
        ins = [t.blocks_inserted for t in traces]
        evi = [-t.blocks_evicted for t in traces]
        ax.bar(xs, ins, width=1.0, color="#2196F3", alpha=0.7, label="Blocks inserted")
        ax.bar(xs, evi, width=1.0, color="#F44336", alpha=0.7, label="Blocks evicted")
        hit_xs = [t.request_idx for t in traces if t.is_hit]
        miss_xs = [t.request_idx for t in traces if not t.is_hit]
        ymax = max(ins) + 2 if ins else 35
        ax.scatter(miss_xs, [ymax] * len(miss_xs), marker='x', color='red',
                   s=15, zorder=5, label="Miss")
        if len(hit_xs) <= 60:
            ax.scatter(hit_xs, [ymax] * len(hit_xs), marker='.', color='green',
                       s=8, zorder=5, label="Hit")
        ax.set_ylabel("Blocks")
        ax.set_title(f"I/O Pattern — {label}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2); ax.axhline(0, color='black', linewidth=0.5)
    axes[-1].set_xlabel("Request Index")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "trace_io_pattern.png"), dpi=150)
    plt.close(fig)
    print("  Saved trace_io_pattern.png")

    # ── Figure 9: Cache utilization (trace) ──────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    for (label, traces), color in zip(trace_results.items(), colors):
        xs = [t.request_idx for t in traces]
        ax.plot(xs, [t.cached_blocks for t in traces],
                color=color, label=label, linewidth=1.5)
    ax.axhline(CACHE_MAX_BLOCKS, color='red', linestyle='--', linewidth=1,
               label=f"Capacity ({CACHE_MAX_BLOCKS} blocks)")
    ax.set_ylabel("Cached Blocks"); ax.set_xlabel("Request Index")
    ax.set_title("Cache Block Utilization Over Time (ShareGPT Trace)")
    ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "trace_cache_utilization.png"), dpi=150)
    plt.close(fig)
    print("  Saved trace_cache_utilization.png")

    # ── Figure 10: R/W volume (trace) ────────────────────────────────────
    fig, axes = plt.subplots(len(configs), 1,
                             figsize=(12, 3 * len(configs)), sharex=True)
    if len(configs) == 1: axes = [axes]
    for ax, (label, traces) in zip(axes, trace_results.items()):
        xs = [t.request_idx for t in traces]
        reads = [t.tokens_cached for t in traces]
        writes = [t.blocks_inserted * BLOCK_SIZE for t in traces]
        evicts = [t.blocks_evicted * BLOCK_SIZE for t in traces]
        ax.fill_between(xs, reads, alpha=0.4, color="#4CAF50", label="Read (hit tokens)")
        ax.fill_between(xs, writes, alpha=0.4, color="#2196F3", label="Write (inserted)")
        ax.fill_between(xs, evicts, alpha=0.4, color="#F44336", label="Evict (discarded)")
        ax.plot(xs, reads, color="#4CAF50", linewidth=1)
        ax.plot(xs, writes, color="#2196F3", linewidth=1)
        ax.plot(xs, evicts, color="#F44336", linewidth=1)
        ax.set_ylabel("Tokens")
        ax.set_title(f"R/W/Evict Volume — {label}", fontsize=10)
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Request Index")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "trace_rw_volume.png"), dpi=150)
    plt.close(fig)
    print("  Saved trace_rw_volume.png")


def plot_comparison(synthetic_traces: dict, trace_traces: dict):
    """Side-by-side comparison: synthetic 50% vs trace interleaved."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Pick a representative synthetic case (50% shared)
    synth_key = 0.5
    synth = synthetic_traces.get(synth_key, list(synthetic_traces.values())[3])

    # Pick interleaved trace
    trace_key = list(trace_traces.keys())[0]
    trace = trace_traces[trace_key]

    WINDOW = 10

    # Top-left: cumulative hit rate
    ax = axes[0, 0]
    xs = [t.request_idx for t in synth]
    ax.plot(xs, [t.cum_hit_rate * 100 for t in synth],
            color="#2196F3", label=f"Synthetic ({synth_key:.0%})", linewidth=1.5)
    xs2 = [t.request_idx for t in trace]
    ax.plot(xs2, [t.cum_hit_rate * 100 for t in trace],
            color="#E91E63", label=f"Trace ({trace_key})", linewidth=1.5)
    ax.set_ylabel("Cumulative Request Hit Rate (%)")
    ax.set_title("Request Hit Rate")
    ax.legend(fontsize=8); ax.set_ylim(-2, 102); ax.grid(True, alpha=0.3)

    # Top-right: cumulative token hit rate
    ax = axes[0, 1]
    ax.plot(xs, [t.cum_token_hit_rate * 100 for t in synth],
            color="#2196F3", label=f"Synthetic ({synth_key:.0%})", linewidth=1.5)
    ax.plot(xs2, [t.cum_token_hit_rate * 100 for t in trace],
            color="#E91E63", label=f"Trace ({trace_key})", linewidth=1.5)
    ax.set_ylabel("Cumulative Token Hit Rate (%)")
    ax.set_title("Token Hit Rate")
    ax.legend(fontsize=8); ax.set_ylim(-2, 102); ax.grid(True, alpha=0.3)

    # Bottom-left: windowed request hit rate
    ax = axes[1, 0]
    for data, color, label in [(synth, "#2196F3", f"Synthetic ({synth_key:.0%})"),
                                (trace, "#E91E63", f"Trace ({trace_key})")]:
        windowed = []
        for i in range(len(data)):
            start = max(0, i - WINDOW + 1)
            w = data[start:i + 1]
            windowed.append(sum(1 for t in w if t.is_hit) / len(w) * 100)
        ax.plot([t.request_idx for t in data], windowed,
                color=color, label=label, linewidth=1.2, alpha=0.8)
    ax.set_ylabel(f"Request Hit Rate (%, {WINDOW}-req window)")
    ax.set_xlabel("Request Index")
    ax.set_title(f"Windowed Request Hit Rate (window={WINDOW})")
    ax.legend(fontsize=8); ax.set_ylim(-2, 102); ax.grid(True, alpha=0.3)

    # Bottom-right: windowed token hit rate
    ax = axes[1, 1]
    for data, color, label in [(synth, "#2196F3", f"Synthetic ({synth_key:.0%})"),
                                (trace, "#E91E63", f"Trace ({trace_key})")]:
        windowed = []
        for i in range(len(data)):
            start = max(0, i - WINDOW + 1)
            w = data[start:i + 1]
            windowed.append(sum(t.tokens_cached for t in w) /
                            sum(t.tokens_prefill for t in w) * 100)
        ax.plot([t.request_idx for t in data], windowed,
                color=color, label=label, linewidth=1.2, alpha=0.8)
    ax.set_ylabel(f"Token Hit Rate (%, {WINDOW}-req window)")
    ax.set_xlabel("Request Index")
    ax.set_title(f"Windowed Token Hit Rate (window={WINDOW})")
    ax.legend(fontsize=8); ax.set_ylim(-2, 102); ax.grid(True, alpha=0.3)

    fig.suptitle("Synthetic vs ShareGPT Trace: Cache Hit Rate Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "comparison_synthetic_vs_trace.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved comparison_synthetic_vs_trace.png")


# ── I/O characterization ────────────────────────────────────────────────────

def print_io_characterization(label: str, traces: List[RequestTrace]):
    """Print I/O characterization for a single trace."""
    total_read = sum(t.tokens_cached for t in traces)
    total_write = sum(t.blocks_inserted * BLOCK_SIZE for t in traces)
    total_evict = sum(t.blocks_evicted * BLOCK_SIZE for t in traces)
    total_prefill = len(traces) * PREFILL_TOKENS

    rw_ratio = total_read / total_write if total_write > 0 else 0.0
    reqs_with_eviction = sum(1 for t in traces if t.evictions_this_req > 0)

    inserting = [t for t in traces if t.blocks_inserted > 0]
    avg_insert = sum(t.blocks_inserted for t in inserting) / len(inserting) if inserting else 0
    evicting = [t for t in traces if t.blocks_evicted > 0]
    avg_evict = sum(t.blocks_evicted for t in evicting) / len(evicting) if evicting else 0

    # Variance in per-request token hit rate (measure of fluctuation)
    per_req_rates = [t.tokens_cached / t.tokens_prefill for t in traces]
    mean_rate = sum(per_req_rates) / len(per_req_rates)
    variance = sum((r - mean_rate) ** 2 for r in per_req_rates) / len(per_req_rates)
    std_rate = variance ** 0.5

    print(f"\n  ── {label} ──")
    print(f"  Read volume:     {total_read:>8,} tokens  ({total_read / total_prefill:.1%} of total)")
    print(f"  Write volume:    {total_write:>8,} tokens  ({total_write / total_prefill:.1%} of total)")
    print(f"  Evict volume:    {total_evict:>8,} tokens  ({total_evict / total_prefill:.1%} of total)")
    print(f"  Read/Write:      {rw_ratio:>8.2f}x")
    print(f"  Eviction freq:   {reqs_with_eviction}/{len(traces)} requests "
          f"({reqs_with_eviction / len(traces):.0%})")
    print(f"  Avg insert:      {avg_insert:.1f} blocks/req")
    print(f"  Avg evict:       {avg_evict:.1f} blocks/req")
    print(f"  Token hit rate:  {mean_rate:.1%} mean, {std_rate:.1%} std "
          f"(fluctuation: {'HIGH' if std_rate > 0.15 else 'MODERATE' if std_rate > 0.05 else 'LOW'})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── Part 1: Synthetic simulation (original) ──────────────────────────
    print("=" * 70)
    print("  PART 1: SYNTHETIC PREFIX SIMULATION")
    print("=" * 70)

    synthetic_traces = {}
    for frac in SHARED_FRACTIONS:
        print(f"  Simulating synthetic {frac:.0%} shared prefix...")
        synthetic_traces[frac] = simulate_synthetic(frac)

    print("\n  Generating synthetic plots...")
    plot_synthetic(synthetic_traces)

    print("\n" + "=" * 70)
    print("  SYNTHETIC I/O CHARACTERIZATION")
    print("=" * 70)
    for frac in SHARED_FRACTIONS:
        print_io_characterization(f"Synthetic {frac:.0%}", synthetic_traces[frac])

    # ── Part 2: Trace-based simulation ───────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  PART 2: SHAREGPT TRACE SIMULATION")
    print("=" * 70)

    trace_results = {}

    print("  Simulating: interleaved multi-turn (5 active)...")
    trace_results["Interleaved (5 active)"] = simulate_trace(
        multi_turn=True, interleave=True, max_active=5)

    print("  Simulating: interleaved multi-turn (10 active)...")
    trace_results["Interleaved (10 active)"] = simulate_trace(
        multi_turn=True, interleave=True, max_active=10)

    print("  Simulating: sequential multi-turn...")
    trace_results["Sequential multi-turn"] = simulate_trace(
        multi_turn=True, interleave=False)

    print("  Simulating: independent (no multi-turn)...")
    trace_results["Independent (single-turn)"] = simulate_trace(
        multi_turn=False)

    print("\n  Generating trace plots...")
    plot_trace(trace_results)

    print("\n" + "=" * 70)
    print("  SHAREGPT TRACE I/O CHARACTERIZATION")
    print("=" * 70)
    for label, traces in trace_results.items():
        print_io_characterization(label, traces)

    # ── Part 3: Side-by-side comparison ──────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  PART 3: SYNTHETIC vs TRACE COMPARISON")
    print("=" * 70)
    plot_comparison(synthetic_traces, trace_results)


if __name__ == "__main__":
    main()
