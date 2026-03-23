"""Plot KV cache hit rate over time and characterize I/O patterns.

Generates per-request time-series data for each shared prefix fraction,
then produces plots showing:
  1. Cumulative cache hit rate over request index
  2. Per-request cache operations (hit/miss/insert/evict) over time
  3. Cache utilization and block inventory over time
  4. I/O characterization: read vs write volume per request
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
from vidur.entities.request import Request

# ── Parameters (same as experiment) ──────────────────────────────────────────
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
    # lookup result
    is_hit: bool
    tokens_cached: int       # tokens returned by match_prefix
    tokens_prefill: int      # original prefill
    # cumulative
    cum_hits: int
    cum_lookups: int
    cum_hit_rate: float
    cum_token_hits: int
    cum_token_lookups: int
    cum_token_hit_rate: float
    # cache state after insert
    cached_blocks: int
    utilization: float
    # operations this request triggered
    blocks_inserted: int
    blocks_evicted: int
    evictions_this_req: int


def simulate(shared_fraction: float) -> List[RequestTrace]:
    """Run simulation and return per-request trace."""
    requests = [
        Request(arrived_at=float(i) * 0.1,
                num_prefill_tokens=PREFILL_TOKENS,
                num_decode_tokens=DECODE_TOKENS)
        for i in range(NUM_REQUESTS)
    ]

    config = PrefixCacheConfig(
        enabled=True,
        num_shared_prefixes=NUM_PREFIX_GROUPS,
        shared_prefix_length_fraction=shared_fraction,
        seed=SEED,
    )
    gen = PrefixTokenGenerator(config)
    gen.assign_token_ids(requests)

    cache = PrefixCacheManager(max_blocks=CACHE_MAX_BLOCKS, block_size=BLOCK_SIZE)
    traces = []

    for i, req in enumerate(requests):
        # snapshot before
        evictions_before = cache.stats.total_evictions
        blocks_evicted_before = cache.stats.total_blocks_evicted
        blocks_inserted_before = cache.stats.total_blocks_inserted

        # lookup
        cached_tokens = cache.match_prefix(req.token_ids)
        is_hit = cached_tokens > 0
        if is_hit:
            req.apply_prefix_cache_hit(cached_tokens)

        # insert (on completion)
        cache.on_request_complete(req.token_ids)

        # snapshot after
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


def plot_all(all_traces: dict):
    """Generate all plots."""
    fracs = list(all_traces.keys())
    colors = plt.cm.viridis([i / (len(fracs) - 1) if len(fracs) > 1 else 0
                             for i in range(len(fracs))])

    # ─────────────────────────────────────────────────────────────────────
    # Figure 1: Cumulative hit rate over time (request-level + token-level)
    # ─────────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for frac, color in zip(fracs, colors):
        traces = all_traces[frac]
        xs = [t.request_idx for t in traces]
        ax1.plot(xs, [t.cum_hit_rate * 100 for t in traces],
                 color=color, label=f"{frac:.0%} shared", linewidth=1.5)
        ax2.plot(xs, [t.cum_token_hit_rate * 100 for t in traces],
                 color=color, label=f"{frac:.0%} shared", linewidth=1.5)

    ax1.set_ylabel("Cumulative Request Hit Rate (%)")
    ax1.set_title("Cache Hit Rate Over Time")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Cumulative Token Hit Rate (%)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_hit_rate_over_time.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_hit_rate_over_time.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 2: Per-request I/O pattern (blocks inserted vs evicted)
    #           for select fractions
    # ─────────────────────────────────────────────────────────────────────
    select_fracs = [0.0, 0.3, 0.7, 0.9]
    select_fracs = [f for f in select_fracs if f in all_traces]
    fig, axes = plt.subplots(len(select_fracs), 1, figsize=(12, 3 * len(select_fracs)),
                             sharex=True)
    if len(select_fracs) == 1:
        axes = [axes]

    for ax, frac in zip(axes, select_fracs):
        traces = all_traces[frac]
        xs = [t.request_idx for t in traces]
        ins = [t.blocks_inserted for t in traces]
        evi = [-t.blocks_evicted for t in traces]  # negative for visual contrast

        ax.bar(xs, ins, width=1.0, color="#2196F3", alpha=0.7, label="Blocks inserted")
        ax.bar(xs, evi, width=1.0, color="#F44336", alpha=0.7, label="Blocks evicted")

        # Mark hits vs misses on top
        hit_xs = [t.request_idx for t in traces if t.is_hit]
        miss_xs = [t.request_idx for t in traces if not t.is_hit]
        ymax = max(ins) + 2 if ins else 35
        ax.scatter(miss_xs, [ymax] * len(miss_xs), marker='x', color='red',
                   s=15, zorder=5, label="Miss")
        if len(hit_xs) <= 40:  # don't clutter if too many
            ax.scatter(hit_xs, [ymax] * len(hit_xs), marker='.', color='green',
                       s=8, zorder=5, label="Hit")

        ax.set_ylabel("Blocks")
        ax.set_title(f"I/O Pattern — {frac:.0%} Shared Prefix", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='black', linewidth=0.5)

    axes[-1].set_xlabel("Request Index")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_io_pattern.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_io_pattern.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 3: Cache utilization (blocks cached) over time
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    for frac, color in zip(fracs, colors):
        traces = all_traces[frac]
        xs = [t.request_idx for t in traces]
        ax.plot(xs, [t.cached_blocks for t in traces],
                color=color, label=f"{frac:.0%} shared", linewidth=1.5)

    ax.axhline(CACHE_MAX_BLOCKS, color='red', linestyle='--', linewidth=1,
               label=f"Capacity ({CACHE_MAX_BLOCKS} blocks)")
    ax.set_ylabel("Cached Blocks")
    ax.set_xlabel("Request Index")
    ax.set_title("Cache Block Utilization Over Time")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_cache_utilization.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_cache_utilization.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 4: Read/Write volume characterization
    #   "Read"  = tokens served from cache (cache hit tokens)
    #   "Write" = blocks inserted × block_size (tokens written to cache)
    #   "Evict" = blocks evicted × block_size (tokens discarded)
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(select_fracs), 1, figsize=(12, 3 * len(select_fracs)),
                             sharex=True)
    if len(select_fracs) == 1:
        axes = [axes]

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
        ax.set_title(f"Read / Write / Evict Volume — {frac:.0%} Shared", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Request Index")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_rw_volume.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_rw_volume.png")

    # ─────────────────────────────────────────────────────────────────────
    # Figure 5: Instantaneous (windowed) hit rate — sliding window of 10
    # ─────────────────────────────────────────────────────────────────────
    WINDOW = 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for frac, color in zip(fracs, colors):
        traces = all_traces[frac]
        # Windowed request hit rate
        windowed_req = []
        windowed_tok = []
        for i in range(len(traces)):
            start = max(0, i - WINDOW + 1)
            window = traces[start:i + 1]
            req_hr = sum(1 for t in window if t.is_hit) / len(window) * 100
            tok_hr = (sum(t.tokens_cached for t in window) /
                      sum(t.tokens_prefill for t in window) * 100)
            windowed_req.append(req_hr)
            windowed_tok.append(tok_hr)

        xs = [t.request_idx for t in traces]
        ax1.plot(xs, windowed_req, color=color, label=f"{frac:.0%} shared",
                 linewidth=1.2, alpha=0.8)
        ax2.plot(xs, windowed_tok, color=color, label=f"{frac:.0%} shared",
                 linewidth=1.2, alpha=0.8)

    ax1.set_ylabel(f"Request Hit Rate (%, {WINDOW}-req window)")
    ax1.set_title(f"Instantaneous Cache Hit Rate (sliding window = {WINDOW})")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel(f"Token Hit Rate (%, {WINDOW}-req window)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "plot_windowed_hit_rate.png"), dpi=150)
    plt.close(fig)
    print("  Saved plot_windowed_hit_rate.png")


def print_io_characterization(all_traces: dict):
    """Print I/O pattern characterization."""
    print("\n" + "=" * 70)
    print("  KV CACHE I/O CHARACTERIZATION")
    print("=" * 70)

    for frac in sorted(all_traces.keys()):
        traces = all_traces[frac]

        total_read_tokens = sum(t.tokens_cached for t in traces)
        total_write_tokens = sum(t.blocks_inserted * BLOCK_SIZE for t in traces)
        total_evict_tokens = sum(t.blocks_evicted * BLOCK_SIZE for t in traces)
        total_prefill = NUM_REQUESTS * PREFILL_TOKENS

        # Phases
        warmup_end = 0
        for t in traces:
            if t.cum_hit_rate < 0.90 and t.request_idx >= NUM_PREFIX_GROUPS:
                warmup_end = t.request_idx
            elif t.request_idx >= NUM_PREFIX_GROUPS:
                break
        if frac == 0.0:
            warmup_end = NUM_REQUESTS  # never warms up

        steady_traces = [t for t in traces if t.request_idx > warmup_end]
        if steady_traces:
            steady_hit_rate = (sum(1 for t in steady_traces if t.is_hit) /
                               len(steady_traces))
            steady_token_rate = (sum(t.tokens_cached for t in steady_traces) /
                                 sum(t.tokens_prefill for t in steady_traces))
        else:
            steady_hit_rate = 0.0
            steady_token_rate = 0.0

        # Read/write ratio
        rw_ratio = total_read_tokens / total_write_tokens if total_write_tokens > 0 else 0.0

        # Eviction frequency
        reqs_with_eviction = sum(1 for t in traces if t.evictions_this_req > 0)

        # Average blocks per operation
        inserting_reqs = [t for t in traces if t.blocks_inserted > 0]
        avg_insert = (sum(t.blocks_inserted for t in inserting_reqs) /
                      len(inserting_reqs)) if inserting_reqs else 0
        evicting_reqs = [t for t in traces if t.blocks_evicted > 0]
        avg_evict = (sum(t.blocks_evicted for t in evicting_reqs) /
                     len(evicting_reqs)) if evicting_reqs else 0

        print(f"\n  ── {frac:.0%} Shared Prefix ──")
        print(f"  Read volume:     {total_read_tokens:>8,} tokens  "
              f"({total_read_tokens / total_prefill:.1%} of total prefill)")
        print(f"  Write volume:    {total_write_tokens:>8,} tokens  "
              f"({total_write_tokens / total_prefill:.1%} of total prefill)")
        print(f"  Evict volume:    {total_evict_tokens:>8,} tokens  "
              f"({total_evict_tokens / total_prefill:.1%} of total prefill)")
        print(f"  Read/Write:      {rw_ratio:>8.2f}x")
        print(f"  Warmup phase:    requests 0–{warmup_end}  "
              f"({warmup_end + 1} requests)")
        print(f"  Steady-state:    request hit rate = {steady_hit_rate:.1%}, "
              f"token hit rate = {steady_token_rate:.1%}")
        print(f"  Eviction freq:   {reqs_with_eviction}/{NUM_REQUESTS} requests "
              f"trigger eviction ({reqs_with_eviction / NUM_REQUESTS:.0%})")
        print(f"  Avg insert size: {avg_insert:.1f} blocks/request")
        print(f"  Avg evict size:  {avg_evict:.1f} blocks/request")

        # Classify I/O pattern
        if frac == 0.0:
            pattern = "WRITE-THROUGH: every request writes full sequence, " \
                      "nothing is re-read. Pure churn."
        elif frac <= 0.2:
            pattern = "WRITE-HEAVY: most blocks are unique suffixes that get " \
                      "written and evicted. Small read benefit from short prefixes."
        elif frac <= 0.5:
            pattern = "BALANCED: significant read hits on shared prefixes, " \
                      "moderate write churn on unique suffixes."
        elif frac <= 0.7:
            pattern = "READ-HEAVY: majority of tokens served from cache. " \
                      "Writes are small (short unique suffixes). Low eviction."
        else:
            pattern = "READ-DOMINATED: cache acts as a read-mostly store. " \
                      "Minimal writes, near-zero eviction after warmup."
        print(f"  I/O pattern:     {pattern}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("=" * 70)
    print("  KV CACHE HIT RATE & I/O PATTERN ANALYSIS")
    print("=" * 70)

    all_traces = {}
    for frac in SHARED_FRACTIONS:
        print(f"  Simulating {frac:.0%} shared prefix...")
        all_traces[frac] = simulate(frac)

    print("\n  Generating plots...")
    plot_all(all_traces)
    print_io_characterization(all_traces)


if __name__ == "__main__":
    main()
