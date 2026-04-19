"""End-to-end KV cache prefix simulation experiment.

Generates synthetic workloads with varying shared prefix fractions and measures
cache hit rates, token savings, eviction behavior, and utilization across
different configurations.

This is a pure simulation — no GPU profiling or execution time prediction.
"""

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from vidur.config.config import PrefixCacheConfig
from vidur.entities.prefix_cache_manager import PrefixCacheManager
from vidur.entities.prefix_token_generator import PrefixTokenGenerator
from vidur.entities.request import Request


# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

NUM_REQUESTS = 200
PREFILL_TOKENS = 512        # tokens per request
DECODE_TOKENS = 128         # (not used by cache, but realistic)
BLOCK_SIZE = 16             # tokens per KV block
NUM_PREFIX_GROUPS = 5       # distinct "system prompts"
SEED = 42

# Shared prefix fractions to sweep
SHARED_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

# Cache sizing: fraction of total blocks that could be needed
# Total blocks if every request were fully cached = NUM_REQUESTS * (PREFILL_TOKENS // BLOCK_SIZE)
# A plausible on-device cache is much smaller. We size it relative to the
# *unique working set* — enough to hold all prefix groups plus some suffix
# blocks, but not every request.
#
# With 5 prefix groups @ 512 prefill tokens => 32 blocks/request.
# Working set of prefixes: 5 groups * 32 blocks = 160 blocks (at 100% shared).
# We provide ~2x that to allow suffix blocks too, capped at a reasonable GPU
# KV cache size.  Real A100 80GB with Llama-70B holds ~2000-4000 blocks.
CACHE_MAX_BLOCKS = 400  # plausible mid-range


@dataclass
class ExperimentResult:
    shared_fraction: float
    num_requests: int
    prefill_tokens_per_request: int
    num_prefix_groups: int
    cache_max_blocks: int
    block_size: int

    # Per-request detail
    per_request: List[Dict[str, Any]]

    # Aggregate cache stats
    total_lookups: int = 0
    total_hits: int = 0
    total_misses: int = 0
    hit_rate: float = 0.0
    token_hit_rate: float = 0.0
    total_tokens_looked_up: int = 0
    total_tokens_hit: int = 0
    total_tokens_missed: int = 0
    total_insertions: int = 0
    total_blocks_inserted: int = 0
    total_evictions: int = 0
    total_blocks_evicted: int = 0
    cache_utilization: float = 0.0
    peak_cached_blocks: int = 0

    # Derived
    total_prefill_tokens_saved: int = 0
    effective_prefill_reduction: float = 0.0
    avg_tokens_saved_per_hit: float = 0.0


def run_experiment(shared_fraction: float) -> ExperimentResult:
    """Run one experiment with a given shared prefix fraction."""

    # 1. Generate requests
    requests = [
        Request(
            arrived_at=float(i) * 0.1,
            num_prefill_tokens=PREFILL_TOKENS,
            num_decode_tokens=DECODE_TOKENS,
        )
        for i in range(NUM_REQUESTS)
    ]

    # 2. Assign token IDs with the given sharing fraction
    config = PrefixCacheConfig(
        enabled=True,
        num_shared_prefixes=NUM_PREFIX_GROUPS,
        shared_prefix_length_fraction=shared_fraction,
        seed=SEED,
    )
    gen = PrefixTokenGenerator(config)
    gen.assign_token_ids(requests)

    # 3. Initialize cache
    cache = PrefixCacheManager(max_blocks=CACHE_MAX_BLOCKS, block_size=BLOCK_SIZE)

    # 4. Simulate request arrival and completion
    per_request = []
    peak_blocks = 0
    total_saved = 0

    for i, req in enumerate(requests):
        # Lookup
        cached_tokens = cache.match_prefix(req.token_ids)

        # Determine which prefix group this request belongs to
        # (first token encodes the group via _PREFIX_TOKEN_BASE offset)
        first_token = req.token_ids[0] if req.token_ids else -1
        shared_len = int(PREFILL_TOKENS * shared_fraction)

        record = {
            "request_id": i,
            "first_token_id": first_token,
            "shared_prefix_len": shared_len,
            "cached_tokens": cached_tokens,
            "prefill_before": PREFILL_TOKENS,
            "prefill_after": PREFILL_TOKENS - cached_tokens,
            "is_hit": cached_tokens > 0,
        }

        if cached_tokens > 0:
            req.apply_prefix_cache_hit(cached_tokens)
            total_saved += cached_tokens

        # Complete the request -> insert into cache
        cache.on_request_complete(req.token_ids)

        peak_blocks = max(peak_blocks, cache.num_cached_blocks)
        per_request.append(record)

    # 5. Collect stats
    stats = cache.stats
    result = ExperimentResult(
        shared_fraction=shared_fraction,
        num_requests=NUM_REQUESTS,
        prefill_tokens_per_request=PREFILL_TOKENS,
        num_prefix_groups=NUM_PREFIX_GROUPS,
        cache_max_blocks=CACHE_MAX_BLOCKS,
        block_size=BLOCK_SIZE,
        per_request=per_request,
        total_lookups=stats.total_lookups,
        total_hits=stats.total_hits,
        total_misses=stats.total_misses,
        hit_rate=stats.hit_rate,
        token_hit_rate=stats.token_hit_rate,
        total_tokens_looked_up=stats.total_tokens_looked_up,
        total_tokens_hit=stats.total_tokens_hit,
        total_tokens_missed=stats.total_tokens_missed,
        total_insertions=stats.total_insertions,
        total_blocks_inserted=stats.total_blocks_inserted,
        total_evictions=stats.total_evictions,
        total_blocks_evicted=stats.total_blocks_evicted,
        cache_utilization=cache.cache_utilization,
        peak_cached_blocks=peak_blocks,
        total_prefill_tokens_saved=total_saved,
        effective_prefill_reduction=(
            total_saved / (NUM_REQUESTS * PREFILL_TOKENS) if NUM_REQUESTS > 0 else 0.0
        ),
        avg_tokens_saved_per_hit=(
            total_saved / stats.total_hits if stats.total_hits > 0 else 0.0
        ),
    )
    return result


def print_result(r: ExperimentResult) -> None:
    """Print a single experiment result."""
    print(f"\n{'='*70}")
    print(f"  Shared Prefix Fraction: {r.shared_fraction:.0%}")
    print(f"{'='*70}")
    print(f"  Workload: {r.num_requests} requests, {r.prefill_tokens_per_request} "
          f"prefill tokens each, {r.num_prefix_groups} prefix groups")
    print(f"  Cache: {r.cache_max_blocks} blocks × {r.block_size} tokens/block "
          f"= {r.cache_max_blocks * r.block_size} token capacity")
    print()
    print(f"  --- Cache Hit Metrics ---")
    print(f"  Hit rate (request-level):  {r.hit_rate:.2%}  "
          f"({r.total_hits} hits / {r.total_lookups} lookups)")
    print(f"  Hit rate (token-level):    {r.token_hit_rate:.2%}  "
          f"({r.total_tokens_hit} / {r.total_tokens_looked_up} tokens)")
    print(f"  Total misses:              {r.total_misses}")
    print()
    print(f"  --- Token Savings ---")
    print(f"  Total prefill tokens saved:   {r.total_prefill_tokens_saved:,}")
    print(f"  Effective prefill reduction:  {r.effective_prefill_reduction:.2%}")
    print(f"  Avg tokens saved per hit:     {r.avg_tokens_saved_per_hit:.1f}")
    print()
    print(f"  --- Cache Operations ---")
    print(f"  Insertions:     {r.total_insertions} ({r.total_blocks_inserted} blocks)")
    print(f"  Evictions:      {r.total_evictions} ({r.total_blocks_evicted} blocks)")
    print(f"  Peak blocks:    {r.peak_cached_blocks} / {r.cache_max_blocks}")
    print(f"  Final util:     {r.cache_utilization:.2%}")

    # Show first few and last few per-request details
    print()
    print(f"  --- Per-Request Sample (first 5 + last 5) ---")
    print(f"  {'Req':>4} {'Hit?':>5} {'Cached':>7} {'Prefill→':>9} {'Saved':>6}")
    for rec in r.per_request[:5]:
        print(f"  {rec['request_id']:4d} {'  YES' if rec['is_hit'] else '   NO'} "
              f"{rec['cached_tokens']:7d} {rec['prefill_before']}→{rec['prefill_after']:4d} "
              f"{rec['cached_tokens']:6d}")
    if len(r.per_request) > 10:
        print(f"  {'...':>4}")
    for rec in r.per_request[-5:]:
        print(f"  {rec['request_id']:4d} {'  YES' if rec['is_hit'] else '   NO'} "
              f"{rec['cached_tokens']:7d} {rec['prefill_before']}→{rec['prefill_after']:4d} "
              f"{rec['cached_tokens']:6d}")


def print_comparison_table(results: List[ExperimentResult]) -> None:
    """Print a side-by-side comparison table."""
    print(f"\n{'='*90}")
    print(f"  COMPARISON ACROSS SHARED PREFIX FRACTIONS")
    print(f"{'='*90}")
    print(f"  {'Frac':>5} │ {'Hit Rate':>9} │ {'Tok Hit%':>9} │ {'Saved Tok':>10} │ "
          f"{'Prefill↓':>9} │ {'Evictions':>9} │ {'Blk Evict':>9} │ {'Peak Blk':>9}")
    print(f"  {'─'*5}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*10}─┼─"
          f"{'─'*9}─┼─{'─'*9}─┼─{'─'*9}─┼─{'─'*9}")
    for r in results:
        print(f"  {r.shared_fraction:5.0%} │ {r.hit_rate:9.2%} │ {r.token_hit_rate:9.2%} │ "
              f"{r.total_prefill_tokens_saved:10,} │ {r.effective_prefill_reduction:9.2%} │ "
              f"{r.total_evictions:9d} │ {r.total_blocks_evicted:9d} │ "
              f"{r.peak_cached_blocks:4d}/{r.cache_max_blocks}")


def main():
    print("=" * 70)
    print("  KV CACHE PREFIX SIMULATION EXPERIMENT")
    print("=" * 70)
    print(f"\n  Configuration:")
    print(f"    Requests:        {NUM_REQUESTS}")
    print(f"    Prefill tokens:  {PREFILL_TOKENS}")
    print(f"    Decode tokens:   {DECODE_TOKENS}")
    print(f"    Block size:      {BLOCK_SIZE} tokens")
    print(f"    Prefix groups:   {NUM_PREFIX_GROUPS}")
    print(f"    Cache capacity:  {CACHE_MAX_BLOCKS} blocks ({CACHE_MAX_BLOCKS * BLOCK_SIZE} tokens)")
    print(f"    Fractions:       {SHARED_FRACTIONS}")
    print(f"    Seed:            {SEED}")

    results = []
    for frac in SHARED_FRACTIONS:
        r = run_experiment(frac)
        results.append(r)
        print_result(r)

    print_comparison_table(results)

    # Save raw results to JSON
    output = {
        "config": {
            "num_requests": NUM_REQUESTS,
            "prefill_tokens": PREFILL_TOKENS,
            "decode_tokens": DECODE_TOKENS,
            "block_size": BLOCK_SIZE,
            "num_prefix_groups": NUM_PREFIX_GROUPS,
            "cache_max_blocks": CACHE_MAX_BLOCKS,
            "seed": SEED,
        },
        "results": [],
    }
    for r in results:
        output["results"].append({
            "shared_fraction": r.shared_fraction,
            "hit_rate": r.hit_rate,
            "token_hit_rate": r.token_hit_rate,
            "total_hits": r.total_hits,
            "total_misses": r.total_misses,
            "total_tokens_hit": r.total_tokens_hit,
            "total_tokens_missed": r.total_tokens_missed,
            "total_prefill_tokens_saved": r.total_prefill_tokens_saved,
            "effective_prefill_reduction": r.effective_prefill_reduction,
            "avg_tokens_saved_per_hit": r.avg_tokens_saved_per_hit,
            "total_insertions": r.total_insertions,
            "total_blocks_inserted": r.total_blocks_inserted,
            "total_evictions": r.total_evictions,
            "total_blocks_evicted": r.total_blocks_evicted,
            "peak_cached_blocks": r.peak_cached_blocks,
            "cache_utilization": r.cache_utilization,
        })

    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "experiment_kv_cache_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Raw results saved to {results_path}")


if __name__ == "__main__":
    main()
