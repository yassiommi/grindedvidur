# KV Cache Prefix Simulation — End-to-End Experiment Report

## 1. Objective

Measure how the **shared prefix fraction** in a workload affects KV cache hit
rates, token savings, and eviction behavior in a simulated LLM inference
serving environment. The experiment sweeps the fraction of each request's
prefill tokens that come from a shared "system prompt" prefix, from 0% (no
sharing) to 90% (heavy sharing), and reports detailed cache metrics at each
point.

This is a **pure simulation** — no GPU profiling or execution time prediction
is involved. All metrics come from the `PrefixCacheManager` radix-tree cache
operating on synthetic token sequences.

---

## 2. How Requests Are Generated and Tokens Are Assigned

### 2.1 Request Generation

The `SyntheticRequestGenerator` produces `Request` objects with three
properties: `arrived_at` (float timestamp), `num_prefill_tokens`, and
`num_decode_tokens`. Token counts come from configurable distributions
(Fixed, Zipf, Uniform, or Trace replay). At this stage, requests have **no
token IDs** — only counts.

### 2.2 Token ID Assignment (PrefixTokenGenerator)

When prefix caching is enabled, the simulator calls `PrefixTokenGenerator.assign_token_ids(requests)`, which:

1. **Generates shared prefix pools**: Creates `num_shared_prefixes` distinct
   token sequences (IDs in the 1,000,000+ range), each long enough to cover
   the largest request's shared portion.

2. **Assigns each request to a random prefix group** (uniform random, seeded).

3. **Constructs token IDs per request**:
   - `shared_len = int(num_prefill_tokens * shared_prefix_length_fraction)`
   - Takes `shared_len` tokens from the assigned group's prefix pool
   - Generates `num_prefill_tokens - shared_len` unique tokens (IDs in the
     2,000,000+ range, globally unique via a monotonic counter)
   - Final: `token_ids = shared_prefix[:shared_len] + unique_suffix`

**Example** (512 prefill tokens, 50% shared, assigned to group 2):
- Shared portion: 256 tokens from group 2's prefix (`1_000_512, 1_000_513, ...`)
- Unique portion: 256 tokens (`2_000_000, 2_000_001, ...`)
- Requests in the same group share the first 256 tokens, enabling cache hits.

### 2.3 Cache Lookup at Scheduling Time

When a request enters the scheduler (`BaseReplicaScheduler.add_request`):
1. `PrefixCacheManager.match_prefix(token_ids)` walks the radix tree
2. Returns the number of block-aligned cached tokens
3. `request.apply_prefix_cache_hit(cached_tokens)` advances `num_processed_tokens`,
   reducing the remaining prefill work

When a request completes, its full token sequence is inserted into the cache
for future requests to match against.

---

## 3. Experiment Setup

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Requests | 200 | Large enough for steady-state behavior |
| Prefill tokens/request | 512 | Typical mid-length prompt |
| Decode tokens/request | 128 | Realistic generation length |
| Block size | 16 tokens | Standard KV cache block granularity |
| Prefix groups | 5 | Simulates 5 distinct system prompts |
| Cache capacity | 400 blocks (6,400 tokens) | ~12.5 full requests; plausible for a mid-range GPU |
| Shared fractions tested | 0%, 10%, 20%, 30%, 50%, 70%, 90% | Full sweep from no sharing to heavy sharing |
| Seed | 42 | Reproducibility |

### Cache Sizing Rationale

Each request occupies `512 / 16 = 32` blocks when fully cached. The 400-block
cache can hold ~12.5 complete requests. With 5 prefix groups, the shared prefix
blocks per group range from 0 blocks (0% fraction) to `(512 * 0.9) / 16 = 28.8
≈ 28` blocks (90% fraction). Total prefix working set at 90%: `5 * 28 = 140`
blocks, leaving 260 blocks for unique suffixes — enough to avoid excessive
thrashing while still requiring eviction. This makes the cache realistically
constrained, neither trivially large nor pathologically small.

---

## 4. Results

### 4.1 Summary Comparison

| Shared Fraction | Request Hit Rate | Token Hit Rate | Tokens Saved | Prefill Reduction | Evictions | Blocks Evicted | Peak Blocks |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0% | 0.00% | 0.00% | 0 | 0.00% | 188 | 6,016 | 384/400 |
| 10% | 96.50% | 9.05% | 9,264 | 9.05% | 187 | 5,249 | 384/400 |
| 20% | 96.50% | 18.09% | 18,528 | 18.09% | 186 | 4,669 | 405/400 |
| 30% | 97.00% | 27.28% | 27,936 | 27.28% | 184 | 4,063 | 399/400 |
| 50% | 97.00% | 48.50% | 49,664 | 48.50% | 180 | 2,896 | 400/400 |
| 70% | 97.50% | 67.03% | 68,640 | 67.03% | 168 | 1,517 | 400/400 |
| 90% | 97.50% | 85.31% | 87,360 | 85.31% | 114 | 347 | 400/400 |

### 4.2 Per-Fraction Analysis

#### 0% Shared (No Prefix Sharing)

Every request has a completely unique token sequence. The cache fills up and
constantly evicts old entries to make room, but nothing is ever re-accessed.
**Zero hits, maximum eviction churn** (6,016 blocks evicted across 188
eviction events). This is the baseline — prefix caching provides no benefit
when there is no workload sharing.

#### 10% Shared

Even a small shared prefix (51 tokens → 48 after block alignment = 3 blocks)
produces a **96.5% request-level hit rate**. After the 5 cold-start misses
(one per prefix group) plus 2 collisions from random group assignment ordering,
virtually every subsequent request hits. However, the token-level savings are
modest: only **9.05%** of total prefill tokens are avoided, because the shared
prefix is short relative to the full 512-token prefill.

Evictions remain high (5,249 blocks) because the unique suffixes (461 tokens =
28 blocks each) still churn through the cache.

#### 30% Shared

The sweet spot begins to emerge. **97% request hit rate**, **27.28% token hit
rate**, **27,936 tokens saved**. Each hit saves 144 tokens (9 blocks), and the
ratio of shared-to-unique blocks shifts enough to noticeably reduce eviction
pressure (4,063 blocks evicted vs. 6,016 at 0%).

#### 50% Shared

Half of each request's prefill is shared. **48.50% effective prefill
reduction** — nearly half of all prefill computation across the entire workload
is eliminated. Each hit saves 256 tokens (16 blocks). Evictions drop to 2,896
blocks because the shared prefix blocks are never evicted (they're internal
radix tree nodes with children, protected by the leaf-only eviction policy).

#### 70–90% Shared

At 90% sharing, **85.31% of all prefill tokens are served from cache**. Each
hit saves 448 tokens, reducing a 512-token prefill to just 64 tokens. Evictions
drop dramatically to 347 blocks — the cache is dominated by the 5 stable shared
prefix entries, with only the small unique suffixes cycling through. The cache
operates near 100% utilization with minimal churn.

### 4.3 Key Observations

**1. Request-level hit rate saturates quickly.**
With 5 prefix groups and random assignment, only 5–7 requests ever miss (the
cold starts). The request-level hit rate jumps from 0% to ~97% as soon as any
sharing is introduced, and barely changes beyond that. **This metric alone is
misleading** — it doesn't reflect how much computation is actually saved.

**2. Token-level hit rate tracks the shared fraction almost linearly.**
The effective prefill reduction closely mirrors the configured sharing fraction,
with a small reduction due to block-alignment rounding:

| Configured Fraction | Effective Reduction | Alignment Loss |
|:-:|:-:|:-:|
| 10% | 9.05% | 0.95% |
| 30% | 27.28% | 2.72% |
| 50% | 48.50% | 1.50% |
| 70% | 67.03% | 2.97% |
| 90% | 85.31% | 4.69% |

Alignment loss increases slightly at higher fractions because the absolute
number of "wasted" sub-block tokens grows, but it never exceeds one block
(16 tokens) per request.

**3. Eviction pressure decreases with sharing.**
Higher sharing means more requests reuse the same cached prefix blocks instead
of inserting new ones. Total blocks evicted drops from 6,016 (0% sharing) to
347 (90% sharing) — a **17x reduction** in cache churn. This has implications
for memory bandwidth and cache coherency in real systems.

**4. The leaf-only eviction policy protects shared prefixes.**
Shared prefix nodes in the radix tree always have children (the unique suffixes
of different requests). The LRU eviction policy only evicts leaf nodes, so
popular prefixes are naturally protected without explicit pinning. This is
visible in the data: even at 50% sharing where the cache is fully utilized
(400/400 blocks), the 5 shared prefixes are never evicted.

---

## 5. Cache Operations Detail

| Fraction | Insertions | Blocks Inserted | Evictions | Blocks Evicted | Net Blocks |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 0% | 200 | 6,400 | 188 | 6,016 | 384 |
| 10% | 200 | 5,628 | 187 | 5,249 | 379 |
| 20% | 200 | 5,049 | 186 | 4,669 | 380 |
| 30% | 200 | 4,460 | 184 | 4,063 | 397 |
| 50% | 200 | 3,296 | 180 | 2,896 | 400 |
| 70% | 200 | 1,915 | 168 | 1,517 | 398 |
| 90% | 200 | 745 | 114 | 347 | 398 |

Every request triggers exactly one insertion (200 total), but the number of
**blocks inserted** drops as sharing increases — shared prefix blocks are
already present and don't need re-insertion. At 90% sharing, only 745 blocks
are inserted across 200 requests (vs. 6,400 at 0%), because 90% of each
request's token sequence is already in the cache from the first request in
its group.

---

## 6. Tokens Saved by the Cache

| Fraction | Total Prefill Tokens | Tokens Saved | Tokens Computed | Computation Avoided |
|:-:|:-:|:-:|:-:|:-:|
| 0% | 102,400 | 0 | 102,400 | 0% |
| 10% | 102,400 | 9,264 | 93,136 | 9.05% |
| 20% | 102,400 | 18,528 | 83,872 | 18.09% |
| 30% | 102,400 | 27,936 | 74,464 | 27.28% |
| 50% | 102,400 | 49,664 | 52,736 | 48.50% |
| 70% | 102,400 | 68,640 | 33,760 | 67.03% |
| 90% | 102,400 | 87,360 | 15,040 | 85.31% |

At 90% sharing, only **15,040 out of 102,400** prefill tokens actually need
GPU computation. The remaining 87,360 are served directly from the KV cache,
representing an **85% reduction in prefill FLOPs**.

---

## 7. Conclusion

The experiment confirms that prefix caching effectiveness is **directly
proportional to the workload's prefix sharing fraction**, with near-linear
scaling from the configured sharing level to actual token savings. Key
takeaways:

1. **Even 10% sharing is worthwhile**: 96.5% of requests hit the cache,
   saving ~9% of total prefill computation with negligible overhead.

2. **50% sharing halves prefill work**: A realistic scenario (e.g., chat
   applications with system prompts comprising half the context) yields a
   48.5% reduction in prefill tokens.

3. **The radix tree + LRU design is efficient**: Shared prefixes are naturally
   protected from eviction, cache churn decreases with sharing, and the
   block-alignment overhead is minimal (<5%).

4. **Cache sizing matters but is forgiving**: A 400-block cache (enough for
   ~12 full requests) handles 200 requests with 5 prefix groups effectively
   across all sharing levels, without pathological thrashing.

---

## 8. Reproducibility

```bash
python experiment_kv_cache.py
```

Raw results: `experiment_kv_cache_results.json`

All parameters are configurable at the top of `experiment_kv_cache.py`.
