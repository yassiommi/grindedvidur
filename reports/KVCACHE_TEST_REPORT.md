# KV Cache / Prefix Cache Simulation — Test Report

**Date:** 2026-03-17
**Component:** `vidur.entities.prefix_cache_manager`, `vidur.entities.prefix_token_generator`, `vidur.entities.request` (prefix cache extensions)

---

## 1. Unit Test Summary

| # | Test Case | Result |
|---|-----------|--------|
| 1 | Basic insert and match | PASSED |
| 2 | Shared prefix hit | PASSED |
| 3 | LRU eviction | PASSED |
| 4 | LRU ordering | PASSED |
| 5 | No match for short sequence (< block size) | PASSED |
| 6 | Cache stats tracking | PASSED |
| 7 | `on_request_complete` caching | PASSED |
| 8 | Multiple prefix groups | PASSED |
| 9 | Empty cache behavior | PASSED |
| 10 | Prefix token generator | PASSED |
| 11 | Request prefix cache hit (`apply_prefix_cache_hit`) | PASSED |
| 12 | State summary output | PASSED |
| 13 | Integration: cache with scheduler flow | PASSED |

**Result: 13/13 passed, 0 failed**

---

## 2. Scenario-Based Integration Tests

### Scenario 1 — Multi-Turn Chat (Shared System Prompt)

Simulates 20 requests sharing a 256-token system prompt, each with a unique 64-token user message.

| Metric | Value |
|--------|-------|
| Requests | 20 |
| Hit rate | 95.00% |
| Token hit rate | 76.00% |
| First request hit | 0 tokens (cold start) |
| Subsequent request hit | 256 tokens (full system prompt) |
| Avg tokens saved | 243.2 per request |
| Cache utilization | 48.00% |
| Evictions | 0 |

**Interpretation:** After the cold-start miss, every subsequent request gets a full cache hit on the shared system prompt. The 76% token hit rate reflects the ratio of cached tokens (256) to total tokens (320) per request. No evictions occurred because the cache was large enough to hold all unique suffixes.

---

### Scenario 2 — Multiple Prefix Groups Under Eviction Pressure

5 prefix groups (128 tokens / 8 blocks each) competing for a 30-block cache (capacity for ~3.75 groups).

| Metric | Value |
|--------|-------|
| Prefix groups | 5 |
| Cache capacity | 30 blocks |
| Total requests | 15 (3 rounds × 5 groups) |
| Hit rate | 0.00% |
| Evictions | 12 |
| Blocks evicted | 120 |
| Cache utilization | 100.00% |

**Interpretation:** With 5 groups cycling through a cache that can hold fewer than 4, LRU eviction removes each group's blocks before its next access. This is the expected "thrashing" behavior — the cache is too small for the working set. The 0% hit rate confirms correct LRU eviction semantics under pressure.

---

### Scenario 3 — No Prefix Sharing (Unique Requests)

20 requests with completely unique token sequences (no shared prefixes).

| Metric | Value |
|--------|-------|
| Requests | 20 |
| Hit rate | 0.00% |
| Token hit rate | 0.00% |
| Evictions | 8 |

**Interpretation:** When there is no prefix sharing between requests, the cache correctly reports zero hits. This validates that the system does not produce false positives.

---

### Scenario 4 — End-to-End with PrefixTokenGenerator

Uses `PrefixTokenGenerator` to assign realistic token IDs with 3 shared prefix groups at 50% prefix length, then runs them through the cache manager.

| Metric | Value |
|--------|-------|
| Requests | 30 |
| Prefix groups | 3 |
| Shared prefix fraction | 50% |
| Prefill tokens per request | 200 |
| Cache hits | 27/30 (90%) |
| Token hit rate | 43.20% |
| Avg tokens saved per hit | 96.0 |
| Total prefill tokens saved | 2,592 |
| Effective prefill reduction | 43.20% |
| Cache utilization | 99.00% |

**Interpretation:** With 3 prefix groups and 30 requests, only the 3 cold-start requests miss. Each subsequent hit saves ~96 tokens (the shared prefix, block-aligned). The 43.2% effective prefill reduction closely tracks the configured 50% sharing fraction (the difference is due to block-size alignment rounding down the matchable prefix).

---

## 3. Backward Compatibility

All checks passed, confirming the prefix cache feature is fully opt-in and does not affect existing behavior:

| Check | Result |
|-------|--------|
| `VllmSchedulerConfig` defaults to `prefix_cache_config.enabled = False` | PASSED |
| `SarathiSchedulerConfig` defaults to `prefix_cache_config.enabled = False` | PASSED |
| `PddSchedulerConfig` defaults to `prefix_cache_config.enabled = False` | PASSED |
| `Request` without `token_ids` has correct defaults | PASSED |
| `Request` with `token_ids` accepts them | PASSED |
| `apply_prefix_cache_hit(40)` on 100-token request → 60 remaining | PASSED |
| `apply_prefix_cache_hit(999)` clamped to `num_prefill_tokens` | PASSED |
| `to_dict()` includes prefix cache fields when hit > 0 | PASSED |
| `to_dict()` omits prefix cache fields when no hit | PASSED |

---

## 4. Key Findings

1. **Core correctness verified.** All 13 unit tests pass. Insert, match, eviction, and stats tracking all behave as specified.

2. **Realistic workloads produce expected savings.** The multi-turn chat scenario shows a 76% token hit rate; the end-to-end generator scenario shows a 43.2% prefill reduction — both consistent with theoretical expectations.

3. **Edge cases handled correctly.** Cold starts produce zero hits, unique-prefix workloads produce zero false positives, over-sized hit values are clamped, and cache thrashing under eviction pressure behaves as expected (no corruption or incorrect hits).

4. **Fully backward compatible.** The feature is disabled by default across all scheduler configs. Existing `Request` objects work identically when no token IDs or cache hits are applied.

---

## 5. Conclusion

The KV cache / prefix cache simulation is **ready for integration**. All functional tests pass, performance characteristics match expectations across diverse workload patterns, and backward compatibility is fully preserved.
