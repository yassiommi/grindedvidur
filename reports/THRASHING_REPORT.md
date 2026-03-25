# KV Cache Thrashing in Agentic Inference — Experiment Report

**Date:** 2026-03-25
**Experiment:** `experiments/experiment_thrashing.py`
**Component:** `vidur.entities.prefix_cache_manager`

---

## 1. Motivation

In agentic inference (tool-use loops, chain-of-thought, multi-step planning), each agent step appends tokens to a growing context window:

```
Step 0: [system_prompt | user_task]                                          →  300 tokens
Step 1: [system_prompt | user_task | thought₁ | tool_call₁ | result₁]       →  500 tokens
Step 2: [... | thought₂ | tool_call₂ | result₂]                             →  700 tokens
  ⋮
Step 12: [...]                                                               → 2700 tokens
```

Each step shares the full prefix of all prior steps. With **N concurrent agent sessions**, the aggregate working set grows until it exceeds cache capacity. At that point, inserting new blocks for session A evicts blocks belonging to session B — but session B needs exactly those blocks on its very next step. This creates a destructive cycle of **evict → insert → evict** that we call **mid-phase thrashing**.

The key insight is that **cache utilization stays high (~85-98%) while hit rate drops sharply** — the cache is full, but full of the wrong data. Standard utilization metrics mask the problem entirely.

---

## 2. Experiment Setup

### Agent Session Model

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| System prompt | 200 tokens | Shared across all sessions |
| Task description | 100 tokens | Unique per session |
| Tokens per step | ~200 (thought: 60, tool call: 30, tool result: ~110) | Models typical tool-use output |
| Steps per session | 12 | Realistic agentic loop |
| Max context at completion | 2700 tokens = 169 blocks | Upper bound per session |
| Total sessions | 40 | Enough to observe sustained thrashing |
| Block size | 16 tokens | |

### Staggered Arrival Model

Sessions are **staggered**: the initial N concurrent sessions start at different points in their lifecycle (evenly spread across steps 0..12), modeling a system that has been running under load. When a session completes, a new one immediately takes its slot at step 0. This keeps the pool permanently full with sessions at mixed stages — no synchronized generations, no artificial recovery gaps.

This is more realistic than batch-synchronized arrivals, which produce a repeating sawtooth pattern as entire generations start and finish together. In production, sessions arrive and depart continuously.

### Sweep Parameters

- **Concurrent sessions:** 2, 4, 6, 8, 12
- **Cache sizes:** 200, 400, 600, 800, 1200 blocks

The critical ratio is **working set / cache capacity**. A single session at max length requires 169 blocks, so 8 concurrent sessions need ~1352 blocks. When the cache holds fewer blocks than the working set demands, thrashing begins — and with staggered arrivals, it persists for the entire simulation.

---

## 3. Results

### 3.1 Sustained Thrashing

With staggered arrivals, overcommitted configurations enter thrashing immediately and **never recover**. The 12-concurrent / 400-block case (5.1× overcommit) thrashes for all 454 requests:

![Severe Thrashing](../report_figures/kv_cache/thrashing_severe.png)

| Config | Phase | Requests | Token Hit Rate | Cache Util | Evictions/req |
|--------|-------|:---:|:---:|:---:|:---:|
| 12 conc, 400 blk | **Thrashing** | 0–453 | **18.4%** | **83%** | 83.5 |
| 8 conc, 400 blk | **Thrashing** | 0–479 | **19.0%** | **85%** | 81.7 |
| 6 conc, 400 blk | **Thrashing** | 0–489 | **25.5%** | **87%** | 74.6 |
| 12 conc, 800 blk | **Thrashing** | 0–453 | **24.0%** | **93%** | 75.8 |

In every case: high utilization, high eviction rate, low hit rate — the full simulation is one sustained thrashing phase with no recovery.

### 3.2 No Thrashing (Working Set Fits)

When the working set fits in the cache, the pattern is completely different:

![No Thrashing](../report_figures/kv_cache/thrashing_none.png)

| Config | Token Hit Rate | Cache Util | Evictions/req |
|--------|:---:|:---:|:---:|
| 2 conc, 400 blk | 80.4% | 96% | 11.3 |
| 2 conc, 800 blk | 80.4% | 94% | 10.6 |
| 4 conc, 800 blk | 80.3% | 95% | 10.8 |

Hit rate is stable at ~80%, evictions are low and steady (just LRU turnover of completed sessions), and there are no phase transitions.

### 3.3 Thrashing Boundary Heatmap

![Thrashing Heatmap](../report_figures/kv_cache/thrashing_heatmap.png)

The heatmap maps mid-phase token hit rate across all (concurrent sessions × cache size) configurations. The boundary is sharp and binary: configurations where the working set fits achieve ~80% hit rate, those that don't drop to 17-25%.

| Concurrent Sessions | Working Set (blocks) | Min Cache to Avoid Thrashing |
|:---:|:---:|:---:|
| 2 | 338 | ~400 |
| 4 | 676 | ~800 |
| 6 | 1014 | ~1200 |
| 8 | 1352 | ~1400+ |
| 12 | 2028 | ~2100+ |

### 3.4 Concurrent Sessions Sweep (Fixed Cache = 600 Blocks)

![Concurrent Sweep](../report_figures/kv_cache/thrashing_concurrent_sweep.png)

At a fixed 600-block cache:
- **2-4 concurrent:** Stable 80%+ hit rate — working set fits comfortably
- **8 concurrent:** Sustained thrashing at ~20-30%, with brief spikes when a session that just completed frees space before a new one fills it
- **12 concurrent:** Pinned at ~15-20% for the entire run — sustained, unrecoverable thrashing

The bottom panel shows **near-100% cache utilization across all configurations** — confirming that utilization is completely uninformative about whether the cache is actually helping.

### 3.5 The Detailed View

The 6-concurrent / 600-block case (1.7× overcommit) shows the borderline behavior:

![Thrashing Detail](../report_figures/kv_cache/thrashing_detail.png)

After a brief warmup (staggered sessions loading their initial contexts), the system reaches steady state. With staggered arrivals, the working set varies continuously as sessions at different stages cycle through, creating persistent mild pressure rather than periodic crises.

---

## 4. Key Findings

### Finding 1: Cache utilization masks thrashing

During severe thrashing (12 concurrent, 400 blocks), cache utilization is **83%** while token hit rate is **18%**. A monitoring system that only tracks utilization would report the cache as healthy. The diagnostic triad is: **high utilization + high eviction rate + low hit rate = thrashing**.

### Finding 2: The thrashing boundary is a cliff, not a slope

There is no graceful degradation. Hit rate is either ~80% (working set fits) or ~20% (it doesn't). The transition is nearly binary — a 30% increase in concurrent sessions can cause a 60-point drop in hit rate. This is because LRU eviction under round-robin access is adversarial: the least-recently-used entry is always the one that will be needed next.

### Finding 3: Staggered arrivals make thrashing worse, not better

With synchronized generations, thrashing is periodic — sessions start together, thrash during the middle steps, then all complete and the cache recovers before the next batch. With realistic staggered arrivals, there is **no recovery window**. The pool is always full of sessions at different stages, so the working set never drops below the thrashing threshold. The system enters thrashing and stays there permanently.

### Finding 4: Agentic workloads are uniquely vulnerable

Unlike multi-turn chat (short contexts, few concurrent users sharing a conversation), agentic workloads have:
- **Long, growing contexts** (2000+ tokens after 10 steps)
- **High temporal correlation** (each step depends on the previous — no reordering possible)
- **Multiple concurrent sessions** competing for the same cache
- **No natural sharing** between sessions (each task is unique)

This combination creates the worst case for LRU caching: every entry is needed exactly once more, but evicted just before that access.

---

## 5. Implications for System Design

| Strategy | Effect | Tradeoff |
|----------|--------|----------|
| **Increase cache size** | Directly raises thrashing boundary | Memory cost scales linearly with concurrent sessions |
| **Limit concurrent sessions** | Keeps working set within cache | Reduces throughput; may increase queuing latency |
| **Session-aware eviction** | Protect active sessions' blocks from eviction | More complex eviction policy; may starve new sessions |
| **Prefix deduplication** | Share system prompt blocks across sessions | Only helps if sessions share a common prefix (~200/2700 = 7% here) |
| **Context compression** | Reduce per-step token growth | Lossy; may degrade agent accuracy |
| **Priority scheduling** | Complete one session before starting the next | Eliminates inter-session thrashing but serializes execution |

The most actionable finding: for agentic workloads, **cache capacity should be provisioned as `N × max_context_length / block_size`** where N is the target concurrent session count. Under-provisioning by even 30% triggers sharp, sustained performance degradation with no self-recovery.

---

## 6. Reproducing

```bash
python experiments/experiment_thrashing.py
```

Output: console characterization + 5 plots in `report_figures/kv_cache/thrashing_*.png`.
