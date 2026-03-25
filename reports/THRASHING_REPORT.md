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
| Total sessions | 60 | Enough to observe all three phases |
| Block size | 16 tokens | |

### Three-Phase Lifecycle Model

The simulation models a realistic production traffic pattern with three distinct phases:

**Phase 1 — Ramp-Up:** Sessions arrive one at a time (one every 2 ticks) until the pool reaches N concurrent sessions. Each session starts at step 0 (cold start). The cache warms up gradually — hit rate rises as each new session benefits from the shared system prompt already cached by earlier sessions. Cache pressure builds steadily.

**Phase 2 — Sustained Load:** The pool stays full at N concurrent sessions. When a session completes, it is immediately replaced by a new session at step 0. This is controlled by a `sustained_replacements` budget (default: 30). If the working set exceeds cache capacity, this phase is sustained thrashing — the system never recovers because new sessions keep entering and maintaining pressure. Sessions at different lifecycle stages create a mixed working set that persists for the entire phase.

**Phase 3 — Drain:** No more replacement sessions arrive. Active sessions complete one by one. Cache pressure drops as the working set shrinks. Evictions slow, and hit rate may recover as the remaining sessions fit within cache capacity. The cache "cools down" as the pool empties.

This three-phase model is more realistic than either batch-synchronized arrivals (which produce repeating sawtooth patterns) or purely staggered arrivals (which skip warmup). In production, services ramp up, run under sustained load, and eventually drain during scale-down or deployment.

### Sweep Parameters

- **Concurrent sessions:** 2, 4, 6, 8, 12
- **Cache sizes:** 200, 400, 600, 800, 1200 blocks

The critical ratio is **working set / cache capacity**. A single session at max length requires 169 blocks, so 8 concurrent sessions need ~1352 blocks. When the cache holds fewer blocks than the working set demands, thrashing begins — and with sustained replacement of completed sessions, it persists for the entire sustained-load phase.

---

## 3. Results

### 3.1 Sustained Thrashing (Phase 2)

During the sustained-load phase, overcommitted configurations thrash continuously with no recovery. The 12-concurrent / 400-block case (5.1x overcommit) shows the pattern clearly:

![Severe Thrashing](../report_figures/kv_cache/thrashing_severe.png)

| Config | Phase | Token Hit Rate | Cache Util | Evictions/req |
|--------|-------|:---:|:---:|:---:|
| 12 conc, 400 blk | Sustained thrashing | **~18%** | **~83%** | ~84 |
| 8 conc, 400 blk | Sustained thrashing | **~19%** | **~85%** | ~82 |
| 6 conc, 400 blk | Sustained thrashing | **~26%** | **~87%** | ~75 |
| 12 conc, 800 blk | Sustained thrashing | **~24%** | **~93%** | ~76 |

In every case: high utilization, high eviction rate, low hit rate. The warmup phase is visible at the start (hit rate initially higher as sessions are still loading), followed by sustained thrashing that persists until the drain phase.

### 3.2 No Thrashing (Working Set Fits)

When the working set fits in the cache, the three-phase pattern shows healthy behavior throughout:

![No Thrashing](../report_figures/kv_cache/thrashing_none.png)

| Config | Token Hit Rate | Cache Util | Evictions/req |
|--------|:---:|:---:|:---:|
| 2 conc, 400 blk | ~80% | ~96% | ~11 |
| 2 conc, 800 blk | ~80% | ~94% | ~11 |
| 4 conc, 800 blk | ~80% | ~95% | ~11 |

Hit rate is stable at ~80% across all three phases, evictions are low and steady (just LRU turnover of completed sessions), and there are no phase transitions in cache behavior.

### 3.3 Thrashing Boundary Heatmap

![Thrashing Heatmap](../report_figures/kv_cache/thrashing_heatmap.png)

The heatmap maps mid-phase token hit rate across all (concurrent sessions x cache size) configurations. The boundary is sharp and binary: configurations where the working set fits achieve ~80% hit rate, those that don't drop to 17-25%.

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
- **8 concurrent:** Sustained thrashing at ~20-30%, with brief spikes when sessions complete during the drain phase
- **12 concurrent:** Pinned at ~15-20% for the entire sustained-load phase — unrecoverable thrashing until drain

The bottom panel shows **near-100% cache utilization across all configurations** — confirming that utilization is completely uninformative about whether the cache is actually helping.

### 3.5 The Detailed View

The 6-concurrent / 600-block case (1.7x overcommit) shows the borderline behavior with all three phases visible:

![Thrashing Detail](../report_figures/kv_cache/thrashing_detail.png)

- **Warmup (early requests):** Sessions arrive one by one. Cache fills gradually. Hit rate starts high as early sessions benefit from the shared system prompt.
- **Sustained load (middle):** Pool is full with sessions at different lifecycle stages. Working set exceeds capacity. Hit rate drops and stays low as sessions continuously evict each other's blocks.
- **Drain (late requests):** No replacements. As sessions complete and leave, the working set shrinks below capacity. Evictions slow and hit rate may recover for the final sessions.

---

## 4. Key Findings

### Finding 1: Cache utilization masks thrashing

During severe thrashing (12 concurrent, 400 blocks), cache utilization is **~83%** while token hit rate is **~18%**. A monitoring system that only tracks utilization would report the cache as healthy. The diagnostic triad is: **high utilization + high eviction rate + low hit rate = thrashing**.

### Finding 2: The thrashing boundary is a cliff, not a slope

There is no graceful degradation. Hit rate is either ~80% (working set fits) or ~20% (it doesn't). The transition is nearly binary — a 30% increase in concurrent sessions can cause a 60-point drop in hit rate. This is because LRU eviction under round-robin access is adversarial: the least-recently-used entry is always the one that will be needed next.

### Finding 3: The three-phase lifecycle makes thrashing visible

The warmup-sustained-drain lifecycle creates a clear narrative:
- **Warmup** establishes the baseline — high hit rates as sessions load and the cache fills.
- **Sustained load** reveals the steady-state behavior — if the working set exceeds capacity, thrashing persists indefinitely because completed sessions are immediately replaced.
- **Drain** shows recovery potential — as sessions leave without replacement, the working set shrinks and hit rate can recover.

This is more realistic than purely staggered arrivals (which skip warmup) or synchronized batches (which produce artificial sawtooth recovery patterns).

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

The most actionable finding: for agentic workloads, **cache capacity should be provisioned as `N x max_context_length / block_size`** where N is the target concurrent session count. Under-provisioning by even 30% triggers sharp, sustained performance degradation with no self-recovery.

---

## 6. Reproducing

```bash
python experiments/experiment_thrashing.py
```

Output: console characterization + 5 plots in `report_figures/kv_cache/thrashing_*.png`.
