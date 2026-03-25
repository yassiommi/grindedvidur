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

The key insight is that **cache utilization stays high (~90%) while hit rate drops sharply** — the cache is full, but full of the wrong data. Standard utilization metrics mask the problem entirely.

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
| Total sessions | 40 | Enough to observe all three phases |
| Block size | 16 tokens | |

### Interleaving Model

Concurrent sessions are interleaved round-robin: each "tick" advances one step in every active session, modeling parallel agent execution on a shared GPU. When a session completes its 12 steps, a new session takes its slot.

### Sweep Parameters

- **Concurrent sessions:** 2, 4, 6, 8, 12
- **Cache sizes:** 200, 400, 600, 800, 1200 blocks

The critical ratio is **working set / cache capacity**. A single session at max length requires 169 blocks, so 8 concurrent sessions need ~1352 blocks. When the cache holds fewer blocks than the working set demands, thrashing begins.

---

## 3. Results

### 3.1 The Three Phases

The 6-concurrent / 600-block configuration cleanly demonstrates all three phases:

![Thrashing Detail](../report_figures/kv_cache/thrashing_detail.png)

| Phase | Request Range | Token Hit Rate | Cache Utilization | Evictions/req |
|-------|:---:|:---:|:---:|:---:|
| **Warmup** | 0–41 | 71.5% | 40% | 0 |
| **Thrashing** | 42–77 | **36.2%** | **92%** | 92.5 |
| **Settling** | 78–519 | 58.5% | 89% | 45.0 |

**Warmup** (requests 0–41): The cache is filling up. Each session's first step is a cold miss, but subsequent steps reuse the growing prefix from prior steps. No evictions occur because the cache has room.

**Thrashing** (requests 42–77): The aggregate working set of 6 sessions crosses the 600-block capacity. Every new step from one session evicts blocks belonging to another. Evictions spike to ~92 blocks/request while hit rate drops to 36%. The cache stays 92% full — it's churning, not empty.

**Settling** (requests 78+): As sessions complete and leave the cache, eviction pressure decreases. Hit rate partially recovers but remains depressed because new sessions keep entering.

### 3.2 Severity Scales with Overcommit Ratio

Comparing the severe case (12 concurrent, 400 blocks) against no-thrashing (2 concurrent, 1200 blocks):

**Severe thrashing (working set / cache = 5.1×):**

![Severe Thrashing](../report_figures/kv_cache/thrashing_severe.png)

**No thrashing (working set / cache = 0.3×):**

![No Thrashing](../report_figures/kv_cache/thrashing_none.png)

Under severe thrashing, the token hit rate collapses to **18%** mid-phase with 91 blocks evicted per request — nearly every cache entry is replaced on every step. Without thrashing, the hit rate reaches **80%** with steady, low-volume evictions.

### 3.3 Thrashing Boundary Heatmap

![Thrashing Heatmap](../report_figures/kv_cache/thrashing_heatmap.png)

The heatmap maps mid-phase token hit rate across all (concurrent sessions × cache size) configurations. The boundary is sharp: configurations below the diagonal (working set < cache) achieve 60–80% hit rates, while those above it (working set > cache) drop to 20–35%.

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
- **2 concurrent:** Smooth 80%+ hit rate, working set (338 blocks) fits comfortably
- **4 concurrent:** Mild dips when context grows large (working set peaks at 676)
- **6 concurrent:** Clear thrashing oscillations, 1014-block working set overflows the cache
- **8–12 concurrent:** Sustained thrashing, hit rate frequently drops below 20%

All configurations show **near-100% cache utilization** in the bottom panel — confirming that utilization is a misleading health metric during thrashing.

---

## 4. Key Findings

### Finding 1: Cache utilization masks thrashing

During the worst thrashing (12 concurrent, 400 blocks), cache utilization is **87%** while token hit rate is **18%**. A monitoring system that only tracks utilization would report the cache as healthy. The fix is to track **eviction rate** and **hit rate** jointly: high utilization + high eviction + low hit rate = thrashing.

### Finding 2: The thrashing boundary is sharp

There is no graceful degradation. When the working set exceeds cache capacity by even a small margin (1.3×), the hit rate drops by ~20 percentage points. At 2.5× overcommit, it drops by ~50 points. This is because LRU eviction is adversarial to round-robin interleaving: the session whose blocks were evicted is always the one that needs them next.

### Finding 3: Agentic workloads are uniquely vulnerable

Unlike multi-turn chat (where contexts are short and few users share the same conversation), agentic workloads have:
- **Long, growing contexts** (2000+ tokens after 10 steps)
- **High temporal correlation** (each step depends on the previous)
- **Multiple concurrent sessions** competing for the same cache
- **No natural sharing** between sessions (each task is unique)

This combination creates the worst case for LRU caching: every entry is needed exactly once more, but evicted just before that access.

### Finding 4: The sawtooth pattern reveals batch structure

The thrashing detail plot shows a distinctive **sawtooth** in both hit rate and I/O volume. Each tooth corresponds to one "generation" of concurrent sessions:
- Rising edge: sessions reuse their own growing prefix (hits)
- Falling edge: contexts grow past cache capacity (evictions begin)
- Valley: peak thrashing as all sessions compete simultaneously

This sawtooth is a diagnostic signature — if you see it in production telemetry, the system is thrashing.

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

The most actionable finding: for agentic workloads, **cache capacity should be provisioned as `N × max_context_length / block_size`** where N is the target concurrent session count. Under-provisioning by even 30% triggers sharp performance degradation.

---

## 6. Reproducing

```bash
python experiments/experiment_thrashing.py
```

Output: console characterization + 5 plots in `report_figures/kv_cache/thrashing_*.png`.
