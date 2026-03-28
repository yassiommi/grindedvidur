"""Simulate KV cache thrashing in agentic inference workloads.

In agentic inference (tool-use loops, chain-of-thought, multi-step planning),
each agent step appends tokens to a growing context:

  Step 0: [system_prompt | user_task]
  Step 1: [system_prompt | user_task | thought_1 | tool_call_1 | result_1]
  Step 2: [... | thought_2 | tool_call_2 | result_2]
  ...

With N concurrent agent sessions, the total working set grows until it exceeds
cache capacity. Then every new step from session A evicts blocks from session B,
but session B needs those blocks on its very next step. This is "thrashing".

Part 1 — Homogeneous agents
  All agents share the same step count and token sizes.  We sweep concurrent
  session counts and cache sizes to map the thrashing boundary.

  Three phases:
    1. WARMUP: cache filling, hit rate rising as sessions reuse their own prefixes
    2. THRASHING: working set > cache capacity, sessions evict each other, hit rate
       drops sharply even as cache stays full
    3. SETTLING: sessions complete and leave the cache, pressure decreases, hit rate
       may recover

Part 2 — Heterogeneous agents  (configurable number of agents, variable step
  counts and step-token sizes)
  Real deployments mix multiple agent archetypes:
    • "short"    : quick lookup agents, few steps, small tool results
    • "medium"   : standard reasoning agents
    • "long"     : deep-research / multi-tool agents, many steps, large outputs
    • "high_var" : agents with unpredictable tool results (web search, code exec)

  We model the interplay between:
    - different working-set sizes per agent type
    - different session lifetimes (short agents cycle fast → higher churn)
    - different per-step token variance (bursty vs smooth insertions)
  and sweep over agent-mix fractions and cache sizes to characterize hit rate
  and cache usage.
"""

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
PLOT_DIR = os.path.join(REPO_ROOT, "report_figures", "kv_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from vidur.entities.prefix_cache_manager import PrefixCacheManager
from vidur.entities.request import Request

# ── Parameters ───────────────────────────────────────────────────────────────

BLOCK_SIZE = 16
SEED = 42
DECODE_TOKENS = 64  # not used by cache, but needed for Request

# Agent session structure
SYSTEM_PROMPT_TOKENS = 200       # shared across all sessions
TASK_PROMPT_TOKENS = 100         # unique per session (user's task description)
STEP_THOUGHT_TOKENS = 60         # agent's reasoning per step
STEP_TOOL_CALL_TOKENS = 30      # function call
STEP_TOOL_RESULT_TOKENS = 110   # tool output (can be large: API responses, code, etc.)
TOKENS_PER_STEP = STEP_THOUGHT_TOKENS + STEP_TOOL_CALL_TOKENS + STEP_TOOL_RESULT_TOKENS

STEPS_PER_SESSION = 12           # typical agentic loop: 8-15 steps
NUM_SESSIONS = 60                # total agent sessions to simulate

# Lifecycle phases: how many sessions to use for each phase
# Ramp-up:  sessions arrive one at a time until pool is full
# Sustained: completed sessions are replaced (pool stays full)
# Drain:    no replacements, pool empties as sessions complete
SUSTAINED_REPLACEMENTS = 30      # how many replacement sessions during sustained phase

# Cache sizes to sweep (in blocks)
# A single session at max length ≈ (200 + 100 + 12*200) / 16 ≈ 169 blocks
MAX_SESSION_TOKENS = SYSTEM_PROMPT_TOKENS + TASK_PROMPT_TOKENS + STEPS_PER_SESSION * TOKENS_PER_STEP
BLOCKS_PER_FULL_SESSION = math.ceil(MAX_SESSION_TOKENS / BLOCK_SIZE)

# Concurrent session counts to sweep
CONCURRENT_SESSIONS_SWEEP = [2, 4, 6, 8, 12]

# Cache sizes: fraction of what N_max concurrent full sessions would need
# At 8 concurrent sessions: 8 * 169 = 1352 blocks needed
# We test from generous (can hold all) to tight (thrashing guaranteed)
CACHE_BLOCKS_SWEEP = [200, 400, 600, 800, 1200]


# ── Token ID generation for agentic workloads ───────────────────────────────

_BASE_SYSTEM = 1_000_000
_BASE_TASK = 2_000_000
_BASE_STEP = 3_000_000


def _make_system_prompt_tokens() -> Tuple[int, ...]:
    """Shared system prompt — identical across all sessions."""
    return tuple(_BASE_SYSTEM + i for i in range(SYSTEM_PROMPT_TOKENS))


def _make_task_tokens(session_id: int) -> Tuple[int, ...]:
    """Unique task description per session."""
    base = _BASE_TASK + session_id * TASK_PROMPT_TOKENS
    return tuple(base + i for i in range(TASK_PROMPT_TOKENS))


def _make_step_tokens(session_id: int, step_idx: int, rng: random.Random) -> Tuple[int, ...]:
    """Tokens for one agent step (thought + tool call + tool result).

    Each step is unique, but deterministic given session_id and step_idx.
    Some randomness in length models variable tool output sizes.
    """
    base = _BASE_STEP + session_id * 100_000 + step_idx * 1_000

    # Vary tool result length: ±50% of nominal
    result_len = max(20, int(STEP_TOOL_RESULT_TOKENS * (0.5 + rng.random())))
    total = STEP_THOUGHT_TOKENS + STEP_TOOL_CALL_TOKENS + result_len

    return tuple(base + i for i in range(total))


def generate_agentic_requests(
    num_sessions: int,
    steps_per_session: int,
    concurrent_sessions: int,
    sustained_replacements: int = SUSTAINED_REPLACEMENTS,
    seed: int = SEED,
) -> List[Tuple[Request, Tuple[int, ...]]]:
    """Generate requests modeling a three-phase agentic inference lifecycle.

    Returns a list of (Request, token_ids) pairs in arrival order.

    The three phases model a realistic production traffic pattern:

    1. RAMP-UP: Sessions arrive one every few ticks until the pool reaches
       N concurrent. The cache warms up gradually — hit rate rises as each
       new session benefits from the shared system prompt already cached
       by earlier sessions.

    2. SUSTAINED LOAD: The pool stays full at N concurrent sessions.
       Completed sessions are immediately replaced. If the working set
       exceeds cache capacity, this phase is sustained thrashing — the
       system never recovers because new sessions keep entering.

    3. DRAIN: No more replacement sessions. Active sessions complete one
       by one. Cache pressure drops, evictions slow, hit rate recovers.
       The cache "cools down" as the working set shrinks below capacity.
    """
    rng = random.Random(seed)
    system_tokens = _make_system_prompt_tokens()

    # Pre-generate all session data
    session_contexts: Dict[int, List[Tuple[int, ...]]] = {}
    for sid in range(num_sessions):
        task_tokens = _make_task_tokens(sid)
        cumulative = system_tokens + task_tokens
        steps = [cumulative]  # step 0: just system + task

        for step_idx in range(steps_per_session):
            step_tokens = _make_step_tokens(sid, step_idx, rng)
            cumulative = cumulative + step_tokens
            steps.append(cumulative)

        session_contexts[sid] = steps

    requests: List[Tuple[Request, Tuple[int, ...]]] = []
    active: List[Tuple[int, int]] = []  # (session_id, current_step)
    next_session_id = 0
    replacements_remaining = sustained_replacements
    time_tick = 0.0

    def _take_session() -> Optional[int]:
        nonlocal next_session_id
        if next_session_id < num_sessions:
            sid = next_session_id
            next_session_id += 1
            return sid
        return None

    # ── Phase 1: RAMP-UP ─────────────────────────────────────────────────
    # Add one session every 2 ticks until pool is full.
    # Each session starts at step 0 (cold start).
    ramp_tick_interval = 2  # ticks between new session arrivals
    ticks_since_last_arrival = ramp_tick_interval  # trigger first arrival immediately

    while len(active) < concurrent_sessions:
        # Check if it's time to add a new session
        if ticks_since_last_arrival >= ramp_tick_interval:
            sid = _take_session()
            if sid is None:
                break
            active.append((sid, 0))
            ticks_since_last_arrival = 0

        # Advance all active sessions one step
        next_active = []
        for sid, step in active:
            token_ids = session_contexts[sid][step]
            req = Request(
                arrived_at=time_tick,
                num_prefill_tokens=len(token_ids),
                num_decode_tokens=DECODE_TOKENS,
            )
            requests.append((req, token_ids))
            time_tick += 0.05

            next_step = step + 1
            if next_step <= steps_per_session:
                next_active.append((sid, next_step))
            # During ramp-up, completed sessions are not replaced
            # (pool is still growing from new arrivals)

        active = next_active
        time_tick += 0.5
        ticks_since_last_arrival += 1

    # ── Phase 2: SUSTAINED LOAD ──────────────────────────────────────────
    # Pool is full. Completed sessions are immediately replaced.
    # This continues until we exhaust the replacement budget.
    while active and replacements_remaining >= 0:
        next_active = []
        for sid, step in active:
            token_ids = session_contexts[sid][step]
            req = Request(
                arrived_at=time_tick,
                num_prefill_tokens=len(token_ids),
                num_decode_tokens=DECODE_TOKENS,
            )
            requests.append((req, token_ids))
            time_tick += 0.05

            next_step = step + 1
            if next_step <= steps_per_session:
                next_active.append((sid, next_step))
            else:
                # Replace completed session
                if replacements_remaining > 0:
                    new_sid = _take_session()
                    if new_sid is not None:
                        next_active.append((new_sid, 0))
                        replacements_remaining -= 1
                    else:
                        replacements_remaining = -1  # no more sessions available
                else:
                    replacements_remaining = -1  # budget exhausted

        active = next_active
        time_tick += 0.5

    # ── Phase 3: DRAIN ───────────────────────────────────────────────────
    # No replacements. Sessions complete and leave. Pool shrinks to zero.
    while active:
        next_active = []
        for sid, step in active:
            token_ids = session_contexts[sid][step]
            req = Request(
                arrived_at=time_tick,
                num_prefill_tokens=len(token_ids),
                num_decode_tokens=DECODE_TOKENS,
            )
            requests.append((req, token_ids))
            time_tick += 0.05

            next_step = step + 1
            if next_step <= steps_per_session:
                next_active.append((sid, next_step))

        active = next_active
        time_tick += 0.5

    return requests


# ── Simulation ───────────────────────────────────────────────────────────────

@dataclass
class StepTrace:
    """Per-request trace for the thrashing experiment."""
    request_idx: int
    session_id: int
    step_in_session: int
    arrived_at: float
    num_prefill_tokens: int
    # cache results
    tokens_cached: int
    is_hit: bool
    token_hit_fraction: float
    # cumulative
    cum_hit_rate: float
    cum_token_hit_rate: float
    # cache state
    cached_blocks: int
    max_blocks: int
    utilization: float
    blocks_inserted: int
    blocks_evicted: int
    # working set estimate
    estimated_working_set_blocks: int


def simulate_thrashing(
    concurrent_sessions: int,
    cache_max_blocks: int,
    num_sessions: int = NUM_SESSIONS,
    steps_per_session: int = STEPS_PER_SESSION,
    sustained_replacements: int = SUSTAINED_REPLACEMENTS,
) -> List[StepTrace]:
    """Run a single thrashing simulation."""
    requests = generate_agentic_requests(
        num_sessions=num_sessions,
        steps_per_session=steps_per_session,
        concurrent_sessions=concurrent_sessions,
        sustained_replacements=sustained_replacements,
        seed=SEED,
    )

    cache = PrefixCacheManager(max_blocks=cache_max_blocks, block_size=BLOCK_SIZE)
    traces = []

    # Re-derive session assignments by replaying the three-phase logic
    active2: List[Tuple[int, int]] = []
    next2 = 0
    repl_remaining = sustained_replacements

    def _take2() -> Optional[int]:
        nonlocal next2
        if next2 < num_sessions:
            sid = next2
            next2 += 1
            return sid
        return None

    assignment_list: List[Tuple[int, int]] = []

    # Phase 1: ramp-up
    ramp_tick = 2
    ticks_since = ramp_tick
    while len(active2) < concurrent_sessions:
        if ticks_since >= ramp_tick:
            sid = _take2()
            if sid is None:
                break
            active2.append((sid, 0))
            ticks_since = 0
        next_active2 = []
        for sid, step in active2:
            assignment_list.append((sid, step))
            if step + 1 <= steps_per_session:
                next_active2.append((sid, step + 1))
        active2 = next_active2
        ticks_since += 1

    # Phase 2: sustained
    while active2 and repl_remaining >= 0:
        next_active2 = []
        for sid, step in active2:
            assignment_list.append((sid, step))
            if step + 1 <= steps_per_session:
                next_active2.append((sid, step + 1))
            else:
                if repl_remaining > 0:
                    new_sid = _take2()
                    if new_sid is not None:
                        next_active2.append((new_sid, 0))
                        repl_remaining -= 1
                    else:
                        repl_remaining = -1
                else:
                    repl_remaining = -1
        active2 = next_active2

    # Phase 3: drain
    while active2:
        next_active2 = []
        for sid, step in active2:
            assignment_list.append((sid, step))
            if step + 1 <= steps_per_session:
                next_active2.append((sid, step + 1))
        active2 = next_active2

    for i, ((req, token_ids), (sid, step)) in enumerate(zip(requests, assignment_list)):
        evictions_before = cache.stats.total_evictions
        blocks_evicted_before = cache.stats.total_blocks_evicted
        blocks_inserted_before = cache.stats.total_blocks_inserted

        # Lookup
        cached_tokens = cache.match_prefix(token_ids)
        is_hit = cached_tokens > 0
        if is_hit:
            req.apply_prefix_cache_hit(cached_tokens)

        # Insert on completion
        cache.on_request_complete(token_ids)

        stats = cache.stats
        hit_frac = cached_tokens / len(token_ids) if token_ids else 0.0

        # Estimate working set from actual active sessions at this point
        # Look at nearby requests (same tick) to get all active session states
        tick_start = max(0, i - (i % concurrent_sessions))
        tick_end = min(len(assignment_list), tick_start + concurrent_sessions)
        ws_blocks = 0
        for j in range(tick_start, tick_end):
            _, s = assignment_list[j]
            ctx = SYSTEM_PROMPT_TOKENS + TASK_PROMPT_TOKENS + s * TOKENS_PER_STEP
            ws_blocks += math.ceil(ctx / BLOCK_SIZE)

        traces.append(StepTrace(
            request_idx=i,
            session_id=sid,
            step_in_session=step,
            arrived_at=req.arrived_at,
            num_prefill_tokens=len(token_ids),
            tokens_cached=cached_tokens,
            is_hit=is_hit,
            token_hit_fraction=hit_frac,
            cum_hit_rate=stats.hit_rate,
            cum_token_hit_rate=stats.token_hit_rate,
            cached_blocks=cache.num_cached_blocks,
            max_blocks=cache_max_blocks,
            utilization=cache.cache_utilization,
            blocks_inserted=stats.total_blocks_inserted - blocks_inserted_before,
            blocks_evicted=stats.total_blocks_evicted - blocks_evicted_before,
            estimated_working_set_blocks=ws_blocks,
        ))

    return traces


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_thrashing_phases(traces: List[StepTrace], label: str, filename: str):
    """Plot the three-phase thrashing pattern for a single configuration."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    xs = [t.request_idx for t in traces]

    # ── Panel 1: Per-request token hit fraction ──────────────────────────
    ax = axes[0]
    colors = ['#4CAF50' if t.token_hit_fraction > 0.5 else
              '#FF9800' if t.token_hit_fraction > 0.1 else
              '#F44336' for t in traces]
    ax.bar(xs, [t.token_hit_fraction * 100 for t in traces],
           width=1.0, color=colors, alpha=0.7)

    # Windowed average
    WINDOW = 10
    windowed = []
    for i in range(len(traces)):
        start = max(0, i - WINDOW + 1)
        w = traces[start:i + 1]
        avg = sum(t.token_hit_fraction for t in w) / len(w) * 100
        windowed.append(avg)
    ax.plot(xs, windowed, color='black', linewidth=2, label=f'{WINDOW}-req moving avg')

    ax.set_ylabel("Token Hit Fraction (%)")
    ax.set_title(f"Cache Thrashing in Agentic Inference — {label}", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Cache utilization + working set ─────────────────────────
    ax = axes[1]
    ax.plot(xs, [t.cached_blocks for t in traces],
            color='#2196F3', linewidth=1.5, label='Cached blocks')
    ax.plot(xs, [t.estimated_working_set_blocks for t in traces],
            color='#F44336', linewidth=1.5, linestyle='--', label='Est. working set')
    ax.axhline(traces[0].max_blocks, color='black', linestyle=':',
               linewidth=1, label=f'Cache capacity ({traces[0].max_blocks})')
    ax.set_ylabel("Blocks")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    # Shade thrashing region (where working set > capacity)
    thrash_start = None
    for i, t in enumerate(traces):
        if t.estimated_working_set_blocks > t.max_blocks and thrash_start is None:
            thrash_start = t.request_idx
        elif t.estimated_working_set_blocks <= t.max_blocks and thrash_start is not None:
            ax.axvspan(thrash_start, t.request_idx, alpha=0.1, color='red')
            thrash_start = None
    if thrash_start is not None:
        ax.axvspan(thrash_start, xs[-1], alpha=0.1, color='red')

    # ── Panel 3: Blocks inserted vs evicted per request ──────────────────
    ax = axes[2]
    ax.bar(xs, [t.blocks_inserted for t in traces],
           width=1.0, color='#2196F3', alpha=0.7, label='Blocks inserted')
    ax.bar(xs, [-t.blocks_evicted for t in traces],
           width=1.0, color='#F44336', alpha=0.7, label='Blocks evicted')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel("Blocks")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 4: Cumulative hit rate ─────────────────────────────────────
    ax = axes[3]
    ax.plot(xs, [t.cum_hit_rate * 100 for t in traces],
            color='#2196F3', linewidth=1.5, label='Request hit rate')
    ax.plot(xs, [t.cum_token_hit_rate * 100 for t in traces],
            color='#4CAF50', linewidth=1.5, label='Token hit rate')
    ax.set_ylabel("Cumulative Hit Rate (%)")
    ax.set_xlabel("Request Index")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_thrashing_heatmap(results: Dict[Tuple[int, int], List[StepTrace]]):
    """Heatmap: steady-state token hit rate vs (concurrent sessions, cache size)."""
    concurrent_vals = sorted(set(k[0] for k in results.keys()))
    cache_vals = sorted(set(k[1] for k in results.keys()))

    # Build matrix: average token hit fraction during middle third of simulation
    matrix = []
    for cs in concurrent_vals:
        row = []
        for cb in cache_vals:
            traces = results.get((cs, cb), [])
            if not traces:
                row.append(0)
                continue
            n = len(traces)
            mid_start, mid_end = n // 4, 3 * n // 4
            mid = traces[mid_start:mid_end]
            avg = sum(t.token_hit_fraction for t in mid) / len(mid) * 100 if mid else 0
            row.append(avg)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100,
                   origin='lower')

    ax.set_xticks(range(len(cache_vals)))
    ax.set_xticklabels([str(c) for c in cache_vals])
    ax.set_yticks(range(len(concurrent_vals)))
    ax.set_yticklabels([str(c) for c in concurrent_vals])
    ax.set_xlabel("Cache Size (blocks)")
    ax.set_ylabel("Concurrent Agent Sessions")
    ax.set_title("Mid-Phase Token Hit Rate (%) — Thrashing Boundary Map",
                 fontsize=12, fontweight='bold')

    # Annotate cells
    for i, cs in enumerate(concurrent_vals):
        for j, cb in enumerate(cache_vals):
            val = matrix[i][j]
            color = 'white' if val < 30 or val > 70 else 'black'
            ax.text(j, i, f"{val:.0f}%", ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    # Draw thrashing boundary (where working set ≈ cache)
    for i, cs in enumerate(concurrent_vals):
        ws = cs * BLOCKS_PER_FULL_SESSION
        for j, cb in enumerate(cache_vals):
            if cb >= ws:
                ax.plot(j, i, 'ko', markersize=8, markerfacecolor='none', markeredgewidth=2)
                break

    fig.colorbar(im, ax=ax, label="Token Hit Rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "thrashing_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved thrashing_heatmap.png")


def plot_concurrent_sweep(results: Dict[Tuple[int, int], List[StepTrace]], cache_size: int):
    """Overlay windowed hit rate for different concurrent session counts at a fixed cache size."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    WINDOW = 15

    colors = plt.cm.coolwarm([i / (len(CONCURRENT_SESSIONS_SWEEP) - 1)
                              for i in range(len(CONCURRENT_SESSIONS_SWEEP))])

    for cs, color in zip(CONCURRENT_SESSIONS_SWEEP, colors):
        traces = results.get((cs, cache_size), [])
        if not traces:
            continue

        xs = [t.request_idx for t in traces]
        windowed = []
        for i in range(len(traces)):
            start = max(0, i - WINDOW + 1)
            w = traces[start:i + 1]
            windowed.append(sum(t.token_hit_fraction for t in w) / len(w) * 100)

        ax1.plot(xs, windowed, color=color, label=f'{cs} concurrent',
                 linewidth=1.5, alpha=0.85)
        ax2.plot(xs, [t.utilization * 100 for t in traces], color=color,
                 label=f'{cs} concurrent', linewidth=1.5, alpha=0.85)

    ax1.set_ylabel(f"Token Hit Rate (%, {WINDOW}-req window)")
    ax1.set_title(f"Agentic Thrashing: Concurrent Sessions Sweep (cache={cache_size} blocks)",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Cache Utilization (%)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "thrashing_concurrent_sweep.png"), dpi=150)
    plt.close(fig)
    print("  Saved thrashing_concurrent_sweep.png")


# ── Characterization ─────────────────────────────────────────────────────────

def print_thrashing_analysis(traces: List[StepTrace], label: str):
    """Analyze and print the three phases."""
    n = len(traces)
    if n == 0:
        return

    # Detect phases by working set vs capacity
    capacity = traces[0].max_blocks
    warmup_end = 0
    thrash_start = None
    thrash_end = None

    for i, t in enumerate(traces):
        if t.estimated_working_set_blocks <= capacity:
            if thrash_start is not None and thrash_end is None:
                thrash_end = i
        else:
            if thrash_start is None:
                thrash_start = i

    if thrash_start is None:
        thrash_start = n  # no thrashing
    if thrash_end is None:
        thrash_end = n

    # Phase stats
    phases = [
        ("Warmup", traces[:thrash_start]),
        ("Thrashing", traces[thrash_start:thrash_end]),
        ("Settling", traces[thrash_end:]),
    ]

    print(f"\n  ── {label} ──")
    print(f"  Total requests: {n}, Capacity: {capacity} blocks, "
          f"Full session: {BLOCKS_PER_FULL_SESSION} blocks")

    for phase_name, phase_traces in phases:
        if not phase_traces:
            print(f"  {phase_name:12s}: (empty)")
            continue

        avg_hit = sum(t.token_hit_fraction for t in phase_traces) / len(phase_traces)
        avg_evict = sum(t.blocks_evicted for t in phase_traces) / len(phase_traces)
        avg_insert = sum(t.blocks_inserted for t in phase_traces) / len(phase_traces)
        avg_util = sum(t.utilization for t in phase_traces) / len(phase_traces)

        print(f"  {phase_name:12s}: reqs {phase_traces[0].request_idx}-{phase_traces[-1].request_idx}, "
              f"hit={avg_hit:.1%}, util={avg_util:.0%}, "
              f"insert={avg_insert:.1f} blk/req, evict={avg_evict:.1f} blk/req")


# ════════════════════════════════════════════════════════════════════════════════
# PART 2 — HETEROGENEOUS AGENT TYPES
#
# Real deployments mix multiple agent archetypes with different step counts
# and tool-call token sizes:
#
#   • "short"    : quick lookup agents, 2-4 steps, ~80 tokens/step
#   • "medium"   : standard reasoning agents, ~12 steps, ~200 tokens/step
#   • "long"     : deep-research agents, ~24 steps, ~300 tokens/step
#   • "high_var" : agents with unpredictable tool results (web search, code
#                  execution), ~10 steps, 50-500 tokens/step
#
# When these types share the same cache, their working sets differ in size
# and growth rate.  Mixing them changes the thrashing dynamics because:
#
#   1. Long agents accumulate large contexts — more blocks at risk of eviction
#   2. High-var agents have bursty insertion patterns — spike evictions
#   3. Short agents cycle fast — high churn of unique tails, low prefix reuse
#   4. Cross-type evictions degrade hit rate for ALL types simultaneously
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class AgentTypeConfig:
    """Configuration for one class of agent in a mixed workload.

    Each type has a fixed number of steps per session and a distribution
    of tokens appended per step (modelling the think → tool_call → result
    loop with variable tool output sizes).
    """
    name: str
    num_steps: int                  # steps per session
    tokens_per_step_mean: int       # mean tokens appended per step
    tokens_per_step_cv: float = 0.3  # coeff. of variation (std/mean); 0 = fixed
    system_prompt_tokens: int = SYSTEM_PROMPT_TOKENS
    task_tokens: int = TASK_PROMPT_TOKENS

    @property
    def max_context_tokens(self) -> int:
        return (self.system_prompt_tokens + self.task_tokens
                + self.num_steps * self.tokens_per_step_mean)

    @property
    def max_context_blocks(self) -> int:
        return math.ceil(self.max_context_tokens / BLOCK_SIZE)


# Canonical archetypes
AGENT_SHORT = AgentTypeConfig(
    "short", num_steps=4, tokens_per_step_mean=80, tokens_per_step_cv=0.2)
AGENT_MEDIUM = AgentTypeConfig(
    "medium", num_steps=12, tokens_per_step_mean=200, tokens_per_step_cv=0.3)
AGENT_LONG = AgentTypeConfig(
    "long", num_steps=24, tokens_per_step_mean=300, tokens_per_step_cv=0.35)
AGENT_HIGH_VAR = AgentTypeConfig(
    "high_var", num_steps=10, tokens_per_step_mean=200, tokens_per_step_cv=0.9)

# Mix configurations to sweep: (name, [(AgentTypeConfig, fraction), ...])
HETERO_MIXES: List[Tuple[str, List[Tuple[AgentTypeConfig, float]]]] = [
    ("all_short",    [(AGENT_SHORT, 1.0)]),
    ("all_medium",   [(AGENT_MEDIUM, 1.0)]),
    ("all_long",     [(AGENT_LONG, 1.0)]),
    ("short+long",   [(AGENT_SHORT, 0.5), (AGENT_LONG, 0.5)]),
    ("high_var",     [(AGENT_HIGH_VAR, 1.0)]),
    ("mixed_3way",   [(AGENT_SHORT, 1/3), (AGENT_MEDIUM, 1/3), (AGENT_LONG, 1/3)]),
]

# Defaults for heterogeneous sweeps
HETERO_CONCURRENT = 8
HETERO_CACHE = 800
HETERO_NUM_SESSIONS = 120
HETERO_SUSTAINED = 50


@dataclass
class HeteroStepTrace:
    """Per-request trace including agent-type metadata."""
    request_idx: int
    session_id: int
    step_in_session: int
    agent_type: str
    agent_num_steps: int
    arrived_at: float
    num_prefill_tokens: int
    tokens_cached: int
    is_hit: bool
    token_hit_fraction: float
    cum_hit_rate: float
    cum_token_hit_rate: float
    cached_blocks: int
    max_blocks: int
    utilization: float
    blocks_inserted: int
    blocks_evicted: int
    estimated_working_set_blocks: int


# ── Token-ID generation for heterogeneous agents ─────────────────────────────

def _hetero_type_offset(name: str) -> int:
    """Deterministic offset from type name (avoids Python hash randomization)."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(name)) % 50_000


def _make_hetero_system_tokens(atype: AgentTypeConfig) -> Tuple[int, ...]:
    """Each agent type shares its own distinct system prompt.

    Within a type all sessions share the same system prompt (intra-type
    prefix sharing).  Across types, system prompts differ, modelling
    different agent roles (coding, search, planning, …).
    """
    offset = _hetero_type_offset(atype.name)
    return tuple(_BASE_SYSTEM + offset + i for i in range(atype.system_prompt_tokens))


def _make_hetero_session_contexts(
    session_types: List[AgentTypeConfig],
    rng: random.Random,
) -> Dict[int, List[Tuple[int, ...]]]:
    """Pre-generate full per-session context snapshots.

    Returns contexts[sid][step] = full token_ids tuple at that step.
    """
    contexts: Dict[int, List[Tuple[int, ...]]] = {}
    for sid, atype in enumerate(session_types):
        system_toks = _make_hetero_system_tokens(atype)
        task_base = _BASE_TASK + sid * 500
        task_toks = tuple(task_base + i for i in range(atype.task_tokens))

        cumulative = system_toks + task_toks
        steps_list: List[Tuple[int, ...]] = [cumulative]

        step_base = _BASE_STEP + sid * 100_000
        for step_idx in range(atype.num_steps):
            if atype.tokens_per_step_cv > 0:
                std = atype.tokens_per_step_mean * atype.tokens_per_step_cv
                n = max(BLOCK_SIZE, int(rng.gauss(atype.tokens_per_step_mean, std)))
            else:
                n = atype.tokens_per_step_mean
            step_toks = tuple(step_base + step_idx * 1_000 + i for i in range(n))
            cumulative = cumulative + step_toks
            steps_list.append(cumulative)

        contexts[sid] = steps_list
    return contexts


# ── Request generation ────────────────────────────────────────────────────────

def generate_hetero_requests(
    agent_configs: List[Tuple[AgentTypeConfig, float]],
    concurrent_sessions: int,
    num_sessions: int,
    sustained_replacements: int = HETERO_SUSTAINED,
    seed: int = SEED,
) -> Tuple[List[Tuple[Request, Tuple[int, ...]]], List[Tuple[int, int, str]]]:
    """Generate a heterogeneous three-phase agentic workload.

    Uses a *cold-start* model: all ``concurrent_sessions`` agents begin
    simultaneously at step 0.  This avoids the ramp-up problem where short
    agents complete before the pool can fill.

    Phases:
      1. SUSTAINED: pool stays full — completed sessions are replaced.
      2. DRAIN:     no replacements — active sessions complete and leave.

    Returns
    -------
    requests : list of (Request, token_ids)
    assignments : list of (session_id, step, agent_type_name)
    """
    rng = random.Random(seed)

    # Normalise fractions and assign a type to each session
    total_frac = sum(f for _, f in agent_configs)
    norm: List[Tuple[AgentTypeConfig, float]] = [
        (c, f / total_frac) for c, f in agent_configs]

    session_types: List[AgentTypeConfig] = []
    for _ in range(num_sessions):
        r = rng.random()
        cumsum = 0.0
        assigned = norm[-1][0]
        for atype, frac in norm:
            cumsum += frac
            if r < cumsum:
                assigned = atype
                break
        session_types.append(assigned)

    contexts = _make_hetero_session_contexts(session_types, rng)

    requests: List[Tuple[Request, Tuple[int, ...]]] = []
    assignments: List[Tuple[int, int, str]] = []
    next_sid = 0
    time_tick = 0.0

    def _take() -> Optional[int]:
        nonlocal next_sid
        if next_sid < num_sessions:
            sid = next_sid
            next_sid += 1
            return sid
        return None

    # Cold-start: fill pool immediately
    active: List[Tuple[int, int]] = []
    for _ in range(concurrent_sessions):
        sid = _take()
        if sid is None:
            break
        active.append((sid, 0))

    repl_remaining = sustained_replacements

    # Sustained phase (with replacement) + drain (no replacement)
    while active:
        next_active: List[Tuple[int, int]] = []
        for sid, step in active:
            atype = session_types[sid]
            token_ids = contexts[sid][step]
            req = Request(
                arrived_at=time_tick,
                num_prefill_tokens=len(token_ids),
                num_decode_tokens=DECODE_TOKENS,
            )
            requests.append((req, token_ids))
            assignments.append((sid, step, atype.name))
            time_tick += 0.05

            if step + 1 <= atype.num_steps:
                next_active.append((sid, step + 1))
            elif repl_remaining > 0:
                new_sid = _take()
                if new_sid is not None:
                    next_active.append((new_sid, 0))
                    repl_remaining -= 1

        active = next_active
        time_tick += 0.5

    return requests, assignments


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_hetero_thrashing(
    agent_configs: List[Tuple[AgentTypeConfig, float]],
    concurrent_sessions: int,
    cache_max_blocks: int,
    num_sessions: int = HETERO_NUM_SESSIONS,
    sustained_replacements: int = HETERO_SUSTAINED,
) -> List[HeteroStepTrace]:
    """Run one heterogeneous thrashing simulation."""
    requests, assignments = generate_hetero_requests(
        agent_configs=agent_configs,
        concurrent_sessions=concurrent_sessions,
        num_sessions=num_sessions,
        sustained_replacements=sustained_replacements,
    )
    type_map: Dict[str, AgentTypeConfig] = {c.name: c for c, _ in agent_configs}
    cache = PrefixCacheManager(max_blocks=cache_max_blocks, block_size=BLOCK_SIZE)
    traces: List[HeteroStepTrace] = []

    for i, ((req, token_ids), (sid, step, type_name)) in enumerate(
            zip(requests, assignments)):
        blk_evict_before = cache.stats.total_blocks_evicted
        blk_insert_before = cache.stats.total_blocks_inserted

        cached_tokens = cache.match_prefix(token_ids)
        is_hit = cached_tokens > 0
        if is_hit:
            req.apply_prefix_cache_hit(cached_tokens)
        cache.on_request_complete(token_ids)

        stats = cache.stats
        hit_frac = cached_tokens / len(token_ids) if token_ids else 0.0
        atype = type_map[type_name]

        # Working-set estimate: look at the surrounding tick's requests
        tick_start = max(0, i - (i % max(1, concurrent_sessions)))
        tick_end = min(len(assignments), tick_start + concurrent_sessions)
        ws_blocks = 0
        seen_sids: set = set()
        for j in range(tick_start, tick_end):
            s_id_j, s_j, tname_j = assignments[j]
            if s_id_j in seen_sids:
                continue
            seen_sids.add(s_id_j)
            at_j = type_map.get(tname_j)
            if at_j:
                ctx = (at_j.system_prompt_tokens + at_j.task_tokens
                       + s_j * at_j.tokens_per_step_mean)
                ws_blocks += math.ceil(ctx / BLOCK_SIZE)

        traces.append(HeteroStepTrace(
            request_idx=i,
            session_id=sid,
            step_in_session=step,
            agent_type=type_name,
            agent_num_steps=atype.num_steps,
            arrived_at=req.arrived_at,
            num_prefill_tokens=len(token_ids),
            tokens_cached=cached_tokens,
            is_hit=is_hit,
            token_hit_fraction=hit_frac,
            cum_hit_rate=stats.hit_rate,
            cum_token_hit_rate=stats.token_hit_rate,
            cached_blocks=cache.num_cached_blocks,
            max_blocks=cache_max_blocks,
            utilization=cache.cache_utilization,
            blocks_inserted=stats.total_blocks_inserted - blk_insert_before,
            blocks_evicted=stats.total_blocks_evicted - blk_evict_before,
            estimated_working_set_blocks=ws_blocks,
        ))

    return traces


# ── Plotting helpers (heterogeneous) ─────────────────────────────────────────

_TYPE_COLORS: Dict[str, str] = {
    "short":    "#4CAF50",   # green
    "medium":   "#2196F3",   # blue
    "long":     "#F44336",   # red
    "high_var": "#FF9800",   # orange
}


def _type_color(name: str) -> str:
    return _TYPE_COLORS.get(name, "#9E9E9E")


def plot_hetero_phases(
    traces: List[HeteroStepTrace],
    agent_configs: List[Tuple[AgentTypeConfig, float]],
    label: str,
    filename: str,
):
    """5-panel detail plot for a heterogeneous thrashing simulation.

    Panels:
      1. Per-request token hit fraction (coloured by agent type)
      2. Cache utilisation + estimated working set
      3. Blocks inserted / evicted per request
      4. Per-agent-type windowed hit rate
      5. Cumulative hit rates
    """
    fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)
    xs = [t.request_idx for t in traces]
    type_names = list(dict.fromkeys(c.name for c, _ in agent_configs))

    # ── Panel 1: per-request hit fraction, coloured by type ───────────────
    ax = axes[0]
    for t in traces:
        ax.bar(t.request_idx, t.token_hit_fraction * 100,
               width=1.0, color=_type_color(t.agent_type), alpha=0.55)
    WINDOW = 10
    for tname in type_names:
        tt = [t for t in traces if t.agent_type == tname]
        if not tt:
            continue
        windowed = []
        for i in range(len(tt)):
            start = max(0, i - WINDOW + 1)
            windowed.append(
                sum(t.token_hit_fraction for t in tt[start:i + 1])
                / (i - start + 1) * 100)
        ax.plot([t.request_idx for t in tt], windowed,
                color=_type_color(tname), linewidth=2,
                label=f"{tname} ({WINDOW}-req avg)")
    ax.set_ylabel("Token Hit Fraction (%)")
    ax.set_title(f"Heterogeneous Agent Thrashing — {label}",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: cache utilisation + working set ──────────────────────────
    ax = axes[1]
    ax.plot(xs, [t.cached_blocks for t in traces],
            color="#2196F3", linewidth=1.5, label="Cached blocks")
    ax.plot(xs, [t.estimated_working_set_blocks for t in traces],
            color="#F44336", linewidth=1.5, linestyle="--",
            label="Est. working set")
    ax.axhline(traces[0].max_blocks, color="black", linestyle=":",
               linewidth=1, label=f"Capacity ({traces[0].max_blocks} blks)")
    ax.set_ylabel("Blocks")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    # shade thrashing region
    ts = None
    for t in traces:
        if t.estimated_working_set_blocks > t.max_blocks and ts is None:
            ts = t.request_idx
        elif t.estimated_working_set_blocks <= t.max_blocks and ts is not None:
            ax.axvspan(ts, t.request_idx, alpha=0.08, color="red")
            ts = None
    if ts is not None:
        ax.axvspan(ts, xs[-1], alpha=0.08, color="red")

    # ── Panel 3: blocks inserted / evicted ────────────────────────────────
    ax = axes[2]
    ax.bar(xs, [t.blocks_inserted for t in traces],
           width=1.0, color="#2196F3", alpha=0.7, label="Inserted")
    ax.bar(xs, [-t.blocks_evicted for t in traces],
           width=1.0, color="#F44336", alpha=0.7, label="Evicted")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Blocks / request")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 4: per-type windowed hit rate ───────────────────────────────
    ax = axes[3]
    WINDOW2 = 15
    for tname in type_names:
        tt = [t for t in traces if t.agent_type == tname]
        if not tt:
            continue
        windowed = []
        for i in range(len(tt)):
            start = max(0, i - WINDOW2 + 1)
            windowed.append(
                sum(t.token_hit_fraction for t in tt[start:i + 1])
                / (i - start + 1) * 100)
        cfg = next(c for c, _ in agent_configs if c.name == tname)
        ax.plot([t.request_idx for t in tt], windowed,
                color=_type_color(tname), linewidth=2,
                label=f"{tname} ({cfg.num_steps} steps, "
                      f"~{cfg.tokens_per_step_mean} tok/step)")
    ax.set_ylabel(f"Per-type Hit Rate (%, {WINDOW2}-req window)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)

    # ── Panel 5: cumulative hit rates ─────────────────────────────────────
    ax = axes[4]
    ax.plot(xs, [t.cum_hit_rate * 100 for t in traces],
            color="#2196F3", linewidth=1.5, label="Request hit rate")
    ax.plot(xs, [t.cum_token_hit_rate * 100 for t in traces],
            color="#4CAF50", linewidth=1.5, label="Token hit rate")
    ax.set_ylabel("Cumulative Hit Rate (%)")
    ax.set_xlabel("Request Index")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_hetero_mix_heatmap(
    results: Dict[Tuple[str, int], List[HeteroStepTrace]],
    mix_names: List[str],
    cache_vals: List[int],
):
    """Heatmap: mid-phase token hit rate vs (agent mix, cache size)."""
    matrix: List[List[float]] = []
    for mix_name in mix_names:
        row: List[float] = []
        for cb in cache_vals:
            traces = results.get((mix_name, cb), [])
            if not traces:
                row.append(0.0)
                continue
            n = len(traces)
            mid = traces[n // 4: 3 * n // 4]
            avg = (sum(t.token_hit_fraction for t in mid) / len(mid) * 100
                   if mid else 0.0)
            row.append(avg)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=100, origin="lower")
    ax.set_xticks(range(len(cache_vals)))
    ax.set_xticklabels([str(c) for c in cache_vals])
    ax.set_yticks(range(len(mix_names)))
    ax.set_yticklabels(mix_names)
    ax.set_xlabel("Cache Size (blocks)")
    ax.set_ylabel("Agent Mix")
    ax.set_title(
        f"Mid-Phase Token Hit Rate (%) — Heterogeneous Agent Mixes "
        f"({HETERO_CONCURRENT} concurrent)",
        fontsize=12, fontweight="bold")

    for i in range(len(mix_names)):
        for j in range(len(cache_vals)):
            val = matrix[i][j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Token Hit Rate (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "hetero_mix_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved hetero_mix_heatmap.png")


def plot_step_count_impact(
    results: Dict[str, List[HeteroStepTrace]],
    cache_size: int,
):
    """Compare windowed hit rate and utilisation across different agent mixes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    WINDOW = 15

    n_mixes = max(1, len(results) - 1)
    colors = plt.cm.tab10([i / n_mixes for i in range(len(results))])

    for (mix_name, traces), color in zip(results.items(), colors):
        if not traces:
            continue
        xs = [t.request_idx for t in traces]
        windowed = []
        for i in range(len(traces)):
            start = max(0, i - WINDOW + 1)
            w = traces[start:i + 1]
            windowed.append(sum(t.token_hit_fraction for t in w) / len(w) * 100)
        ax1.plot(xs, windowed, color=color, label=mix_name,
                 linewidth=1.8, alpha=0.85)
        ax2.plot(xs, [t.utilization * 100 for t in traces],
                 color=color, label=mix_name, linewidth=1.5, alpha=0.85)

    ax1.set_ylabel(f"Token Hit Rate (%, {WINDOW}-req window)")
    ax1.set_title(
        f"Step Count & Duration Impact on Thrashing "
        f"(cache={cache_size} blks, {HETERO_CONCURRENT} concurrent)",
        fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.set_ylim(-2, 102)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Cache Utilization (%)")
    ax2.set_xlabel("Request Index")
    ax2.legend(loc="lower right", fontsize=8, ncol=2)
    ax2.set_ylim(-2, 102)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "hetero_step_count_sweep.png"), dpi=150)
    plt.close(fig)
    print("  Saved hetero_step_count_sweep.png")


def plot_per_type_breakdown(
    traces: List[HeteroStepTrace],
    agent_configs: List[Tuple[AgentTypeConfig, float]],
    label: str,
    filename: str,
):
    """Per-agent-type hit-rate distribution (box plot) + cache-pressure stacked
    bar chart.
    """
    type_names = list(dict.fromkeys(c.name for c, _ in agent_configs))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: box plot of per-request hit fraction by type
    ax = axes[0]
    data_by_type = {n: [] for n in type_names}
    for t in traces:
        if t.agent_type in data_by_type:
            data_by_type[t.agent_type].append(t.token_hit_fraction * 100)

    bdata = [data_by_type[n] for n in type_names]
    bp = ax.boxplot(bdata, tick_labels=type_names, patch_artist=True, notch=False)
    for patch, name in zip(bp["boxes"], type_names):
        patch.set_facecolor(_type_color(name))
        patch.set_alpha(0.7)
    ax.set_ylabel("Token Hit Fraction (%)")
    ax.set_title(f"Per-type Hit Distribution\n{label}", fontsize=10)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: stacked bar of blocks_inserted by type, binned over time
    ax = axes[1]
    n_bins = 20
    max_req = max(t.request_idx for t in traces) + 1
    bin_size = max(1, max_req // n_bins)

    bin_inserts: Dict[str, List[int]] = {n: [0] * n_bins for n in type_names}
    for t in traces:
        b = min(n_bins - 1, t.request_idx // bin_size)
        if t.agent_type in bin_inserts:
            bin_inserts[t.agent_type][b] += t.blocks_inserted

    bin_xs = list(range(n_bins))
    bottom = [0.0] * n_bins
    for name in type_names:
        vals = bin_inserts[name]
        ax.bar(bin_xs, vals, bottom=bottom, label=name,
               color=_type_color(name), alpha=0.75)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xlabel(f"Request bucket (~{bin_size} reqs each)")
    ax.set_ylabel("Total blocks inserted")
    ax.set_title(f"Cache Pressure by Agent Type\n{label}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def print_hetero_analysis(
    traces: List[HeteroStepTrace],
    agent_configs: List[Tuple[AgentTypeConfig, float]],
    label: str,
):
    """Print per-type cache metrics for a heterogeneous run."""
    type_names = list(dict.fromkeys(c.name for c, _ in agent_configs))
    n = len(traces)
    if n == 0:
        return

    print(f"\n  ── {label} ──")
    print(f"  Total requests: {n}, Capacity: {traces[0].max_blocks} blocks")

    # Overall
    overall_hit = sum(t.token_hit_fraction for t in traces) / n
    overall_util = sum(t.utilization for t in traces) / n
    print(f"  Overall: hit={overall_hit:.1%}, util={overall_util:.0%}")

    # Per-type
    for tname in type_names:
        tt = [t for t in traces if t.agent_type == tname]
        if not tt:
            continue
        cfg = next(c for c, _ in agent_configs if c.name == tname)
        avg_hit = sum(t.token_hit_fraction for t in tt) / len(tt)
        avg_insert = sum(t.blocks_inserted for t in tt) / len(tt)
        avg_evict = sum(t.blocks_evicted for t in tt) / len(tt)
        print(f"    {tname:12s} ({cfg.num_steps:2d} steps, "
              f"~{cfg.tokens_per_step_mean:3d} tok/step): "
              f"reqs={len(tt):4d}, hit={avg_hit:.1%}, "
              f"ins={avg_insert:.1f} blk/req, evict={avg_evict:.1f} blk/req")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 70)
    print("  KV CACHE THRASHING IN AGENTIC INFERENCE")
    print("=" * 70)
    print(f"\n  Agent session structure:")
    print(f"    System prompt:     {SYSTEM_PROMPT_TOKENS} tokens (shared)")
    print(f"    Task description:  {TASK_PROMPT_TOKENS} tokens (unique/session)")
    print(f"    Per step:          ~{TOKENS_PER_STEP} tokens (thought + tool call + result)")
    print(f"    Steps/session:     {STEPS_PER_SESSION}")
    print(f"    Max context:       {MAX_SESSION_TOKENS} tokens = {BLOCKS_PER_FULL_SESSION} blocks")
    print(f"    Total sessions:    {NUM_SESSIONS}")
    print(f"    Block size:        {BLOCK_SIZE} tokens")

    # ── 1. Detailed single-config analysis ───────────────────────────────
    print(f"\n{'─'*70}")
    print("  DETAILED ANALYSIS: 6 concurrent, 600-block cache (staggered)")
    print(f"{'─'*70}")

    traces_detail = simulate_thrashing(concurrent_sessions=6, cache_max_blocks=600)
    plot_thrashing_phases(traces_detail,
                         label="6 concurrent, 600 blocks (staggered arrivals)",
                         filename="thrashing_detail.png")
    print_thrashing_analysis(traces_detail, "6 concurrent, 600 blocks")

    # ── 2. Full sweep ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  SWEEP: concurrent sessions × cache size")
    print(f"{'─'*70}")

    all_results: Dict[Tuple[int, int], List[StepTrace]] = {}
    for cs in CONCURRENT_SESSIONS_SWEEP:
        for cb in CACHE_BLOCKS_SWEEP:
            print(f"  Simulating {cs} concurrent, {cb} blocks...")
            all_results[(cs, cb)] = simulate_thrashing(
                concurrent_sessions=cs, cache_max_blocks=cb)

    # ── 3. Plots ─────────────────────────────────────────────────────────
    print(f"\n  Generating plots...")

    plot_thrashing_heatmap(all_results)
    plot_concurrent_sweep(all_results, cache_size=600)

    # Detailed plots for extreme cases
    plot_thrashing_phases(
        all_results[(12, 400)],
        label="12 concurrent, 400 blocks (severe thrashing)",
        filename="thrashing_severe.png")

    plot_thrashing_phases(
        all_results[(2, 1200)],
        label="2 concurrent, 1200 blocks (no thrashing)",
        filename="thrashing_none.png")

    # ── 4. Full characterization ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  THRASHING CHARACTERIZATION")
    print(f"{'─'*70}")

    for cs in CONCURRENT_SESSIONS_SWEEP:
        for cb in [400, 800]:
            traces = all_results.get((cs, cb), [])
            if traces:
                ws = cs * BLOCKS_PER_FULL_SESSION
                ratio = ws / cb
                print_thrashing_analysis(
                    traces,
                    f"{cs} concurrent, {cb} blocks (ws/cache={ratio:.1f}x)")

    # ════════════════════════════════════════════════════════════════════════
    # PART 2: HETEROGENEOUS AGENT TYPES
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  PART 2 — HETEROGENEOUS AGENT TYPES")
    print(f"{'═'*70}")

    print("\n  Agent archetypes:")
    for atype in [AGENT_SHORT, AGENT_MEDIUM, AGENT_LONG, AGENT_HIGH_VAR]:
        print(f"    {atype.name:12s}: {atype.num_steps:2d} steps, "
              f"~{atype.tokens_per_step_mean:3d} tok/step "
              f"(CV={atype.tokens_per_step_cv:.1f}), "
              f"max_ctx={atype.max_context_tokens} tok "
              f"= {atype.max_context_blocks} blks")
    print(f"    Concurrent:  {HETERO_CONCURRENT}")
    print(f"    Sessions:    {HETERO_NUM_SESSIONS}")
    print(f"    Replacements:{HETERO_SUSTAINED}")

    # ── 5. Detailed: short + long mix (the most interesting case) ────────
    print(f"\n{'─'*70}")
    print(f"  DETAILED: 50% short + 50% long, "
          f"{HETERO_CONCURRENT} concurrent, {HETERO_CACHE} blocks")
    print(f"{'─'*70}")

    mixed_sl = [(AGENT_SHORT, 0.5), (AGENT_LONG, 0.5)]
    traces_mixed = simulate_hetero_thrashing(
        agent_configs=mixed_sl,
        concurrent_sessions=HETERO_CONCURRENT,
        cache_max_blocks=HETERO_CACHE,
    )
    plot_hetero_phases(
        traces_mixed, mixed_sl,
        label=f"50% short / 50% long, {HETERO_CONCURRENT} conc., "
              f"{HETERO_CACHE} blks",
        filename="hetero_detail_short_long.png")
    plot_per_type_breakdown(
        traces_mixed, mixed_sl,
        label=f"50% short / 50% long, {HETERO_CACHE} blocks",
        filename="hetero_type_breakdown.png")
    print_hetero_analysis(traces_mixed, mixed_sl,
                          "50% short + 50% long")

    # ── 6. Detailed: 3-way mix ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  DETAILED: 3-way mix (short/medium/long), "
          f"{HETERO_CONCURRENT} concurrent, {HETERO_CACHE} blocks")
    print(f"{'─'*70}")

    mixed_3w = [(AGENT_SHORT, 1/3), (AGENT_MEDIUM, 1/3), (AGENT_LONG, 1/3)]
    traces_3way = simulate_hetero_thrashing(
        agent_configs=mixed_3w,
        concurrent_sessions=HETERO_CONCURRENT,
        cache_max_blocks=HETERO_CACHE,
    )
    plot_hetero_phases(
        traces_3way, mixed_3w,
        label=f"3-way mix, {HETERO_CONCURRENT} conc., {HETERO_CACHE} blks",
        filename="hetero_detail_3way.png")
    plot_per_type_breakdown(
        traces_3way, mixed_3w,
        label=f"3-way mix, {HETERO_CACHE} blocks",
        filename="hetero_3way_breakdown.png")
    print_hetero_analysis(traces_3way, mixed_3w, "3-way mix")

    # ── 7. Sweep: all mixes × cache sizes ────────────────────────────────
    print(f"\n{'─'*70}")
    print("  SWEEP: agent mix × cache size")
    print(f"{'─'*70}")

    hetero_results: Dict[Tuple[str, int], List[HeteroStepTrace]] = {}
    mix_names_ordered: List[str] = []
    for mix_name, mix_cfgs in HETERO_MIXES:
        mix_names_ordered.append(mix_name)
        for cb in CACHE_BLOCKS_SWEEP:
            print(f"  {mix_name:14s} × {cb:5d} blocks...")
            hetero_results[(mix_name, cb)] = simulate_hetero_thrashing(
                agent_configs=mix_cfgs,
                concurrent_sessions=HETERO_CONCURRENT,
                cache_max_blocks=cb,
            )

    plot_hetero_mix_heatmap(hetero_results, mix_names_ordered, CACHE_BLOCKS_SWEEP)

    # ── 8. Step-count impact overlay ─────────────────────────────────────
    step_results: Dict[str, List[HeteroStepTrace]] = {
        mn: hetero_results[(mn, HETERO_CACHE)]
        for mn, _ in HETERO_MIXES
        if (mn, HETERO_CACHE) in hetero_results
    }
    plot_step_count_impact(step_results, cache_size=HETERO_CACHE)

    # ── 9. Detailed: high-variance agents ────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  DETAILED: high-variance agents, "
          f"{HETERO_CONCURRENT} concurrent, {HETERO_CACHE} blocks")
    print(f"{'─'*70}")

    hv_cfgs: List[Tuple[AgentTypeConfig, float]] = [(AGENT_HIGH_VAR, 1.0)]
    traces_hv = simulate_hetero_thrashing(
        agent_configs=hv_cfgs,
        concurrent_sessions=HETERO_CONCURRENT,
        cache_max_blocks=HETERO_CACHE,
    )
    plot_hetero_phases(
        traces_hv, hv_cfgs,
        label=f"High-variance (CV={AGENT_HIGH_VAR.tokens_per_step_cv}), "
              f"{HETERO_CACHE} blks",
        filename="hetero_high_var.png")
    print_hetero_analysis(traces_hv, hv_cfgs, "All high-variance")

    # ── 10. Characterisation: per-type hit rates across all mixes ────────
    print(f"\n{'─'*70}")
    print("  HETEROGENEOUS CHARACTERIZATION")
    print(f"{'─'*70}")

    for mix_name, mix_cfgs in HETERO_MIXES:
        for cb in [400, 800]:
            traces = hetero_results.get((mix_name, cb), [])
            if traces:
                print_hetero_analysis(
                    traces, mix_cfgs,
                    f"{mix_name}, {cb} blocks")


if __name__ == "__main__":
    main()
