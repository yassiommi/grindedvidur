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

This experiment demonstrates three phases:
  1. WARMUP: cache filling, hit rate rising as sessions reuse their own prefixes
  2. THRASHING: working set > cache capacity, sessions evict each other, hit rate
     drops sharply even as cache stays full
  3. SETTLING: sessions complete and leave the cache, pressure decreases, hit rate
     may recover

We sweep over concurrent session counts and cache sizes to map the thrashing
boundary and characterize the I/O pattern.
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


if __name__ == "__main__":
    main()
