#!/usr/bin/env python3
"""IndexCache validation: single request, 200K seq_len, BS=1.

Walks one decode step end-to-end and prints, for both DSA (all-F) and
IndexCache (F:S:S:S):
  1. Per-layer cost (idx_io, kv_io, idx_comp, other_compute)
  2. Cumulative pipeline timeline tick-by-tick
  3. Identity checks:
        sum(io)         == ic_io_total_ms
        sum(compute)    == ic_compute_total_ms
        pipe_total      == compute_total + io_exposed
        io_exposed      == io_first + sum max(0, io_{i+1} - comp_i)
  4. Cross-check vs the single-call simulate() output

Run:
    python -m experiments.validate_indexcache_200k
"""

from __future__ import annotations

from experiments.indexcache_model import (
    build_layer_costs,
    schedule_pipelined,
    simulate,
)
from experiments.dsa_timing_model import NUM_LAYERS, load_profiles


SEQ_LEN = 200 * 1024  # 204800 tokens
BATCH_SIZE = 1


def _print_header(title: str) -> None:
    print()
    print("=" * 92)
    print(f" {title}")
    print("=" * 92)


def _print_layer_table(costs, max_rows=None):
    print(f"  {'i':>3}  {'kind':>4}  {'idx_io':>9}  {'kv_io':>9}  {'idx_comp':>9}  "
          f"{'other_cmp':>10}  {'layer_io':>9}  {'layer_cmp':>10}")
    print("  " + "-" * 78)
    rows = costs if max_rows is None else costs[:max_rows]
    for i, c in enumerate(rows):
        print(f"  {i:>3}  {c.kind:>4}  {c.idx_io_ms:>9.4f}  {c.kv_io_ms:>9.4f}  "
              f"{c.idx_comp_ms:>9.4f}  {c.other_compute_ms:>10.4f}  "
              f"{c.total_io:>9.4f}  {c.total_compute:>10.4f}")
    if max_rows is not None and len(costs) > max_rows:
        print(f"  ... [{len(costs) - max_rows} more layers omitted] ...")


def _walk_pipeline(costs, label: str) -> None:
    """Print the pipeline timeline tick by tick."""
    print(f"\n  Pipeline walk ({label}):")
    print(f"  {'step':>20}  {'compute_i':>10}  {'io_next':>10}  {'tick':>10}  {'cum_t':>10}")
    print("  " + "-" * 70)

    t = costs[0].total_io  # cold prefetch
    cum_io = costs[0].total_io
    cum_compute = 0.0
    cum_overlap = 0.0
    print(f"  {'cold prefetch L0':>20}  {0.0:>10.4f}  {costs[0].total_io:>10.4f}  "
          f"{costs[0].total_io:>10.4f}  {t:>10.4f}")

    for i in range(len(costs)):
        comp_i = costs[i].total_compute
        io_next = costs[i + 1].total_io if i + 1 < len(costs) else 0.0
        tick = max(comp_i, io_next)
        t += tick
        cum_compute += comp_i
        cum_io += io_next
        cum_overlap += min(comp_i, io_next) if io_next > 0 else 0.0
        if i < 6 or i >= len(costs) - 4:
            label_i = f"compute L{i} || prefetch L{i+1}" if i + 1 < len(costs) else f"compute L{i} (last)"
            print(f"  {label_i:>20}  {comp_i:>10.4f}  {io_next:>10.4f}  "
                  f"{tick:>10.4f}  {t:>10.4f}")
        elif i == 6:
            print(f"  {'... [middle layers omitted] ...':>20}")
    print(f"\n  Σ compute        = {cum_compute:>10.4f} ms")
    print(f"  Σ io (incl cold) = {cum_io:>10.4f} ms")
    print(f"  Σ overlapped     = {cum_overlap:>10.4f} ms")
    print(f"  io_exposed       = {cum_io - cum_overlap:>10.4f} ms")
    print(f"  pipeline total   = {t:>10.4f} ms")


def _check(label: str, lhs: float, rhs: float, tol: float = 1e-6) -> None:
    ok = abs(lhs - rhs) < tol
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}]  {label}:  {lhs:.6f}  vs  {rhs:.6f}  (Δ = {abs(lhs - rhs):.2e})")


def _validate_one(mode: str, f_period: int, tables) -> None:
    label = "DSA (all-F)" if f_period == 1 else f"IndexCache (F:S^{f_period - 1})"
    _print_header(f"{label}  |  mode={mode}  |  seq_len=200K  |  BS=1")

    costs = build_layer_costs(SEQ_LEN, BATCH_SIZE, mode, f_period, tables)
    pipe = schedule_pipelined(costs)

    print(f"\n  N_layers = {len(costs)}   "
          f"F = {sum(1 for c in costs if c.kind == 'F')}   "
          f"S = {sum(1 for c in costs if c.kind == 'S')}")

    print("\n  Per-layer costs (first 8 + last 4):")
    _print_layer_table(costs, max_rows=8)

    _walk_pipeline(costs, label)

    print("\n  Identity checks:")
    io_total_summed = sum(c.total_io for c in costs)
    cmp_total_summed = sum(c.total_compute for c in costs)
    io_exposed_closed = costs[0].total_io + sum(
        max(0.0, (costs[i + 1].total_io if i + 1 < len(costs) else 0.0) - costs[i].total_compute)
        for i in range(len(costs))
    )

    _check("Σ layer IO == pipe.io_total_ms", io_total_summed, pipe["io_total_ms"])
    _check("Σ layer compute == pipe.compute_total_ms", cmp_total_summed, pipe["compute_total_ms"])
    _check("io_exposed == io_total - hidden",
           pipe["io_total_ms"] - pipe["io_hidden_ms"], pipe["io_exposed_ms"])
    _check("io_exposed == closed-form sum",
           pipe["io_exposed_ms"], io_exposed_closed)
    _check("pipe_total == compute_total + io_exposed",
           pipe["total_ms"], pipe["compute_total_ms"] + pipe["io_exposed_ms"])

    sim = simulate(SEQ_LEN, BATCH_SIZE, mode, f_period, tables=tables)
    _check("simulate() agrees with manual pipe.total_ms",
           sim["ic_pipe_total_ms"], pipe["total_ms"])

    # Headline numbers
    print(f"\n  Result: pipeline total = {pipe['total_ms']:.4f} ms"
          f"   ({pipe['total_ms'] / 1000:.4f} s per decode token)")


def main() -> None:
    print(f"IndexCache validation — single request, seq_len = 200K ({SEQ_LEN:,} tokens), "
          f"BS = {BATCH_SIZE}")
    print(f"H100 SXM, DeepSeek-V3 ({NUM_LAYERS} layers)")

    tables = load_profiles()

    for mode in ("hbm", "offload"):
        for fp in (1, 4):
            _validate_one(mode, fp, tables)

    # Final cross-mode summary
    _print_header("Cross-mode comparison @ 200K, BS=1")
    print(f"  {'mode':>8}  {'f_period':>8}  {'compute':>9}  {'io_total':>9}  "
          f"{'io_exposed':>11}  {'pipe_total':>11}  {'speedup':>8}")
    print("  " + "-" * 80)
    rows = []
    for mode in ("hbm", "offload"):
        for fp in (1, 4):
            r = simulate(SEQ_LEN, BATCH_SIZE, mode, fp, tables=tables)
            rows.append(r)
            print(f"  {mode:>8}  {fp:>8}  {r['ic_compute_total_ms']:>9.3f}  "
                  f"{r['ic_io_total_ms']:>9.3f}  {r['ic_io_exposed_ms']:>11.3f}  "
                  f"{r['ic_pipe_total_ms']:>11.3f}  {r['speedup_pipe']:>7.3f}x")
    print()


if __name__ == "__main__":
    main()
