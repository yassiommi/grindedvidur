#!/usr/bin/env python3
"""IndexCache validation: single request, 200K seq_len, BS=1.

Walks one decode step end-to-end with the producer/consumer pipeline
model (deep prefetch buffer; IO bus runs back-to-back, compute waits
when cumulative IO falls behind).

Prints:
  1. Variable values used (sizes, bandwidths, per-token costs)
  2. Per-layer cost decomposition (idx_io, kv_io, block_a/c, MoE)
  3. Cumulative pipeline timeline (end_io_i, end_comp_i)
  4. End-to-end DSA vs IndexCache time comparisons across modes/F-periods
  5. Identity checks:
        end_comp_N-1  == compute_total + io_stalled_compute  (within fp tol)
        sum(layer IO) == io_total
        sum(compute)  == compute_total
        manual pipeline == simulate() output

Run:
    python -m experiments.validate_indexcache_200k
"""

from __future__ import annotations

from experiments.indexcache_model import (
    build_layer_costs,
    schedule_pipelined,
    simulate,
)
from experiments.dsa_timing_model import (
    DSA_ATTENDED,
    DSA_SELECTED_TOKENS,
    DSA_SLIDING_WINDOW,
    EP,
    H100_HBM_FLOOR_MS,
    H100_HBM_PEAK_GBS,
    H100_PCIE_FLOOR_MS,
    H100_PCIE_PEAK_GBS,
    HIDDEN_SIZE,
    INDEXER_K_BYTES_PER_TOKEN,
    KV_LORA_RANK,
    MLA_KV_BYTES_PER_TOKEN,
    NUM_HEADS,
    NUM_LAYERS,
    NUM_ROUTED_EXPERTS,
    Q_LORA_RANK,
    load_profiles,
)


SEQ_LEN = 200 * 1024  # 204800 tokens
BATCH_SIZE = 1


def _print_variables(seq_len: int, bs: int) -> None:
    print()
    print("=" * 92)
    print(" Variable values")
    print("=" * 92)
    print(f"  Hardware (H100 SXM):")
    print(f"    HBM peak BW         = {H100_HBM_PEAK_GBS:>8.1f} GB/s   "
          f"(floor {H100_HBM_FLOOR_MS:.3f} ms)")
    print(f"    PCIe Gen4 x16 BW    = {H100_PCIE_PEAK_GBS:>8.1f} GB/s   "
          f"(floor {H100_PCIE_FLOOR_MS:.3f} ms)")
    print()
    print(f"  Model (DeepSeek-V3):")
    print(f"    NUM_LAYERS          = {NUM_LAYERS}")
    print(f"    HIDDEN_SIZE         = {HIDDEN_SIZE}")
    print(f"    NUM_HEADS           = {NUM_HEADS}")
    print(f"    KV_LORA_RANK        = {KV_LORA_RANK}")
    print(f"    Q_LORA_RANK         = {Q_LORA_RANK}")
    print(f"    NUM_ROUTED_EXPERTS  = {NUM_ROUTED_EXPERTS}   (EP={EP})")
    print()
    print(f"  DSA / IndexCache:")
    print(f"    Top-k selected      = {DSA_SELECTED_TOKENS}")
    print(f"    Sliding window      = {DSA_SLIDING_WINDOW}")
    print(f"    Attended per layer  = {DSA_ATTENDED}")
    print(f"    Indexer K bytes/tok = {INDEXER_K_BYTES_PER_TOKEN} B   "
          f"(KV_LORA_RANK={KV_LORA_RANK} × FP8={1} B)")
    print(f"    MLA  KV bytes/tok   = {MLA_KV_BYTES_PER_TOKEN} B   "
          f"(KV_LORA_RANK + ROPE)·FP16")
    print()
    print(f"  Workload (this validation):")
    print(f"    seq_len             = {seq_len:>8,}  ({seq_len // 1024} K tokens)")
    print(f"    batch_size          = {bs}")
    print()
    # Derived sizes
    idx_K_per_layer = bs * seq_len * INDEXER_K_BYTES_PER_TOKEN
    kv_per_layer    = bs * seq_len * MLA_KV_BYTES_PER_TOKEN
    kv_gather_per_layer = bs * DSA_ATTENDED * MLA_KV_BYTES_PER_TOKEN
    print(f"  Derived sizes (bytes):")
    print(f"    Indexer K per layer            = {idx_K_per_layer:>13,}  "
          f"({idx_K_per_layer / (1024**2):>8.2f} MB)")
    print(f"    Indexer K all 61 F layers      = {61 * idx_K_per_layer:>13,}  "
          f"({61 * idx_K_per_layer / (1024**3):>8.2f} GB)")
    print(f"    Indexer K only 16 F (F:S:S:S)  = {16 * idx_K_per_layer:>13,}  "
          f"({16 * idx_K_per_layer / (1024**3):>8.2f} GB)")
    print(f"    Full MLA KV per layer          = {kv_per_layer:>13,}  "
          f"({kv_per_layer / (1024**2):>8.2f} MB)")
    print(f"    KV gather per layer ({DSA_ATTENDED} tok)  = {kv_gather_per_layer:>13,}  "
          f"({kv_gather_per_layer / (1024**2):>8.2f} MB)")


def _print_header(title: str) -> None:
    print()
    print("=" * 92)
    print(f" {title}")
    print("=" * 92)


def _print_layer_table(costs, max_rows=None):
    print(f"  {'i':>3}  {'kind':>4}  {'idx_io':>8}  {'kv_io':>8}  {'idx_cmp':>8}  "
          f"{'block_a':>8}  {'block_c':>8}  {'moe':>8}  {'T_io':>8}  {'T_cmp':>8}")
    print("  " + "-" * 86)
    rows = costs if max_rows is None else costs[:max_rows]
    for i, c in enumerate(rows):
        print(f"  {i:>3}  {c.kind:>4}  {c.idx_io_ms:>8.4f}  {c.kv_io_ms:>8.4f}  "
              f"{c.idx_comp_ms:>8.4f}  {c.block_a_ms:>8.4f}  {c.block_c_ms:>8.4f}  "
              f"{c.moe_ms:>8.4f}  {c.total_io:>8.4f}  {c.total_compute:>8.4f}")
    if max_rows is not None and len(costs) > max_rows:
        print(f"  ... [{len(costs) - max_rows} more layers omitted] ...")


def _walk_pipeline(costs, label: str) -> None:
    """Print the producer/consumer pipeline timeline.

    Two streams:
      end_io_i   = cumulative IO done after layer i's IO completes
      end_cmp_i  = max(end_cmp_{i-1}, end_io_i) + T_compute_i
    """
    print(f"\n  Pipeline walk ({label}) — producer/consumer (deep prefetch buffer):")
    print(f"  {'i':>3}  {'kind':>4}  {'T_io':>8}  {'T_cmp':>8}  "
          f"{'end_io':>9}  {'cmp_start':>10}  {'end_cmp':>9}  {'note':>14}")
    print("  " + "-" * 80)

    end_io = 0.0
    end_cmp = 0.0
    cum_compute = 0.0
    cum_stalled = 0.0

    for i, c in enumerate(costs):
        end_io += c.total_io
        cmp_start = max(end_cmp, end_io)
        stall = max(0.0, end_io - end_cmp)
        if i > 0 and stall > 0:
            cum_stalled += stall
        end_cmp = cmp_start + c.total_compute
        cum_compute += c.total_compute
        note = ("IO-bound" if cmp_start > end_cmp - c.total_compute - 1e-12
                and end_io > (end_cmp - c.total_compute - 1e-9)
                else "")
        # Simpler note: which constraint set cmp_start
        note = "IO-stall" if end_io > (end_cmp - c.total_compute) + 1e-9 else "compute"
        if i == 0:
            note = "cold-IO"
        if i < 6 or i >= len(costs) - 4 or c.kind == "F":
            print(f"  {i:>3}  {c.kind:>4}  {c.total_io:>8.4f}  {c.total_compute:>8.4f}  "
                  f"{end_io:>9.4f}  {cmp_start:>10.4f}  {end_cmp:>9.4f}  {note:>14}")
    cold = costs[0].total_io
    io_total = end_io
    print(f"\n  Σ compute        = {cum_compute:>10.4f} ms")
    print(f"  Σ IO (back-to-back) = {io_total:>10.4f} ms")
    print(f"  Cold IO (layer 0)   = {cold:>10.4f} ms")
    print(f"  IO stall after cold = {cum_stalled:>10.4f} ms")
    print(f"  Total IO exposed    = {(end_cmp - cum_compute):>10.4f} ms")
    print(f"  Pipeline total      = {end_cmp:>10.4f} ms")
    print(f"  Identity check: end_cmp == compute_total + io_exposed   "
          f"({end_cmp:.4f} == {cum_compute + (end_cmp - cum_compute):.4f})")


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
    _check("Σ layer IO == pipe.io_total_ms", io_total_summed, pipe["io_total_ms"])
    _check("Σ layer compute == pipe.compute_total_ms", cmp_total_summed, pipe["compute_total_ms"])
    _check("pipe_total == compute_total + io_exposed",
           pipe["total_ms"], pipe["compute_total_ms"] + pipe["io_exposed_ms"])
    _check("io_exposed >= cold_start (cold IO is always exposed)",
           min(pipe["io_exposed_ms"], pipe["cold_start_ms"]), pipe["cold_start_ms"])

    sim = simulate(SEQ_LEN, BATCH_SIZE, mode, f_period, tables=tables)
    _check("simulate() agrees with manual pipe.total_ms",
           sim["ic_pipe_total_ms"], pipe["total_ms"])

    # Headline numbers
    print(f"\n  Result: pipeline total = {pipe['total_ms']:.4f} ms"
          f"   ({pipe['total_ms'] / 1000:.4f} s per decode token)")


def _e2e_comparison(tables) -> None:
    """End-to-end DSA vs IndexCache time comparison across regimes."""
    _print_header("End-to-end DSA vs IndexCache time comparisons (per decode step)")
    print(f"  Pipeline model: producer/consumer with deep prefetch buffer.")
    print(f"  Each cell = ms per decode step (single H100 SXM, all 61 layers).\n")

    scenarios = [
        # (label, seq_len, bs)
        ("4K  / BS=1",   4 * 1024,         1),
        ("32K / BS=1",   32 * 1024,        1),
        ("128K / BS=1",  128 * 1024,       1),
        ("200K / BS=1",  200 * 1024,       1),
        ("512K / BS=1",  512 * 1024,       1),
        ("1M  / BS=1",   1024 * 1024,      1),
        ("32K / BS=16",  32 * 1024,       16),
        ("128K / BS=16", 128 * 1024,      16),
        ("200K / BS=16", 200 * 1024,      16),
        ("128K / BS=64", 128 * 1024,      64),
        ("200K / BS=64", 200 * 1024,      64),
    ]

    for mode in ("hbm", "offload"):
        print(f"  ── mode = {mode} " + "─" * (76 - len(mode)))
        print(f"  {'scenario':>14}  | {'DSA pipe':>9}  {'IC pipe':>9}  "
              f"{'speedup':>8}  | {'DSA io':>8}  {'IC io':>8}  "
              f"{'compute':>8}  {'IC stall':>9}")
        print("  " + "-" * 88)
        for label, sl, bs in scenarios:
            dsa = simulate(sl, bs, mode, f_period=1, tables=tables)
            ic = simulate(sl, bs, mode, f_period=4, tables=tables)
            print(f"  {label:>14}  | "
                  f"{dsa['ic_pipe_total_ms']:>9.2f}  "
                  f"{ic['ic_pipe_total_ms']:>9.2f}  "
                  f"{dsa['ic_pipe_total_ms']/ic['ic_pipe_total_ms']:>7.2f}x  | "
                  f"{dsa['ic_io_total_ms']:>8.2f}  "
                  f"{ic['ic_io_total_ms']:>8.2f}  "
                  f"{ic['ic_compute_total_ms']:>8.2f}  "
                  f"{ic['ic_io_exposed_ms']:>9.2f}")
        print()


def main() -> None:
    print(f"IndexCache validation — single request, seq_len = 200K ({SEQ_LEN:,} tokens), "
          f"BS = {BATCH_SIZE}")
    print(f"H100 SXM, DeepSeek-V3 ({NUM_LAYERS} layers)")

    tables = load_profiles()

    _print_variables(SEQ_LEN, BATCH_SIZE)

    for mode in ("hbm", "offload"):
        for fp in (1, 4):
            _validate_one(mode, fp, tables)

    _e2e_comparison(tables)


if __name__ == "__main__":
    main()
