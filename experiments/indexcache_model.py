"""IndexCache simulation for DeepSeek Sparse Attention (DSA).

Background
----------
DSA inserts a *lightning indexer* at every transformer layer. The indexer
selects the top-k=2048 historical tokens that the layer's attention will
look at. Empirically, the top-k selections at adjacent layers overlap by
70-100%, so most indexer work is redundant.

IndexCache (Hu et al., arXiv:2603.12201) splits the N layers into:
    - F (Full)  layers: run the indexer, materialise a fresh top-k.
    - S (Shared) layers: reuse the most recent F layer's top-k set.

With pattern F:S:S:S (1 F per 4 layers), 75% of indexer compute *and*
75% of the indexer-K cache reads disappear.

Cost model
----------
Per layer, decompose the work into three concurrent streams:

  Stream IDX  - indexer K cache load (PCIe in offload, HBM in HBM mode)
                + the small indexer matmul + top-k. F layers only.

  Stream KV   - fetched MLA KV load for the 2560 attended tokens.
                F and S layers both pay this. (Sliding window + the top-k
                from this layer for F, or the inherited top-k for S.)

  Stream COMP - the rest of the layer: pre-norm, q_down, q_up, rope,
                kv_up_proj, attn core, o_proj, residual, MoE block.

The *per-layer* time (no pipelining) is:

    layer_time = max(block_a, block_b) + block_c + moe

For S layers, block_b drops to (kv_load + tiny topk-bookkeeping), which
is normally well below block_a — i.e. block_b vanishes from the critical
path.

The pipelined time across N layers exploits cross-layer prefetch: while
layer i is computing, the IO needed by layer i+1 (indexer-K and KV gather)
is being read into HBM on a separate stream. Steady-state cost per layer
becomes max(layer_compute, layer_io_next), modulo edge effects.

IDX and KV share one PCIe link in offload (and one HBM bus in HBM mode),
so they serialise on the IO channel. Per-layer IO = idx_read + kv_read.

We compare three modes:
  - "dsa"        : every layer is F. (Baseline.)
  - "indexcache" : layers patterned as F:S:S:S (configurable F_ratio).
  - "hbm"/"offload" : where the indexer K + KV physically live.

Care taken (lessons learned)
----------------------------
1. The indexer K read does NOT scale with batch_size only — it scales as
   seq_len * batch_size, because each sequence has its own indexer cache.
   This already comes from the underlying timing model.
2. When prefetching across layers, the *first* layer in the chain has no
   prior compute to hide its IO behind, so the simulator computes a
   tail-corrected total: io_first + sum_{i>0} max(compute_{i-1}, io_i)
   + compute_last. We do not silently amortise away the cold start.
3. For S layers, indexer_compute and topk are both zero — they're
   inherited. The KV gather still happens (the actual top-k tokens were
   chosen at the last F layer; this layer's KV is still loaded for them).
4. IDX and KV share the PCIe link, so we sum their costs per layer
   (no parallel IO engine).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from experiments.dsa_layer_analyzer import analyze_layer_bs
from experiments.dsa_timing_model import (
    DSA_ATTENDED,
    INDEXER_K_BYTES_PER_TOKEN,
    MLA_KV_BYTES_PER_TOKEN,
    NUM_LAYERS,
    ProfileTables,
    io_read_ms,
    load_profiles,
)


# ─────────────────────────────────────────────────────────────────────
# Layer-level cost record
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LayerCost:
    """Per-layer cost decomposed for pipeline scheduling."""
    kind: str           # "F" or "S"
    idx_io_ms: float    # indexer-K read (0 for S layers)
    kv_io_ms: float     # MLA-KV gather (always > 0)
    idx_comp_ms: float  # indexer matmul + topk (0 for S layers)
    other_compute_ms: float  # block_a + block_c + moe + small bookkeeping

    @property
    def total_io(self) -> float:
        """IDX and KV share the IO channel: their costs sum."""
        return self.idx_io_ms + self.kv_io_ms

    @property
    def total_compute(self) -> float:
        return self.idx_comp_ms + self.other_compute_ms

    def sequential_layer_ms(self) -> float:
        """No cross-layer overlap. Within-layer block A||B as in baseline."""
        # block_b = idx_io + idx_comp + kv_io  (single IO channel)
        block_b = self.idx_io_ms + self.idx_comp_ms + self.kv_io_ms
        # block_a + block_c + moe lumped into other_compute_ms; we cannot
        # split block_a here, so we approximate: within-layer overlap is
        # already captured at the underlying analyze_layer_bs level for the
        # F case, which is what we use as the reference. For the S case we
        # rebuild it: A||(kv_io) + C + MoE.
        # This method is informational only; the pipelined methods below are
        # what we actually report.
        return self.idx_io_ms + self.idx_comp_ms + self.kv_io_ms + self.other_compute_ms


def _f_pattern(n_layers: int, f_period: int) -> List[str]:
    """First layer is F, then every f_period-th layer is F, rest are S.

    f_period=1 → all F (baseline DSA). f_period=4 → F:S:S:S.
    """
    return ["F" if (i % f_period == 0) else "S" for i in range(n_layers)]


def build_layer_costs(
    seq_len: int,
    batch_size: int,
    mode: str,
    f_period: int,
    tables: ProfileTables,
    n_layers: int = NUM_LAYERS,
) -> List[LayerCost]:
    """Construct per-layer cost records for a given (seq_len, bs, mode, F-period).

    We re-use the existing analyze_layer_bs to source compute and IO times,
    but explicitly null out indexer-K read and indexer compute on S layers.
    """
    # Reference numbers from the established model — these are what F layers cost.
    ref = analyze_layer_bs(seq_len, mode, batch_size, tables)

    # Components we will reuse:
    idx_io_F = ref["indexer_read_ms"]
    idx_comp_F = 0.005 + max(0.005, 0.005 * batch_size)  # indexer_compute + topk floor
    # The 0.005 + ... above mirrors analyze_layer_bs' indexer_compute_ms (5us
    # kernel floor) + topk_ms (~bs us scaling, floored at 5us). We round
    # up to the same floors used in the source model.
    # block_a + block_c + moe: this is "other_compute" on the critical path.
    other_compute = ref["block_a_ms"] + ref["block_c_ms"] + ref["moe_total_ms"]
    kv_io = ref["fetch_kv_ms"]

    pattern = _f_pattern(n_layers, f_period)
    costs: List[LayerCost] = []
    for kind in pattern:
        if kind == "F":
            costs.append(LayerCost(
                kind="F",
                idx_io_ms=idx_io_F,
                kv_io_ms=kv_io,
                idx_comp_ms=idx_comp_F,
                other_compute_ms=other_compute,
            ))
        else:
            costs.append(LayerCost(
                kind="S",
                idx_io_ms=0.0,
                kv_io_ms=kv_io,
                idx_comp_ms=0.0,
                other_compute_ms=other_compute,
            ))
    return costs


# ─────────────────────────────────────────────────────────────────────
# Pipeline schedulers
# ─────────────────────────────────────────────────────────────────────
def schedule_sequential(costs: List[LayerCost]) -> Dict[str, float]:
    """No cross-layer overlap, but within-layer block_a || block_b.

    block_b = idx_io + idx_comp + kv_io   (single IO channel)
    block_a is part of other_compute_ms — but block_a is small (~0.08ms)
    relative to other_compute, and overlaps with block_b. To stay
    conservative and not double-credit, we treat the pipeline as
        layer = max(block_b, 0) + other_compute
    where block_a's overlap with block_b is captured implicitly in the
    underlying ref's block_a_ms (which we keep inside other_compute). For
    long seq_len / big bs, block_b dwarfs block_a anyway and this matters
    by <5%.
    """
    total = 0.0
    for c in costs:
        block_b = c.idx_io_ms + c.idx_comp_ms + c.kv_io_ms
        total += block_b + c.other_compute_ms
    return {"total_ms": total}


def schedule_pipelined(costs: List[LayerCost]) -> Dict[str, float]:
    """Cross-layer prefetch: layer i+1's IO runs while layer i computes.

    Steady-state per-layer cost = max(compute_i, io_{i+1}).
    Cold start: layer 0's IO must complete before layer 0's compute begins
    (no preceding compute to hide it behind).
    Tail: layer N-1's compute still has to finish; no further IO is needed.

    Schedule:
        t = io_0                       # cold prefetch of layer 0
        for i in 0..N-1:
            t += max(compute_i, io_{i+1})    # io_N = 0 by convention
    """
    if not costs:
        return {"total_ms": 0.0, "io_hidden_ms": 0.0, "io_total_ms": 0.0,
                "compute_total_ms": 0.0}

    io_of = lambda c: c.total_io

    io_total = sum(io_of(c) for c in costs)
    compute_total = sum(c.total_compute for c in costs)

    t = io_of(costs[0])  # cold prefetch of layer 0 — fully exposed
    overlapped = 0.0
    for i in range(len(costs)):
        comp_i = costs[i].total_compute
        io_next = io_of(costs[i + 1]) if i + 1 < len(costs) else 0.0
        step = max(comp_i, io_next)
        if io_next > 0:
            overlapped += min(comp_i, io_next)
        t += step
    return {
        "total_ms": t,
        "io_total_ms": io_total,
        "compute_total_ms": compute_total,
        "io_hidden_ms": overlapped,
        "io_exposed_ms": io_total - overlapped,
    }


# ─────────────────────────────────────────────────────────────────────
# Top-level scenario runner
# ─────────────────────────────────────────────────────────────────────
def simulate(
    seq_len: int,
    batch_size: int,
    mode: str,
    f_period: int,
    n_layers: int = NUM_LAYERS,
    tables: ProfileTables = None,
) -> Dict[str, float]:
    """One scenario: returns timing dict for sequential and pipelined runs."""
    if tables is None:
        tables = load_profiles()
    costs = build_layer_costs(seq_len, batch_size, mode, f_period, tables, n_layers)

    seq = schedule_sequential(costs)
    pipe = schedule_pipelined(costs)

    n_F = sum(1 for c in costs if c.kind == "F")
    n_S = sum(1 for c in costs if c.kind == "S")

    # Reference: every-layer-F (DSA baseline)
    ref_costs = build_layer_costs(seq_len, batch_size, mode, f_period=1,
                                  tables=tables, n_layers=n_layers)
    ref_seq = schedule_sequential(ref_costs)
    ref_pipe = schedule_pipelined(ref_costs)

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "mode": mode,
        "f_period": f_period,
        "n_F": n_F,
        "n_S": n_S,
        "indexer_savings_pct": 100.0 * n_S / n_layers,
        # IndexCache numbers
        "ic_seq_total_ms": seq["total_ms"],
        "ic_pipe_total_ms": pipe["total_ms"],
        "ic_io_total_ms": pipe["io_total_ms"],
        "ic_compute_total_ms": pipe["compute_total_ms"],
        "ic_io_hidden_ms": pipe["io_hidden_ms"],
        "ic_io_exposed_ms": pipe["io_exposed_ms"],
        # Baseline DSA (all-F) for comparison
        "dsa_seq_total_ms": ref_seq["total_ms"],
        "dsa_pipe_total_ms": ref_pipe["total_ms"],
        "dsa_io_total_ms": ref_pipe["io_total_ms"],
        "dsa_compute_total_ms": ref_pipe["compute_total_ms"],
        "dsa_io_hidden_ms": ref_pipe["io_hidden_ms"],
        "dsa_io_exposed_ms": ref_pipe["io_exposed_ms"],
        # Speedup
        "speedup_seq": ref_seq["total_ms"] / seq["total_ms"] if seq["total_ms"] else 0.0,
        "speedup_pipe": ref_pipe["total_ms"] / pipe["total_ms"] if pipe["total_ms"] else 0.0,
        # IO fully hidden in pipelined IndexCache?
        "ic_io_fully_hidden": pipe["io_exposed_ms"] <= 1e-6,
        "dsa_io_fully_hidden": ref_pipe["io_exposed_ms"] <= 1e-6,
    }


def sweep(
    seq_lens: List[int],
    batch_sizes: List[int],
    modes: Tuple[str, ...] = ("hbm", "offload"),
    f_periods: Tuple[int, ...] = (1, 2, 4, 8),
) -> List[Dict[str, float]]:
    tables = load_profiles()
    rows = []
    for mode in modes:
        for sl in seq_lens:
            for bs in batch_sizes:
                for fp in f_periods:
                    rows.append(simulate(sl, bs, mode, fp, tables=tables))
    return rows
