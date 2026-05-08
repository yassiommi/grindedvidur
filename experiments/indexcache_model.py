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
    """Per-layer cost decomposed for pipeline scheduling.

    Streams (resource model):
      - IO bus  : carries idx_io and kv_io (sum, single channel).
      - Compute : block_a + idx_comp + block_c + moe.

    Within-layer overlap: block_a (Q projection compute) runs concurrent
    with block_b (the IO + idx_comp + topk on the IO bus). Block_c
    cannot start until block_b's KV gather has landed. So per-layer:
        within_layer_time = max(block_a, block_b_io) + block_c + moe
    For pipeline scheduling we split this into:
        T_io       = idx_io + kv_io        (IO bus time, the part that
                                              must complete before block_c)
        T_compute  = max(0, block_a - T_io) + block_c + moe + idx_comp
                     (compute work; block_a is freebie if shorter than IO)
    """
    kind: str           # "F" or "S"
    idx_io_ms: float    # indexer-K read (0 for S layers)
    kv_io_ms: float     # MLA-KV gather (always > 0)
    idx_comp_ms: float  # indexer matmul + topk (0 for S layers)
    block_a_ms: float   # pre-norm + q_down + q_up + rope (compute, parallel with IO)
    block_c_ms: float   # kv_up_proj + attn core + o_proj + residual
    moe_ms: float       # MoE block (sequential after attention)

    @property
    def total_io(self) -> float:
        """IDX and KV share the IO channel: their costs sum."""
        return self.idx_io_ms + self.kv_io_ms

    @property
    def block_a_exposed_ms(self) -> float:
        """block_a portion that compute pays even after overlapping with IO bus.

        If block_a < total_io, block_a runs entirely concurrent with the IO
        on its own compute units, so its compute cost is hidden in the IO
        latency: zero compute payment.
        If block_a > total_io, the part beyond total_io still has to be paid
        on the compute stream.
        """
        return max(0.0, self.block_a_ms - self.total_io)

    @property
    def total_compute(self) -> float:
        """Compute work counted against the compute stream.

        block_c + moe + idx_comp must follow the IO. block_a's exposed part
        (if any) is also counted; the rest of block_a is hidden in IO.
        """
        return (self.block_a_exposed_ms + self.idx_comp_ms
                + self.block_c_ms + self.moe_ms)

    def within_layer_time_ms(self) -> float:
        """Sequential within-layer time (no cross-layer prefetch).

            within_layer = max(block_a, block_b_io) + block_c + moe + idx_comp
        """
        return (max(self.block_a_ms, self.total_io)
                + self.idx_comp_ms + self.block_c_ms + self.moe_ms)


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

    # Components we reuse from the underlying timing model:
    idx_io_F = ref["indexer_read_ms"]
    idx_comp_F = 0.005 + max(0.005, 0.005 * batch_size)  # indexer matmul (5us floor) + topk
    block_a = ref["block_a_ms"]
    block_c = ref["block_c_ms"]
    moe = ref["moe_total_ms"]
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
                block_a_ms=block_a,
                block_c_ms=block_c,
                moe_ms=moe,
            ))
        else:
            costs.append(LayerCost(
                kind="S",
                idx_io_ms=0.0,
                kv_io_ms=kv_io,
                idx_comp_ms=0.0,
                block_a_ms=block_a,
                block_c_ms=block_c,
                moe_ms=moe,
            ))
    return costs


# ─────────────────────────────────────────────────────────────────────
# Pipeline schedulers
# ─────────────────────────────────────────────────────────────────────
def schedule_sequential(costs: List[LayerCost]) -> Dict[str, float]:
    """No cross-layer overlap. Each layer fully serial except block_a||block_b.

    Per-layer = max(block_a, total_io) + idx_comp + block_c + moe
    """
    total = sum(c.within_layer_time_ms() for c in costs)
    return {"total_ms": total}


def schedule_pipelined(costs: List[LayerCost]) -> Dict[str, float]:
    """Producer/consumer pipeline with deep prefetch buffer.

    Two streams (IO bus and compute) run independently. Layer i's compute
    cannot start until both:
      (a) layer i-1's compute has finished (sequential layer dependency), and
      (b) layer i's IO has completed on the IO bus (block_c needs the
          gathered KV; we conservatively also require idx_io done).

    The IO bus runs back-to-back across all layers — a layer's IO is
    issued as soon as the bus becomes free, NOT just one layer ahead.
    This matches a system with a deep prefetch queue (≥ a few F-layer
    buffers, e.g. ~100 MB at sl=200K BS=1, ~32 GB at sl=1M BS=64).

    Recursion:
        end_io_i      = sum_{j<=i} T_io_j               (cumulative IO)
        end_comp_{-1} = 0
        end_comp_i    = max(end_comp_{i-1}, end_io_i) + T_comp_i

    Final pipeline time = end_comp_{N-1}.
    """
    if not costs:
        return {"total_ms": 0.0, "io_hidden_ms": 0.0, "io_total_ms": 0.0,
                "compute_total_ms": 0.0, "io_exposed_ms": 0.0,
                "io_stalled_compute_ms": 0.0}

    cum_io = 0.0
    end_comp = 0.0
    io_stalled = 0.0   # ms compute spent waiting on IO
    compute_busy = 0.0
    io_busy = 0.0
    last_comp_end = 0.0
    for c in costs:
        cum_io += c.total_io
        io_busy += c.total_io
        # Compute can start at max(prev compute end, this layer's IO end)
        start = max(last_comp_end, cum_io)
        if start > last_comp_end and last_comp_end > 0:
            io_stalled += start - last_comp_end  # compute waited on IO
        elif cum_io > start and start == last_comp_end:
            pass  # IO already done before compute slot — IO was hidden
        last_comp_end = start + c.total_compute
        compute_busy += c.total_compute

    end_comp = last_comp_end

    io_total = cum_io
    compute_total = compute_busy
    # Identity: end_comp = io_stalled_or_cold + compute_total
    # The "exposed" IO is exactly the time compute spent stalled (incl cold).
    # Cold start: layer 0's compute can't begin before layer 0's IO completes.
    cold_start = costs[0].total_io if costs else 0.0
    io_stalled_total = end_comp - compute_total
    overlapped = io_total - io_stalled_total

    return {
        "total_ms": end_comp,
        "io_total_ms": io_total,
        "compute_total_ms": compute_total,
        "io_hidden_ms": overlapped,
        "io_exposed_ms": io_stalled_total,    # time compute waited on IO
        "io_stalled_compute_ms": io_stalled_total,
        "cold_start_ms": cold_start,
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
