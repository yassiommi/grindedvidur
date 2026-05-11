# Agentic Memory System (AMS) Integration Design

Status: design, not yet implemented.
Target branch: `claude/gpu-memory-integration-design-KSsP1`.

## 1. Overview

This document describes how an **Agentic Memory System (AMS)** — a SCADA-like
library that accelerates tiny, random, GPU-initiated I/O by aggregating peer
GPUs' PCIe lanes — would be modeled inside the vidur/InferLens simulator.

The integration introduces a generic `MemoryBackend` abstraction with two
implementations: a no-op `HbmOnlyBackend` (preserves today's behavior bit for
bit) and an `AmsBackend` that models aggregated peer-PCIe bandwidth, IOPS
limits, a small-I/O penalty curve, and per-peer contention. Four concrete
consumers can sit on top of the backend; the recommended first consumer is the
KV-cache offload tier.

## 2. Motivation

The simulator today reasons about data movement as a single PCIe number per
device and lumps KV-cache load + weight load onto one DMA stream in
`LayerExecutionTime`. This works for big bulk DMAs but misses two regimes that
matter for next-generation inference:

1. **Tiny random reads** (sub-page, KB-scale). Real PCIe cannot be saturated by
   bandwidth alone here — the IOPS ceiling dominates, and per-request doorbell
   overhead is non-trivial.
2. **Peer-PCIe aggregation**. When other GPUs are not saturating their PCIe
   lanes, an AMS-like substrate can fan out a single logical GPU's I/O across
   N peers, multiplying effective bandwidth.

Concrete questions we want to answer in simulation:

- How much KV cache *effectively* fits if cold blocks live behind AMS?
- What does MoE expert streaming cost at decode-token granularity?
- Does PDD's prefill-to-decode KV handoff benefit from aggregated peer PCIe?
- Is prefix-cache metadata cost a real concern at long contexts?

A simulator-side answer is far cheaper than a CUDA prototype and informs
hardware/library investment.

## 3. AMS background

Properties of a real AMS the model must honor:

| Property                | What it means in the cost model                          |
|-------------------------|----------------------------------------------------------|
| GPU-initiated           | No host round-trip; doorbell overhead instead            |
| Tiny random I/O         | Sub-page reads (64 B – 64 KB), high IOPS                 |
| Peer-PCIe aggregation   | Effective BW ≈ Σ `peer_pcie_bw[i] * (1 − util_i(t))`     |
| Tiered store            | CPU RAM and/or NVMe behind the fabric                    |
| Coalescing              | Many tiny ops in one tick amortize to one doorbell       |
| Pinning                 | Consumers can keep hot keys resident                     |

## 4. Where AMS fits in the simulator

```
            +-------------------- Consumers --------------------+
            |                                                   |
       (A) KV offload       (B) MoE expert       (C) PDD KV     (D) Prefix-cache
       attention/decode     streaming            transfer        metadata
            |                    |                  |                |
            v                    v                  v                v
       +---------------------------------------------------------------+
       |                       IoIssuer                                |
       |  - per-tick coalescing across consumers                       |
       |  - emits AmsIoEvent into the discrete-event queue             |
       |  - updates LayerExecutionTime.dma_ams stream end              |
       +---------------------------------------------------------------+
                                  |
                                  v
       +---------------------------------------------------------------+
       |                     MemoryBackend (ABC)                       |
       |  - residency(key) -> Tier                                     |
       |  - schedule_read / write / batch -> IoHandle                  |
       |  - reserve_peer(peer_id, t_start, t_end, bytes)               |
       |                                                               |
       |   HbmOnlyBackend (default, no-op)                             |
       |   AmsBackend     (full cost model)                            |
       +---------------------------------------------------------------+
```

Two principles drive the split:

- **Backend = residency + cost.** It does not touch the event queue.
- **IoIssuer = scheduling.** It funnels submissions, coalesces, emits events,
  updates the layer-time accountant. Multiple consumers share one issuer per
  replica so cross-consumer coalescing and BW reservation are coherent.

## 5. Integration options (consumers)

All four use the same backend; this section is about *who calls it*.

### Option A — KV cache offload tier (recommended first consumer)

- HBM holds hot KV blocks; cold blocks live in an AMS-fronted CPU/NVMe pool.
- Attention emits AMS reads for any non-resident block. Cost is overlapped
  with compute via the new `dma_ams` stream.
- Showcase: long-context decode where HBM is smaller than the working set.
  Compare TTFT, TBT, and throughput with and without AMS.
- Touches: `BaseReplicaScheduler` allocator (must become a real block map),
  `LayerExecutionTime`, attention KV-load path in the predictor.

### Option B — MoE expert weight streaming

- Replace bulk PCIe expert load (`per_layer_weight_load_times`,
  `moe_expert_load_time`) with per-token AMS fetches.
- Add `pin(key)` / `unpin(key)` so hot experts stay resident between tokens.
- Touches: `BaseExecutionTimePredictor` MoE path, expert-load configs.

### Option C — PDD KV transfer over aggregated peer PCIe

- `KvCacheTransferEvent` currently does `bytes / pcie_bw`. Route it through
  AMS to benefit from peer-PCIe aggregation, or collapse to a tier-flip no-op
  when the source blocks are already AMS-resident.
- Touches: `vidur/events/kv_cache_transfer_event.py`.

### Option D — Prefix-cache metadata lookup

- The radix-tree walk in `PrefixCacheManager` is free today. With AMS we can
  model the tiny random reads of a remote/peer-resident tree (lookup + node
  fetch).
- Mostly a research knob; small effect on TTFT but realistic.
- Touches: `vidur/entities/prefix_cache_manager.py`.

**Recommendation:** Option A first. It exercises the full backend (read, write,
residency, contention, coalescing) and produces the most compelling end-to-end
metric story. B, C, and D layer on without re-plumbing.

## 6. Prerequisite: upgrade the allocator

Today `BaseReplicaScheduler` is a *counter* of allocated blocks
(`_allocation_map`, `_num_allocated_blocks` at
`vidur/scheduler/replica_scheduler/base_replica_scheduler.py:40-160`). There
is no per-block identity, so "residency tier per block" is not expressible.

Promote it to a real block map keyed by `(request_id, logical_block_idx) ->
BlockResidency`, where:

```python
@dataclass
class BlockResidency:
    tier: Tier              # HBM | AMS_CPU | AMS_NVME | EVICTED
    phys_id: int            # physical slot in that tier
    refcount: int           # > 1 when prefix-cache shares the block
    pinned: bool
```

Refcounting matters: prefix-cache nodes are shared blocks. Demotion of a
shared block to AMS_CPU must update the radix node's residency annotation in
`prefix_cache_manager.py`, since "prefix hit" no longer implies "free read".

This allocator work is a prerequisite for every option and should be scheduled
first.

## 7. Core `MemoryBackend` API

New package `vidur/memory_backend/`.

```python
class Tier(Enum):
    HBM, AMS_CPU, AMS_NVME, EVICTED

@dataclass(frozen=True)
class BlockKey:
    replica_id: int
    request_id: int
    layer_idx: int
    logical_block_idx: int

@dataclass
class IoHandle:
    t_issue: float
    t_complete: float
    nbytes: int
    peer_path: list[int]    # peer GPUs whose PCIe was used

class MemoryBackend(ABC):
    def schedule_read(self, gpu_id, key, nbytes, t_now)  -> IoHandle: ...
    def schedule_write(self, gpu_id, key, nbytes, t_now) -> IoHandle: ...
    def schedule_batch(self, gpu_id, reqs, t_now)        -> list[IoHandle]: ...
    def residency(self, key)                              -> Tier: ...
    def pin(self, key) -> None: ...
    def unpin(self, key) -> None: ...
    def reserve_peer(self, peer_id, t_start, t_end, bytes_) -> None: ...
    def stats(self) -> BackendStats: ...
```

Implementations:

- `HbmOnlyBackend` — residency always `HBM`, zero cost, no-op for `reserve_peer`.
  Default for all existing configurations; baseline runs remain bit-identical.
- `AmsBackend` — full cost model (Section 8), per-peer BW interval set,
  coalescing buffer, capacity tracking.

Registration mirrors the existing pattern:

```python
class MemoryBackendRegistry(BaseRegistry):
    pass

MemoryBackendRegistry.register("hbm_only", HbmOnlyBackend)
MemoryBackendRegistry.register("ams",      AmsBackend)
```

The thin `IoIssuer` companion:

```python
class IoIssuer:
    def __init__(self, backend: MemoryBackend, event_queue, layer_time):
        ...
    def submit(self, consumer_id, key, nbytes, t_now) -> IoHandle:
        # coalesce within the current tick, return handle,
        # emit AmsIoEvent, advance layer_time.dma_ams_end
        ...
    def flush_tick(self, t_now): ...
```

## 8. Cost model

```
eff_bw(t) = sum_i ( peer_pcie_bw[i] * (1 - util_i(t)) ) * BW_EFFICIENCY     # 0.8
t_bw      = nbytes / eff_bw(t)
t_iops    = 1 / ( iops_cap * small_io_penalty(nbytes) )
t_complete = t_now + doorbell_overhead + max(t_bw, t_iops)
```

Notes:

- `small_io_penalty(n)` is a piecewise-linear curve, e.g.
  `(64 B -> 0.10), (1 KB -> 0.40), (16 KB -> 0.90), (>= 64 KB -> 1.00)`.
  Captures the tiny-IO regime where PCIe cannot be saturated by BW alone.
- `util_i(t)` comes from a sorted **interval set** per peer. Every consumer
  calls `reserve_peer(...)` before reading the available BW, so two consumers
  enabled at once (e.g. A + B) cannot double-spend a peer's lane.
- One **shared IOPS cap** across all consumers, not per-consumer.
- **Coalescing**: a single `schedule_batch` call in a tick is charged one
  doorbell, not N. Cross-consumer coalescing is achieved by funnelling all
  AMS submissions through the IoIssuer's per-tick queue and flushing once.
- **Determinism**: a deterministic tiebreaker `(t_complete, event_id,
  consumer_priority)` resolves floating-point ties in the event queue. The
  small-IO penalty curve is stateless; any random jitter is gated by
  `AmsBackendConfig.seed`.
- **Backpressure**: if `eff_bw(t) -> 0`, the consumer stalls. The stall
  interval is recorded as `ams_stall_time` and the issuer must not busy-loop.

## 9. Stream model: split DMA into PCIe and AMS

`LayerExecutionTime` exposes three streams today (SM, DMA, NCCL), with DMA
conflating KV load + weight load (`vidur/entities/layer_execution_time.py:35-72`).
Once AMS is a second I/O engine, that conflation under-counts concurrency.

Change the model to four streams:

```
wall_clock = max( compute_end, dma_pcie_end, dma_ams_end, comm_end )
```

`to_dict()` and any Chrome-trace exporter must update in lockstep. When AMS is
disabled, `dma_ams_end == 0` and `dma_pcie_end` retains today's semantics.

## 10. Configuration

Add to `vidur/config/config.py` (dataclass, auto-flattened to argparse by
`vidur/config/flat_dataclass.py`):

```python
@dataclass
class AmsBackendConfig:
    enabled: bool = False
    backend_type: str = "ams"                       # MemoryBackendRegistry key
    peer_gpu_ids: list[int] = field(default_factory=list)
    per_peer_pcie_bw_gbps: dict[int, float] | None = None
    doorbell_overhead_us: float = 1.0
    iops_cap: int = 5_000_000
    small_io_penalty_curve: list[tuple[int, float]] = ...
    store_tier: str = "cpu"                         # "cpu" | "nvme"
    capacity_gb: float = 64.0
    eviction_policy: str = "lru"                    # "lru" | "fifo" | "scheduler"
    seed: int | None = None
```

Attach it as an optional sub-config to `BaseReplicaSchedulerConfig` and
`PddSchedulerConfig`. When absent, `HbmOnlyBackend` is used and behavior is
identical to today.

## 11. Metrics

Extend `vidur/metrics/metrics_store.py` and `vidur/metrics/constants.py`:

- **Counters**: `ams_read_count`, `ams_write_count`, `ams_bytes_read`,
  `ams_bytes_written`, `ams_coalesced_ops`.
- **Latency histograms**: `ams_read_latency_p50`, `ams_read_latency_p99`,
  AMS queue depth, doorbell amortization.
- **Per-peer time series**: `peer_pcie_utilization[peer_id]`,
  `peer_pcie_bw_inflight[peer_id]`.
- **Derived**: `bw_amplification = eff_bw_avg / single_peer_bw`,
  `ams_stall_time`.
- **Residency**: per-tier block counts over time; HBM vs AMS hit rates.
- **Impact**: TTFT / TBT / throughput deltas vs paired baseline run.

Outputs land in the existing `simulator_output/<ts>/` directory (CSV +
Chrome trace), with new fields appended. No new file format.

## 12. File map

### Create

- `vidur/memory_backend/__init__.py`
- `vidur/memory_backend/base.py`         — `MemoryBackend` ABC, `BlockKey`, `Tier`, `IoHandle`
- `vidur/memory_backend/hbm_only.py`     — no-op default
- `vidur/memory_backend/ams_backend.py`  — full cost model
- `vidur/memory_backend/registry.py`     — mirrors existing registry pattern
- `vidur/memory_backend/io_issuer.py`    — event emission + per-tick coalescing
- `vidur/events/ams_io_event.py`         — discrete event for AMS reads/writes
- `vidur/entities/block_residency.py`    — per-block residency record (refcount, pin)

### Modify

| File | Lines (approx.) | Change |
|------|-----------------|--------|
| `vidur/scheduler/replica_scheduler/base_replica_scheduler.py` | 40-160   | replace counter allocator with block-map allocator; route reads through backend |
| `vidur/entities/layer_execution_time.py`                       | 1-132    | split DMA stream into `dma_pcie` and `dma_ams`; update wall-clock and `to_dict()` |
| `vidur/execution_time_predictor/base_execution_time_predictor.py` | 43-47, 81-108, 193 | KV load and weight load go through `IoIssuer` when AMS is enabled |
| `vidur/events/kv_cache_transfer_event.py`                      | 13-131   | Option C: AMS path, or tier-flip when blocks already AMS-resident |
| `vidur/entities/prefix_cache_manager.py`                       | —        | residency-aware lookup (refcount, tier annotation); Option D toggles cost |
| `vidur/config/config.py`                                       | ~376-408 | add `AmsBackendConfig`, attach to scheduler configs |
| `vidur/metrics/metrics_store.py`, `vidur/metrics/constants.py` | —        | new metric ids and aggregation hooks |
| `vidur/main.py`                                                | —        | wire backend instantiation through `SimulationConfig.create_from_cli_args()` if required |

### Out of scope (this pass)

- NVMe latency curves beyond a single `store_tier` switch.
- CXL or other future tiers.
- Real-trace AMS workloads (synthetic experiments first).

## 13. Implementation roadmap

Suggested ordering for the follow-up implementation session:

1. **Allocator upgrade.** Replace the counter with the block-map allocator and
   add `BlockResidency`. Refcount and integrate with the prefix-cache manager.
2. **Backend skeleton.** Land `MemoryBackend`, `HbmOnlyBackend`, registry,
   `IoIssuer` with no AMS cost model yet. Verify baseline parity.
3. **Stream split.** Introduce `dma_pcie` / `dma_ams` in `LayerExecutionTime`;
   update wall-clock and exporters. `dma_ams_end` stays 0 until step 5.
4. **Config + metrics plumbing.** `AmsBackendConfig`, metric ids, output
   columns. Still no behavior change with AMS disabled.
5. **`AmsBackend` cost model.** Per-peer interval set, IOPS cap, small-IO
   penalty curve, coalescing, doorbell overhead.
6. **Option A wiring.** KV cache offload tier. First end-to-end experiment.
7. **Option B / C / D.** Layer on once Option A is stable.

## 14. Verification

### End-to-end

1. **Synthetic long-context decode** (new file under `experiments/`): HBM
   sized to ~30 % of working set, batch >= 64, decode-heavy. Run twice — with
   and without AMS. Compare TTFT, TBT, throughput, HBM/AMS hit ratios.
2. **MoE streaming experiment** (Option B sanity): MoE model where experts
   exceed HBM. AMS-on should reduce expert-load tail latency.
3. **PDD-with-AMS experiment** (Option C): re-run `experiments/test_pdd_simple.py`
   with AMS enabled; verify transfer time drops when blocks are tier-flips.

### Parity

4. Existing ShareGPT / Azure trace runs produce bit-identical metrics when
   `AmsBackendConfig.enabled = False`. CI test.

### Unit

5. **Closed-form cost test**: one peer at known BW, one read of known size,
   assert `t_complete` within 1 ns.
6. **BW saturation**: many concurrent reads — `eff_bw` never exceeds
   `Σ peer_pcie_bw`.
7. **IOPS saturation**: many tiny reads — `t_iops` dominates `t_bw`.
8. **Contention**: reserve a peer's BW — `eff_bw` drops accordingly.
9. **Coalescing**: N reads in the same tick — one doorbell charge, not N.
10. **Determinism**: same seed produces identical event order and timestamps.

Pre-existing tests in `experiments/` (`test_pdd_simple.py`,
`test_prefix_cache.py`, `run_pdd_test.py`) must continue to pass unchanged.

## 15. Risks and open questions

- **Prefix-cache refcount semantics.** Demotion of a shared block to AMS_CPU
  invalidates the radix node's "free read" assumption — needs explicit
  handling in `PrefixCacheManager`.
- **Floating-point timestamp ties.** The deterministic tiebreaker
  `(t_complete, event_id, consumer_priority)` must be enforced everywhere
  the event queue is consumed, not just in AMS events.
- **Capacity and eviction on the AMS tier.** Behavior when AMS_CPU is full
  (spill to NVMe vs reject vs stall) must be specified before Option A goes
  live.
- **Backpressure correctness.** `eff_bw(t) -> 0` must produce a recorded
  stall, not a busy loop in the issuer.
- **Async semantics.** The `IoHandle` exposes both `t_issue` and `t_complete`
  so the layer-time accountant can compute overlap with compute correctly.
- **PDD × AMS interaction.** When KV blocks are already AMS-resident, the PDD
  transfer is a tier-flip, not a copy — explicit code path, not implicit.
- **MoE × KV co-existence.** Both consumers competing for the same peer-PCIe
  BW is the main reason for the shared `reserve_peer` interface; this should
  be exercised by a dedicated unit test once Option B lands.

## 16. Future work

- NVMe-tier latency modeling (curve, queue depth).
- CXL / additional tiers as new `Tier` enum values + `MemoryBackend` subclasses.
- Real-trace replay with AMS enabled, including ablations across
  `iops_cap` and `peer_gpu_ids` topologies.
- AMS-aware schedulers: scheduling decisions that take residency tier and
  predicted AMS latency into account when admitting new requests.
