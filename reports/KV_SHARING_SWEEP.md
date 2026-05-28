# KV Sharing Config Comparison — Where SSD Hits the Wall

## Setup

- Cluster: PP=2, TP=2, EP=8 (16 H100s), FP8 + MLA absorption
- Scheme: IndexCache (F:S:S:S — 16 F + 45 S layers of 61)
- Model output of `exec_calculator.html`, mirrored in `experiments/kv_sharing_sweep.py`

## Configs compared

| key | KV residence | Within-stage sharing | Local I/O mode |
|---|---|---|---|
| `hbm_full` | HBM | off (duplicates) | n/a |
| `hbm_shard` | HBM | on (1/G local + (G-1)/G via NVLink) | n/a |
| `ssd_full` | SSD | off | per-token IOPS-counted |
| `ssd_shard` | SSD | on | per-token IOPS-counted |
| `ssd_shard_merge` | SSD | on | merged → bandwidth-bound |

`G = EP = 8` GPUs per PP stage.

## The story (4 plots in `reports/figures/kv_sharing/`)

### `tpot_vs_sl.png` — TPOT vs sequence length at BS_cluster=24
- **SSD, no sharing** (red): pinned at ~170 ms regardless of sl — IOPS saturated even at small sl.
- **SSD, sharded IOPS-counted** (yellow): walks above the 30 ms target at sl≈1M; collapses fast past that.
- **SSD, sharded + merged** (purple) and **both HBM** (green/cyan): tied until ~2M where SSD-merged starts to drift above HBM (SSD BW becomes binding).

### `tpot_vs_bs.png` — TPOT vs BS_cluster at sl=512K
- **SSD, no sharing**: misses 30 ms target even at BS=4. Useless.
- **SSD, sharded IOPS-counted**: hits target at BS≈22, climbs IOPS-bound thereafter.
- **SSD, sharded + merged** and both HBM lines: track each other up to BS≈24 (compute-bound regime — MoE GEMM is the floor for all of them), merged diverges slowly past BS=64.

### `ssd_saturation.png` — Where SSD runs out (BW and IOPS utilization)
- **SSD, no sharing**: **~170 % of IOPS cap** — fundamentally infeasible.
- **SSD, sharded IOPS-counted**: sits at ~100 % IOPS — at the cliff.
- **SSD, sharded + merged**: ~0 % IOPS (collapsed to ~1 op/layer) — bandwidth-bound at ~5 % of cap.
- **BW panel** confirms SSD is IOPS-bound (left axis tops at 10 %), not BW-bound — until merging removes IOPS, then it's BW-bound from below.

### `max_bs_at_tpot.png` — Max BS_cluster at 30 ms TPOT, vs sl
- **SSD, no sharing**: collapses to 0 (even BS=1 misses target) past 1M sl.
- All others tied at **BS_cluster ≈ 24** — this is the **compute ceiling** (MoE GEMM at BS=6/rank), not a KV-IO limit. The KV-IO differences don't show up in the max-BS metric at this TPOT target.

## Headline

**Even SSD is not enough at scale.** With no sharing, IOPS is at 170 % of cap before you even start — the config is broken from sl=128K. With sharing alone, you reach ~100 % IOPS at ~24 batch — right at the cliff. Only **merging the per-token I/Os into sequential transfers** unblocks IOPS, after which **the SSD BW (28 GB/s) becomes the next ceiling** and diverges from HBM (1384 GB/s) past sl≈2M / BS≈64.

The compute ceiling (MoE expert GEMM, unsharded across TP) caps useful BS at ≈24 for every config that gets that far — so the KV-IO win shows in *how late you hit the wall*, not the wall height itself.

## Reproduce

```bash
python -m experiments.kv_sharing_sweep
```
Outputs to `reports/figures/kv_sharing/`.
