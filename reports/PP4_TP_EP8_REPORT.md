# DSA vs IndexCache at PP=4, TP={2, 4}, EP=8

Decode-time simulation of DeepSeek-V3 sparse attention with and without
IndexCache under a **realistic moderate-cluster parallelism**: 4-way
pipeline parallelism, 2× or 4× tensor parallelism, 8× expert
parallelism. **All compute and IO are derived analytically** from H100
peak bandwidth and FLOPS — no profiled CSV. The validator at
`experiments/validate_pp2_tp.py` hand-derives every metric and runs 11
identity checks against the model; all pass to floating-point precision.

## Why this config?

| Parameter | Value | Reasoning |
|---|---|---|
| PP | 4 | Shards layers (KV/IDX caches and weights). 61/4 = 16 layers/stage. |
| TP | 2 or 4 | Shards attention/MLP weights and compute. **Does not** help MLA KV memory (duplication tax). |
| EP | 8 | One active expert per rank per token on average at BS=1 (8 active of 256, 8 EP groups). EP=16 over-shards (0.5 active/rank, half idle). |
| Total GPUs | 32 (TP=2) or 32 (TP=4) | 4 nodes × 8 H100s. Same cluster size regardless of TP. |

PP=2 EP=8 was the natural request, but with FP16 weights it doesn't
fit: 31 layers/stage × 21 GB/layer / 8 EP = **81 GB per rank** for
expert weights alone, exceeding H100's 80 GB. Two ways to fix:
(a) FP8 expert weights (production DSV3.2 approach, halves it to 40
GB), or (b) bump PP to 4 (halves layers per stage to 16, expert
weights drop to 42 GB). We pick (b) for clarity and to keep all
weights at FP16.

## Per-rank memory (hand-verified)

```
Constants:
  non-expert weights / layer = 450.50 MB (FP16, 8 GEMMs summed)
  expert weights / layer     =  21.00 GB (FP16, 256 × 84 MB)
  total weights              =  1307.84 GB
```

At BS=1, sl=200K:

| Component | PP=4 TP=2 EP=8 | PP=4 TP=4 EP=8 |
|---|---:|---:|
| KV (16 × 1 × 200K × 1152)         |  3.52 GB |  3.52 GB |
| IDX (16 × 1 × 200K × 512)         |  1.56 GB |  1.56 GB |
| Non-expert W / TP / PP            |  3.52 GB |  1.76 GB |
| Expert W / EP / PP                | 42.00 GB | 42.00 GB |
| **Total per rank**                | **50.60 GB** | **48.84 GB** |
| Headroom in 80 GB                 | 29.40 GB | 31.16 GB |

The 42 GB of expert weights dominate. KV and IDX scale linearly with
`BS × sl × 16` (= layers per stage). Feasible envelope at PP=4 TP=4 EP=8:

| seq_len | max BS that fits (with 8 GB headroom) |
|---|---:|
| 4K   | 32+ (any reasonable) |
| 32K  | ~32 |
| 128K | ~8  |
| 200K | ~5  |
| 512K | 1   |
| 1M   | 1   |

This is a much larger feasible region than EP=16 (which over-sharded
experts and gained nothing in the BS=1 regime).

## Analytical per-piece per-layer cost (BS=1 sl=200K offload, TP=1 baseline)

```
─── IO stream (placement-dependent) ─────────────────────────
  idx_io      1 × 200K × 512 B / 51.5 GB/s   =  1.896 ms   (F only)
  kv_io       1 × 2560 × 1152 B / 51.5 GB/s  =  0.053 ms   (every layer)
─── Block A: Q projection (TP-shardable) ────────────────────
  pre_norm + q_down + q_up + rope + kv_down  =  0.111 ms
─── Block C: attended-token attn + o_proj ───────────────────
  kv_up + attn core + o_proj + residual      =  0.456 ms
─── Indexer compute (F only) ────────────────────────────────
  matmul + topk floors                       =  0.010 ms
─── MoE shardable ───────────────────────────────────────────
  moe_norm + router + softmax + topk
  + shared expert + residual                 =  0.149 ms
─── MoE NOT TP-shardable ────────────────────────────────────
  expert_gemm  (EP=8, 1 active expert × 84 MB)= 0.059 ms
  ep_comms     (8 NVLink msgs × 2 passes)    =  0.080 ms
─────────────────────────────────────────────────────────────
  T_compute_F (TP=1) = max(0, 0.111 − 1.949)  ← block_a hidden in idx_io
                     + 0.456 + 0.010 + 0.149
                     + 0.059 + 0.080
                     = 0.754 ms

  T_compute_S (TP=1) = max(0, 0.111 − 0.053)  ← block_a partly exposed (no idx_io)
                     + 0.456 + 0.000 + 0.149
                     + 0.059 + 0.080
                     = 0.802 ms
```

Note the EP=8 expert_gemm is **0.059 ms** (1 active expert × 84 MB at
HBM 1384 GB/s) vs the previous EP=16 figure of 0.030 ms. EP=8 doubles
expert_gemm time but doubles the active-expert load, so it's the
correct shard for BS=1 (avoids the "half ranks idle" artefact).

### TP scaling

Pieces divided by TP: `block_a, block_c, idx_comp, moe_shardable`.
Not divided by TP: `expert_gemm, ep_comms`. So:

```
T_compute(TP=k) = max(0, block_a/k − T_io)
                + (block_c + idx_comp + moe_shardable) / k
                + expert_gemm + ep_comms
```

| TP | T_F (ms) | T_S (ms) |
|----|---------:|---------:|
|  1 |  0.7545  |  0.8020  |
|  2 |  0.4470  |  0.4441  |
|  4 |  0.2933  |  0.2908  |

At TP=4 the floor `expert_gemm + ep_comms = 0.139 ms` dominates the
per-layer compute (compute can't shrink below that even at TP→∞).

## PP=4 step-time schedule (BS=1 sl=200K offload, TP=4)

PP=4 means 4 stages with 16, 16, 16, 13 layers. Each stage has its
own PCIe bus (independent IO prefetch).

```
DSA (all-F, 61 F-layers):
  Stage 0 (16 F): io_total = 31.19 ms, compute = 4.69 ms (TP=4)
                  producer/consumer: IO dominates → end = 31.49 ms
  Stage 1 (16 F): io_total = 31.19, compute = 4.69
                  IO has prefetched since t=0 → ready by t=31.19
                  compute_start = max(31.49, 31.19) = 31.49 → end = 36.18 ms
  Stage 2 (16 F): end = 40.87 ms
  Stage 3 (13 F): io = 25.34, compute = 3.81 → end = 44.69 ms

IndexCache F:S:S:S (16 F + 45 S):
  Stage 0 (4 F + 12 S): io_total = 8.44 ms, compute = 4.66 ms
                        producer/consumer → end =  9.44 ms
  Stage 1 (4 F + 12 S): end = 14.11 ms
  Stage 2 (4 F + 12 S): end = 18.77 ms
  Stage 3 (4 F + 9 S):  end = 22.56 ms
```

**Speedup = 44.69 / 22.56 = 1.98×**.

## Full sweep across feasible corners

### PP=4 TP=4 EP=8 — offload mode

`TPOT` = time per output token, per user (= step time, since each
user receives one token per decode step). `tok/s` = aggregate
throughput across the BS users in flight = `BS × 1000 / step_ms`.

| scenario     | mem/rank | fits? | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s | DSA stage 0 | IC stage 0 |
|--------------|---------:|------:|---------:|--------:|--------:|---------:|--------:|----------:|---------:|------------:|-----------:|
| 4K   / BS=1  |  43.9 GB | YES   |  17.99 ms |  17.87 | 1.01× |  17.99 |  17.87 |   55.6 |   56.0 |     4.79  |     4.76 |
| 32K  / BS=1  |  44.6 GB | YES   |  19.20 ms |  18.14 | 1.06× |  19.20 |  18.14 |   52.1 |   55.1 |     6.00  |     5.02 |
| 128K / BS=1  |  47.0 GB | YES   |  33.76 ms |  19.83 | 1.70× |  33.76 |  19.83 |   29.6 |   50.4 |    20.56  |     6.71 |
| **200K / BS=1** | **48.8 GB** | **YES** | **44.69 ms** | **22.56** | **1.98×** | **44.69** | **22.56** | **22.4** | **44.3** | **31.49** | **9.44** |
| 512K / BS=1  |  56.8 GB | YES   |  92.02 ms |  34.39 | 2.68× |  92.02 |  34.39 |   10.9 |   29.1 |    78.82  |    21.28 |
| 1M   / BS=1  |  69.8 GB | YES   | 169.69 ms |  53.81 | 3.15× | 169.69 |  53.81 |    5.9 |   18.6 |   156.49  |    40.69 |
| 4K   / BS=8  |  44.6 GB | YES   |  76.12 ms |  75.61 | 1.01× |  76.12 |  75.61 |  105.1 |  105.8 |    20.50  |    20.37 |
| 32K  / BS=8  |  50.3 GB | YES   | 102.51 ms |  77.74 | 1.32× | 102.51 |  77.74 |   78.0 |  102.9 |    46.90  |    22.49 |
| 128K / BS=8  |  69.8 GB | YES   | 219.02 ms | 104.53 | 2.10× | 219.02 | 104.53 |   36.5 |   76.5 |   163.40  |    49.29 |
| 4K   / BS=32 |  47.0 GB | YES   | 246.92 ms | 245.06 | 1.01× | 246.92 | 245.06 |  129.6 |  130.6 |    66.92  |    66.42 |
| 32K  / BS=32 |  69.8 GB | YES   | 366.64 ms | 255.53 | 1.43× | 366.64 | 255.53 |   87.3 |  125.2 |   186.65  |    76.90 |

Reading the table:
- TPOT and step are the same number — one token per decode step per
  user. Lower TPOT = each user perceives faster generation.
- At BS=1 sl=200K, IndexCache cuts per-user TPOT from 44.7 ms to
  22.6 ms — **22.4 tok/s → 44.3 tok/s for that single user**.
- At BS=8 sl=128K the cluster's *aggregate* throughput grows from
  36.5 tok/s (DSA) to 76.5 tok/s (IC); each of the 8 users still
  waits ~105 ms vs ~219 ms between their tokens.
- The "DSA stage 0" column is the IO-bound bottleneck stage; when
  DSA stage 0 ≫ IC stage 0, IndexCache's speedup is that ratio.

### PP=4 TP=2 EP=8 — offload mode

Same workloads at TP=2 (slower compute, same IO):

| scenario     | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K / BS=1    |   27.36 |   27.23 | 1.00× |  27.36 |  27.23 | 36.6 | 36.7 |
| 200K / BS=1  |   51.76 |   30.08 | 1.72× |  51.76 |  30.08 | 19.3 | 33.2 |
| 1M / BS=1    |  176.76 |   61.33 | 2.88× | 176.76 |  61.33 |  5.7 | 16.3 |
| 32K / BS=8   |  135.98 |  121.61 | 1.12× | 135.98 | 121.61 | 58.8 | 65.8 |
| 128K / BS=8  |  252.48 |  139.78 | 1.81× | 252.48 | 139.78 | 31.7 | 57.2 |

TP=2 doubles compute, which makes the per-stage compute slightly more
significant relative to IO — DSA gets a slightly bigger compute share,
and IndexCache's relative speedup shrinks (1.72× at TP=2 vs 1.98× at
TP=4 for the 200K/BS=1 corner).

### PP=4 TP=4 EP=8 — HBM mode

| scenario     | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K / BS=1    |   17.92 |   18.38 | 0.98× | 17.92 | 18.38 | 55.8 | 54.4 |
| 200K / BS=1  |   17.98 |   18.44 | 0.98× | 17.98 | 18.44 | 55.6 | 54.2 |
| 512K / BS=1  |   18.09 |   18.55 | 0.98× | 18.09 | 18.55 | 55.3 | 53.9 |
| 1M / BS=1    |   19.51 |   18.73 | 1.04× | 19.51 | 18.73 | 51.2 | 53.4 |

HBM mode is compute-bound except at sl≥1M. IndexCache then has nothing
to save and the lost block_a‖idx_io overlap on S layers costs ~2-3%.

## Validation summary

All 11 identity checks at every step (`experiments/validate_pp2_tp.py`):

| Check | Status |
|---|---|
| Non-expert weights per layer (FP16, 8 GEMMs summed) | ✓ |
| Expert weights per layer (256 × 84 MB) | ✓ |
| KV bytes per rank at PP=4 TP=2 EP=8 | ✓ |
| IDX bytes per rank at PP=4 TP=2 EP=8 | ✓ |
| Non-expert W bytes per rank | ✓ |
| Expert W bytes per rank | ✓ |
| Total bytes per rank | ✓ |
| (same five for PP=4 TP=4 EP=8) | ✓ |
| F-layer idx_io = 200K × 512 / 51.5 GB/s | ✓ |
| F-layer kv_io = 2560 × 1152 / 51.5 GB/s | ✓ |
| T_compute(TP=2) F-layer composition | ✓ |
| Stage 0 producer/consumer end (16-layer manual walk) | ✓ |

## Comparison vs. earlier (over-sharded) configs

For reference, results at the same workload (BS=1 sl=200K offload) across configs:

| Config | DSA step | IC step | speedup | per-rank memory | comments |
|---|---:|---:|---:|---:|---|
| PP=2 TP=4 EP=16 | 68.6 ms |  25.4 ms | 2.71× | 53.9 GB | over-sharded EP |
| **PP=4 TP=4 EP=8** | **44.7 ms** | **22.6 ms** | **1.98×** | **48.8 GB** | balanced, realistic |
| PP=8 TP=1 EP=8 | 54.0 ms |  46.0 ms | 1.17× | 27.1 GB | very high PP, IO already hidden |

Going from PP=2 to PP=4 cuts the per-stage IO load in half, so DSA's
step time drops 35% (less IO-bound). IndexCache's relative speedup
correspondingly drops from 2.71× to 1.98×.

**The picture:** higher PP makes DSA itself less IO-bound (each stage
holds fewer layers' worth of indexer K reads), so IndexCache's headline
win shrinks. The absolute step time, however, is best at the chosen
PP=4 EP=8 config — both DSA and IC are faster than at PP=2, just with
a smaller gap between them.

## Key takeaways

1. **IndexCache delivers 1.0–3.2× speedup** at PP=4 TP=4 EP=8 offload,
   monotonically increasing with `BS × seq_len`. At long context
   (sl ≥ 128K BS=1), the speedup approaches the F-period ceiling
   `N / n_F = 61/16 ≈ 3.81×`.

2. **EP=8 is the right shard for BS=1 inference**: 1 active expert per
   rank per token on average. EP=16 over-shards (0.5 active/rank,
   half-idle ranks at BS=1).

3. **PP shards both memory and parallelizes IO**: per-rank cache shrinks
   linearly with PP, and the IO bus can prefetch in parallel across
   stages. Higher PP shrinks the IndexCache benefit (because DSA itself
   becomes less IO-bound).

4. **TP=4 vs TP=2** at the same PP/EP halves per-stage compute but
   leaves IO unchanged → DSA becomes more IO-bound → IndexCache
   relative win grows (1.72× → 1.98× at 200K/BS=1).

5. **MLA duplication tax**: TP doesn't reduce per-rank KV/IDX cache,
   only PP does. PP=4 is the smallest PP that fits EP=8 (FP16 expert
   weights). To go to PP=2 EP=8 you'd need FP8 expert weights
   (DeepSeek-V3.2-Exp production approach).

## Reproducing

```bash
python -m experiments.validate_pp2_tp
```

Self-contained: imports only `experiments.analytical_layer` (pure
analytical), runs all identity checks, prints every fine-grained
metric. Total runtime: under 1 second.
