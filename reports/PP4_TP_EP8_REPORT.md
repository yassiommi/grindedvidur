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
| 128K / BS=32 | 147.8 GB | NO    | 832.66 ms | 372.04 | 2.24× | 832.66 | 372.04 |   38.4 |   86.0 |   652.67  |   193.40 |
| 200K / BS=32 | 206.3 GB | NO    |1182.18 ms | 459.42 | **2.57×** |1182.18 | 459.42 |   27.1 |   69.7 |  1002.18  |   280.78 |
| 512K / BS=32 | 459.8 GB | NO    |2696.74 ms | 838.06 | **3.22×** |2696.74 | 838.06 |   11.9 |   38.2 |  2516.74  |   659.42 |

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

### PP=4 TP=4 EP=8 — HBM mode (full sweep, FP16)

Rows marked **`NO`** in the `fits?` column have per-rank memory above
80 GB at this PP/TP/EP — shown for the trend; would need higher PP or
H200 (141 GB) to actually deploy. Times still reflect the workload if
the cache existed in HBM.

| scenario     | mem/rank | fits? | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|------:|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K   / BS=1  |  43.9 GB | YES |   17.92 |   18.38 | 0.98× |  17.92 |  18.38 |   55.8 |   54.4 |
| 32K  / BS=1  |  44.6 GB | YES |   17.92 |   18.38 | 0.98× |  17.92 |  18.38 |   55.8 |   54.4 |
| 128K / BS=1  |  47.0 GB | YES |   17.95 |   18.41 | 0.98× |  17.95 |  18.41 |   55.7 |   54.3 |
| 200K / BS=1  |  48.8 GB | YES |   17.98 |   18.44 | 0.98× |  17.98 |  18.44 |   55.6 |   54.2 |
| 512K / BS=1  |  56.8 GB | YES |   18.09 |   18.55 | 0.98× |  18.09 |  18.55 |   55.3 |   53.9 |
| 1M   / BS=1  |  69.8 GB | YES |   19.51 |   18.73 | **1.04×** |  19.51 |  18.73 |   51.2 |   53.4 |
| 2M   / BS=1  |  95.8 GB | NO  |   25.29 |   19.09 | **1.32×** |  25.29 |  19.09 |   39.5 |   52.4 |
| 4M   / BS=1  | 147.8 GB | NO  |   36.86 |   20.72 | **1.78×** |  36.86 |  20.72 |   27.1 |   48.3 |
| 4K   / BS=8  |  44.6 GB | YES |   75.42 |   75.45 | 1.00× |  75.42 |  75.45 |  106.1 |  106.0 |
| 32K  / BS=8  |  50.3 GB | YES |   75.49 |   75.52 | 1.00× |  75.49 |  75.52 |  106.0 |  105.9 |
| 128K / BS=8  |  69.8 GB | YES |   75.76 |   75.79 | 1.00× |  75.76 |  75.79 |  105.6 |  105.6 |
| 200K / BS=8  |  84.4 GB | NO  |   75.97 |   76.00 | 1.00× |  75.97 |  76.00 |  105.3 |  105.3 |
| 512K / BS=8  | 147.8 GB | NO  |   80.23 |   76.88 | **1.04×** |  80.23 |  76.88 |   99.7 |  104.1 |
| 4K   / BS=32 |  47.0 GB | YES |  244.11 |  242.25 | 1.01× | 244.11 | 242.25 |  131.1 |  132.1 |
| 32K  / BS=32 |  69.8 GB | YES |  244.42 |  242.57 | 1.01× | 244.42 | 242.57 |  130.9 |  131.9 |
| 128K / BS=32 | 147.8 GB | NO  |  245.51 |  243.65 | 1.01× | 245.51 | 243.65 |  130.3 |  131.3 |
| 200K / BS=32 | 206.3 GB | NO  |  246.32 |  244.46 | 1.01× | 246.32 | 244.46 |  129.9 |  130.9 |
| 512K / BS=32 | 459.8 GB | NO  |  277.50 |  247.98 | **1.12×** | 277.50 | 247.98 |  115.3 |  129.0 |

**The HBM crossover** is clearly visible in the BS=1 column: at
sl≤512K the per-stage compute (4.7 ms at TP=4) absorbs the per-stage
HBM indexer-K read; beyond 1M the IO catches up and IndexCache starts
to genuinely help. At BS=1 sl=4M, IndexCache is **1.78×** even in HBM
mode — the HBM bandwidth is being saturated by 16 GB of indexer-K read
per stage. This regime needs PP=8 or H200 to actually fit.

For BS=8 the crossover sits between 200K and 512K: at sl=512K we get
1.04× HBM speedup, and that 512K BS=8 row uses 147.8 GB / rank — also
needs PP=8 or H200.

### PP=4 TP=2 EP=8 — HBM mode (full sweep, FP16)

Same workloads at TP=2 (slower compute → HBM crossover happens later
in sl):

| scenario     | mem/rank | fits? | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|------:|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K   / BS=1  |  45.6 GB | YES |   28.85 |   29.30 | 0.98× |  28.85 |  29.30 |   34.7 |   34.1 |
| 200K / BS=1  |  50.6 GB | YES |   27.36 |   28.95 | 0.94× |  27.36 |  28.95 |   36.6 |   34.5 |
| 512K / BS=1  |  58.5 GB | YES |   27.47 |   29.06 | 0.95× |  27.47 |  29.06 |   36.4 |   34.4 |
| 1M   / BS=1  |  71.5 GB | YES |   27.65 |   29.24 | 0.95× |  27.65 |  29.24 |   36.2 |   34.2 |
| 2M   / BS=1  |  97.5 GB | NO  |   32.37 |   29.60 | **1.09×** |  32.37 |  29.60 |   30.9 |   33.8 |
| 4M   / BS=1  | 149.5 GB | NO  |   43.93 |   30.32 | **1.45×** |  43.93 |  30.32 |   22.8 |   33.0 |
| 4K   / BS=8  |  46.3 GB | YES |  121.30 |  120.97 | 1.00× | 121.30 | 120.97 |   65.9 |   66.1 |
| 32K  / BS=8  |  52.0 GB | YES |  119.88 |  120.65 | 0.99× | 119.88 | 120.65 |   66.7 |   66.3 |
| 128K / BS=8  |  71.5 GB | YES |  120.15 |  120.92 | 0.99× | 120.15 | 120.92 |   66.6 |   66.2 |
| 200K / BS=8  |  86.1 GB | NO  |  120.35 |  121.12 | 0.99× | 120.35 | 121.12 |   66.5 |   66.0 |
| 512K / BS=8  | 149.5 GB | NO  |  121.23 |  122.00 | 0.99× | 121.23 | 122.00 |   66.0 |   65.6 |
| 4K   / BS=32 |  48.8 GB | YES |  408.72 |  405.01 | 1.01× | 408.72 | 405.01 |   78.3 |   79.0 |
| 32K  / BS=32 |  71.5 GB | YES |  409.04 |  405.32 | 1.01× | 409.04 | 405.32 |   78.2 |   78.9 |
| 128K / BS=32 | 149.5 GB | NO  |  410.12 |  406.41 | 1.01× | 410.12 | 406.41 |   78.0 |   78.7 |
| 200K / BS=32 | 208.0 GB | NO  |  410.93 |  407.22 | 1.01× | 410.93 | 407.22 |   77.9 |   78.6 |
| 512K / BS=32 | 461.5 GB | NO  |  414.45 |  410.74 | 1.01× | 414.45 | 410.74 |   77.2 |   77.9 |

**Big takeaways for HBM mode:**

- In the feasible region (fits in 80 GB H100) at BS≥8, HBM is fully
  compute-bound and IndexCache is a wash (1.00–1.01×). The TPOT and
  tok/s numbers are essentially identical with or without IndexCache.
- The single-user BS=1 long-context corner is where HBM-mode IC starts
  to genuinely help — but only at sl≥1M, which is beyond what fits in
  PP=4. At sl=2M, IC is 1.09× (TP=2) or 1.32× (TP=4); at sl=4M it's
  1.45× (TP=2) or 1.78× (TP=4).
- These long-context HBM corners are exactly where the next paragraph
  about FP8 becomes interesting — FP8 halves the per-rank memory, so
  the 2M BS=1 corner *does* fit, and FP8 halves compute, which shifts
  the HBM crossover to shorter sl (more contexts where IC actually
  helps in HBM).

## FP8 path (matches DeepSeek-V3.2-Exp production)

Adding FP8 for weights and KV cache (1 byte/element instead of FP16's
2). Indexer K is FP8 in both paths (unchanged). Net effect:

- **Per-rank memory roughly halves**: at PP=4 TP=4 EP=8 BS=1 sl=200K,
  total goes from 48.84 GB → **25.20 GB**. The 21 GB expert weights
  become 10.5 GB; non-expert weights 1.76 → 0.88 GB; KV 3.52 → 1.76 GB;
  IDX unchanged at 1.56 GB.
- **Per-layer compute decreases** since most GEMMs are HBM-weight-read-
  bound at BS=1 and reads halve. F-layer T_compute at TP=1 drops from
  0.755 ms (FP16) to 0.467 ms (FP8). The non-halvable floor is
  `ep_comms + ep_dispatch` (NVLink, 0.080 ms).
- **Indexer-K IO is unchanged** because it was already FP8. So the
  *relative* IO/compute ratio goes up under FP8 — IndexCache wins
  slightly more.

Hand-verified FP8 memory at PP=4 TP=4 EP=8 BS=1 sl=200K (all 5 checks pass):

```
KV   = 16 layers × 1 × 200K × 576 B = 1.76 GB
IDX  = 16 layers × 1 × 200K × 512 B = 1.56 GB
nonW = 16 × 225.25 MB / 4 (TP)       = 0.88 GB
expW = 16 × 10.5 GB     / 8 (EP)     = 21.00 GB
Total                                = 25.20 GB
```

### PP=4 TP=4 EP=8 — FP8 — offload mode

| scenario     | mem/rank | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K   / BS=1  |  22.0 GB |   12.23 |   12.11 | 1.01× |  12.23 |  12.11 |   81.8 |   82.6 |
| 32K  / BS=1  |  22.4 GB |   14.45 |   12.38 | 1.17× |  14.45 |  12.38 |   69.2 |   80.8 |
| 128K / BS=1  |  24.0 GB |   29.01 |   14.88 | 1.95× |  29.01 |  14.88 |   34.5 |   67.2 |
| **200K / BS=1** | **25.2 GB** | **39.94** | **17.61** | **2.27×** | **39.94** | **17.61** | **25.0** | **56.8** |
| 512K / BS=1  |  30.4 GB |   87.27 |   29.44 | 2.96× |  87.27 |  29.44 |   11.5 |   34.0 |
| 1M   / BS=1  |  38.9 GB |  164.94 |   48.86 | 3.38× | 164.94 |  48.86 |    6.1 |   20.5 |
| 4K   / BS=8  |  22.4 GB |   43.92 |   43.42 | 1.01× |  43.92 |  43.42 |  182.1 |  184.3 |
| 32K  / BS=8  |  26.1 GB |   74.98 |   46.95 | 1.60× |  74.98 |  46.95 |  106.7 |  170.4 |
| 128K / BS=8  |  38.9 GB |  191.49 |   76.07 | 2.52× | 191.49 |  76.07 |   41.8 |  105.2 |
| 200K / BS=8  |  48.4 GB |  278.87 |   97.92 | **2.85×** | 278.87 |  97.92 |   28.7 |   81.7 |
| 512K / BS=8  |  89.9 GB (NO) |  657.51 |  192.58 | **3.41×** | 657.51 | 192.58 |   12.2 |   41.5 |
| 4K   / BS=32 |  24.0 GB |  138.46 |  136.60 | 1.01× | 138.46 | 136.60 |  231.1 |  234.3 |
| 32K  / BS=32 |  38.9 GB |  271.85 |  158.00 | 1.72× | 271.85 | 158.00 |  117.7 |  202.5 |
| 128K / BS=32 |  89.9 GB (NO) |  737.86 |  274.51 | **2.69×** | 737.86 | 274.51 |   43.4 |  116.6 |
| 200K / BS=32 | 128.1 GB (NO) | 1087.38 |  361.89 | **3.00×** |1087.38 | 361.89 |   29.4 |   88.4 |
| 512K / BS=32 | 293.9 GB (NO) | 2601.94 |  740.53 | **3.51×** |2601.94 | 740.53 |   12.3 |   43.2 |

### PP=4 TP=2 EP=8 — FP8 — offload mode

| scenario     | mem/rank | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K   / BS=1  |  22.8 GB |   17.68 |   17.84 | 0.99× |  17.68 |  17.84 |   56.6 |   56.1 |
| 200K / BS=1  |  26.1 GB |   44.05 |   22.21 | 1.98× |  44.05 |  22.21 |   22.7 |   45.0 |
| 1M   / BS=1  |  39.8 GB |  169.05 |   53.46 | 3.16× | 169.05 |  53.46 |    5.9 |   18.7 |
| 32K / BS=8   |  27.0 GB |   94.12 |   70.41 | 1.34× |  94.12 |  70.41 |   85.0 |  113.6 |
| 128K / BS=8  |  39.8 GB |  210.63 |   96.05 | 2.19× | 210.63 |  96.05 |   38.0 |   83.3 |
| 32K / BS=32  |  39.8 GB |  342.66 |  237.15 | 1.44× | 342.66 | 237.15 |   93.4 |  134.9 |

### PP=4 TP=4 EP=8 — FP8 — HBM mode (full sweep)

| scenario     | mem/rank | fits? | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|------:|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K   / BS=1  |  22.0 GB | YES |   12.19 |   12.20 | 1.00× |  12.19 |  12.20 |   82.0 |   82.0 |
| 32K  / BS=1  |  22.4 GB | YES |   12.19 |   12.20 | 1.00× |  12.19 |  12.20 |   82.0 |   82.0 |
| 128K / BS=1  |  24.0 GB | YES |   12.22 |   12.23 | 1.00× |  12.22 |  12.23 |   81.8 |   81.8 |
| 200K / BS=1  |  25.2 GB | YES |   12.25 |   12.25 | 1.00× |  12.25 |  12.25 |   81.7 |   81.6 |
| 512K / BS=1  |  30.4 GB | YES |   12.36 |   12.36 | 1.00× |  12.36 |  12.36 |   80.9 |   80.9 |
| 1M   / BS=1  |  38.9 GB | YES |   15.19 |   12.54 | **1.21×** |  15.19 |  12.54 |   65.8 |   79.7 |
| 2M   / BS=1  |  55.9 GB | YES |   20.97 |   12.90 | **1.63×** |  20.97 |  12.90 |   47.7 |   77.5 |
| 4M   / BS=1  |  89.9 GB | NO  |   32.53 |   15.75 | **2.07×** |  32.53 |  15.75 |   30.7 |   63.5 |
| 4K   / BS=8  |  22.4 GB | YES |   43.44 |   43.05 | 1.01× |  43.44 |  43.05 |  184.2 |  185.8 |
| 32K  / BS=8  |  26.1 GB | YES |   43.51 |   43.13 | 1.01× |  43.51 |  43.13 |  183.9 |  185.5 |
| 128K / BS=8  |  38.9 GB | YES |   43.78 |   43.40 | 1.01× |  43.78 |  43.40 |  182.7 |  184.3 |
| 200K / BS=8  |  48.4 GB | YES |   43.99 |   43.60 | 1.01× |  43.99 |  43.60 |  181.9 |  183.5 |
| 512K / BS=8  |  89.9 GB | NO  |   56.10 |   44.48 | **1.26×** |  56.10 |  44.48 |  142.6 |  179.9 |
| 4K   / BS=32 |  24.0 GB | YES |  136.47 |  134.61 | 1.01× | 136.47 | 134.61 |  234.5 |  237.7 |
| 32K  / BS=32 |  38.9 GB | YES |  136.78 |  134.93 | 1.01× | 136.78 | 134.93 |  233.9 |  237.2 |
| 128K / BS=32 |  89.9 GB | NO  |  137.87 |  136.01 | 1.01× | 137.87 | 136.01 |  232.1 |  235.3 |
| 200K / BS=32 | 128.1 GB | NO  |  139.49 |  136.83 | 1.02× | 139.49 | 136.83 |  229.4 |  233.9 |
| 512K / BS=32 | 293.9 GB | NO  |  195.85 |  140.35 | **1.40×** | 195.85 | 140.35 |  163.4 |  228.0 |

### PP=4 TP=2 EP=8 — FP8 — HBM mode (full sweep)

| scenario     | mem/rank | fits? | DSA step | IC step | speedup | DSA TPOT | IC TPOT | DSA tok/s | IC tok/s |
|--------------|---------:|------:|---------:|--------:|--------:|---------:|--------:|----------:|---------:|
| 4K   / BS=1  |  22.8 GB | YES |   17.96 |   18.41 | 0.98× |  17.96 |  18.41 |   55.7 |   54.3 |
| 200K / BS=1  |  26.1 GB | YES |   17.70 |   18.38 | 0.96× |  17.70 |  18.38 |   56.5 |   54.4 |
| 1M   / BS=1  |  39.8 GB | YES |   19.30 |   18.67 | **1.03×** |  19.30 |  18.67 |   51.8 |   53.6 |
| 2M   / BS=1  |  56.8 GB | YES |   25.08 |   19.03 | **1.32×** |  25.08 |  19.03 |   39.9 |   52.5 |
| 4M   / BS=1  |  90.8 GB | NO  |   36.64 |   20.67 | **1.77×** |  36.64 |  20.67 |   27.3 |   48.4 |
| 4K   / BS=8  |  23.3 GB | YES |   69.14 |   68.80 | 1.00× |  69.14 |  68.80 |  115.7 |  116.3 |
| 32K  / BS=8  |  27.0 GB | YES |   68.89 |   68.79 | 1.00× |  68.89 |  68.79 |  116.1 |  116.3 |
| 128K / BS=8  |  39.8 GB | YES |   69.16 |   69.06 | 1.00× |  69.16 |  69.06 |  115.7 |  115.8 |
| 200K / BS=8  |  49.3 GB | YES |   69.37 |   69.27 | 1.00× |  69.37 |  69.27 |  115.3 |  115.5 |
| 512K / BS=8  |  90.8 GB | NO  |   75.23 |   70.15 | **1.07×** |  75.23 |  70.15 |  106.3 |  114.0 |
| 4K   / BS=32 |  24.9 GB | YES |  230.38 |  226.84 | 1.02× | 230.38 | 226.84 |  138.9 |  141.1 |
| 32K  / BS=32 |  39.8 GB | YES |  230.70 |  227.15 | 1.02× | 230.70 | 227.15 |  138.7 |  140.9 |
| 128K / BS=32 |  90.8 GB | NO  |  231.78 |  228.24 | 1.02× | 231.78 | 228.24 |  138.1 |  140.2 |
| 200K / BS=32 | 129.0 GB | NO  |  232.59 |  229.05 | 1.02× | 232.59 | 229.05 |  137.6 |  139.7 |
| 512K / BS=32 | 294.8 GB | NO  |  266.67 |  232.57 | **1.15×** | 266.67 | 232.57 |  120.0 |  137.6 |

**Putting it together for the requested 200K/512K × BS=8/32 HBM corners:**

| scenario       | TP=4 FP16 spd | TP=4 FP8 spd | TP=2 FP16 spd | TP=2 FP8 spd |
|---------------|-------------:|-------------:|-------------:|-------------:|
| 200K / BS=8   | 1.00× | 1.01× | 0.99× | 1.00× |
| 512K / BS=8   | 1.04× | **1.26×** | 0.99× | **1.07×** |
| 200K / BS=32  | 1.01× | 1.02× | 1.01× | 1.02× |
| 512K / BS=32  | **1.12×** | **1.40×** | 1.01× | **1.15×** |

All four cells *exceed* the H100 80 GB budget at PP=4 EP=8, so they
need either bumping to PP=8 or moving to H200 (141 GB). At PP=4 EP=8
*if they fit*, the **512K/BS=32 FP8 corner gets 1.40× IC speedup**
because the indexer-K read per stage at BS=32 sl=512K under FP8 is
~31 ms — comparable to per-stage compute of ~34 ms. Below 512K BS=32
in HBM mode, the indexer read fits inside compute and IC saves
nothing.

### Key observations under FP8

- **At BS=1 sl=200K**, FP8 speeds up *both* DSA (44.7 → 39.9 ms) and
  IndexCache (22.6 → 17.6 ms), and IndexCache's relative speedup grows
  to **2.27×** (FP16 was 1.98×). Why: compute halves but indexer-K IO
  doesn't, so the IO/compute ratio rises and IC's IO savings matter
  more.
- **TPOT for a single BS=1 user at 200K context**: FP16 DSA = 44.7 ms
  (22 tok/s) → FP8 DSA = 39.9 ms (25 tok/s) → FP8 IC = 17.6 ms
  (57 tok/s). FP8 + IndexCache together give a **2.5× tok/s lift**
  over FP16 DSA.
- **Memory headroom is huge under FP8**: 25.2 GB / rank at BS=1
  sl=200K vs 48.8 GB / rank under FP16. PP=4 EP=8 FP8 fits BS=1
  sl=1M comfortably (38.9 GB) — the entire workload could probably
  drop to PP=2 EP=8 FP8 (which DeepSeek's production stack actually
  does). At PP=2 EP=8 FP8 BS=1 sl=200K, per-rank ~50 GB — fits cleanly.
- **HBM mode is compute-bound everywhere** under FP8 too (1.00× IC
  speedup at BS≥8). Same story as FP16 HBM.

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
