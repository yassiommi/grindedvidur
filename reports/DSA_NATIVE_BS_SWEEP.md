# DSA-Native Batch-Size Sweep at sl=128K

**Config:** PP=2, TP=2, EP=8, FP8, sl=128K, all-F (DSA), MLA absorption


**Storage regimes compared:**
1. **HBM** — MLA KV cache and DSA indexer K both resident in HBM
2. **KV-on-SSD** — KV cache lives on a local 4× Gen4 NVMe RAID (28 GB/s)
   per GPU; indexer K stays in HBM

This report enumerates every variable, derives the memory and compute
costs from first principles, walks the pipeline schedule, and presents
the BS-sweep tables for both storage regimes.

---

## 1. Architecture & DSA constants

| Name | Value | Meaning |
|---|---:|---|
| `NUM_LAYERS` | 61 | total transformer layers |
| `HIDDEN` | 7168 | model hidden dim |
| `NUM_HEADS` | 128 | attention heads |
| `Q_LORA_RANK` | 1536 | Q low-rank dim |
| `KV_LORA_RANK` | 512 | MLA latent dim |
| `QK_NOPE` | 128 | Q/K no-RoPE dim per head |
| `QK_ROPE` | 64 | RoPE dim per head |
| `V_HEAD` | 128 | V dim per head |
| `Q_TOTAL_DIM` | 24576 | NUM_HEADS·(QK_NOPE+QK_ROPE) |
| `KV_UP_OUT_DIM` | 32768 | NUM_HEADS·(QK_NOPE+V_HEAD) |
| `O_PROJ_IN_DIM` | 16384 | NUM_HEADS·V_HEAD |
| `NUM_ROUTED_EXPERTS` | 256 | MoE expert count |
| `NUM_EXPERTS_PER_TOK` | 8 | active per token |
| `EXPERT_INTERMEDIATE` | 2048 | per-expert MLP inner dim |
| `EXPERT_WEIGHT_BYTES` | 84 MB | 3·HIDDEN·EXPERT_INTERMEDIATE·2 (FP16) |
| `DSA_TOPK` | 2048 | indexer-selected tokens |
| `DSA_SLIDING` | 512 | sliding-window kept tokens |
| `DSA_ATTENDED` | 2560 | DSA_TOPK + DSA_SLIDING |
| `INDEXER_DIM` | 128 | DSA lightning-indexer K projection dim |
| `MLA_KV_BYTES_PER_TOKEN` | 1152 (FP16) / **576 (FP8)** | (KV_LORA_RANK+QK_ROPE)·elem |
| `INDEXER_K_BYTES_PER_TOKEN` | **128** | INDEXER_DIM·1B (FP8 always) |

---

## 2. Hardware constants

| Name | Value | Source |
|---|---:|---|
| HBM peak | 1384 GB/s | H100 SXM5 spec |
| FP16 peak compute | 989.5 TFLOPS | H100 spec |
| MFU | 0.50 | observed at decode in this model |
| HBM floor | 0.015 ms | CUDA kernel-launch overhead |
| NVLink latency | 5 µs / message | per-hop |
| NVLink bandwidth | 600 GB/s | symmetric per direction |
| HBM per GPU | 80 GB | H100 SXM5 |
| Usable HBM fraction | 0.90 | fragmentation overhead |
| CUDA scratch budget | 3 GB / rank | runtime context + small activations |
| SSD floor latency | 50 µs / read | OS + driver submission overhead |
| 4× Gen4 RAID BW | 28 GB/s | aggregate across 4 NVMe drives |
| HBM peak IOPS (576 B/op, KV) | 2,580 M/s | 1384 GB/s ÷ 576 B |
| HBM peak IOPS (128 B/op, idx) | **11,610 M/s** | 1384 GB/s ÷ 128 B |
| SSD peak IOPS (576 B/op, KV) | 52.2 M/s | 28 GB/s ÷ 576 B |

---

## 3. Cluster layout (16 GPUs)

`16 = PP × EP = 2 × 8`. Each GPU is **both** an EP rank for the MoE and
a TP/DP rank for attention.

```
2 PP stages (1 DGX node each, 8 GPUs/stage)
└── Within each stage, 8 GPUs:
    └── EP = 8 (each GPU holds 256/8 = 32 experts)
    └── For attention:  TP = 2,  attention-DP groups = 8/TP = 4
```

**MLA duplication tax:** TP shards the non-expert weights but **not** the
MLA latent KV or the DSA indexer K cache. KV/IDX caches replicate across
all TP ranks within a stage.

**BS convention:** throughout this report **BS means cluster-wide batch
size** — total concurrent requests across all 16 GPUs. With
`DP_attn = EP/TP = 4` independent attention-DP groups per stage, each
attention rank sees `BS/4` requests. Memory and TPOT scale with that
per-rank load. DP multiplies concurrency, not latency — TPOT per request
is the same regardless of how many other DP groups run in parallel.

---

## 4. MLA absorption — what kv_up is and why we drop it

### 4.1 Pre-absorption

```
kv_up = gemm(BS·DSA_ATTENDED, KV_LORA_RANK → KV_UP_OUT_DIM)
      = gemm(BS·2560, 512, 32768)
```

This GEMM decompresses every attended-token's latent KV into per-head
form (16384 K-dims + 16384 V-dims). Plus an attention core reading the
**decompressed** buffer (`BS·DSA_ATTENDED·KV_UP_OUT_DIM` bytes).

**Per-layer FP8 cost at BS=1 (TP=1):**

- kv_up: `gemm_ms(2560, 512, 32768) × 0.5`
  - FLOPs = 2·2560·512·32768 = 85.9 GFLOPs → compute time = 85.9 / (989.5·1024·0.5) × 1000 = **0.170 ms** (FP16)
  - Weight bytes = 512·32768·2 = 32 MB; act bytes = (2560·512 + 2560·32768)·2 = 170 MB; memory = 202/1384 × 1000 = 0.146 ms
  - max(0.170, 0.146) = 0.170 ms → ×0.5 FP8 = **0.085 ms**
- attn_core: read decompressed = 2560·32768·1 = 80 MB FP8 → **0.058 ms**
- o_proj: weight bytes 16384·7168·1 = 112 MB → **0.081 ms**
- residual: floor **0.015 ms**
- **block_c ≈ 0.235 ms** (model reports 0.2353 ms — matches)

### 4.2 MLA absorption (DeepSeek reference inference)

DeepSeek does **not** materialize the per-head K/V at inference. Instead:

1. **W^UK absorbed into Q**: each layer's `q_nope` projection is followed
   by a per-head multiplication with W^UK, mapping (BS·NUM_HEADS·QK_NOPE)
   into (BS·NUM_HEADS·KV_LORA_RANK). Attention dot products run in
   **latent space** against the compressed KV cache.
2. **W^UV applied on-the-fly to score@V_latent** before o_proj. Same
   per-head pattern. Output stays the same shape, so o_proj is unchanged.

This eliminates the kv_up GEMM. Memory traffic in attn_core collapses
because the only KV read is the **compressed latent**, which `kv_io`
already accounts for. The new overhead is two BS-bounded weight reads
(W^UK and W^UV), each 8.4 MB FP8 → both hit the HBM floor (0.015 ms).

**Per-layer FP8 cost at BS=1 (TP=1) with absorption:**

- kv_up: **0 ms**
- attn_core (latent dot products): compute-floor, **0.015 ms** (HBM floor)
- W^UK on-the-fly read: 128·128·512·1 = 8.4 MB → floor **0.015 ms**
- W^UV on-the-fly read: 128·512·128·1 = 8.4 MB → floor **0.015 ms**
- o_proj: unchanged **0.081 ms**
- residual: floor **0.015 ms**
- **block_c ≈ 0.139 ms** (model reports 0.1390 ms — matches)

### 4.3 Per-layer savings summary

| Piece | Pre-abs (FP8, BS=1) | Post-abs | Δ |
|---|---:|---:|---:|
| kv_up | 0.085 ms | 0 | **−0.085** |
| attn_core | 0.058 ms | 0.015 ms (floor) | **−0.043** |
| W^UK absorbed read | — | 0.015 ms | +0.015 |
| W^UV absorbed read | — | 0.015 ms | +0.015 |
| o_proj | 0.081 ms | 0.081 ms | 0 |
| residual | 0.015 ms | 0.015 ms | 0 |
| **block_c total** | **0.235 ms** | **0.139 ms** | **−0.096 ms** |

After TP=2 (block_c is TP-shardable): **−0.048 ms / layer / rank**.
Per stage (31 layers): **−1.49 ms / token**.

---

## 5. Memory accounting per rank


```
layers_per_stage = ceil(NUM_LAYERS / PP)              = ceil(61/2) = 31
kv_bytes_per_token  = KV_LORA_RANK + QK_ROPE          = 576 B (FP8)
idx_bytes_per_token = INDEXER_DIM                     = 128 B (FP8)

KV     = layers_per_stage · (BS/4) · seq_len · 576 B  (NOT divided by TP; BS/4 = per-rank requests)
IDX    = layers_per_stage · (BS/4) · seq_len · 128 B  (NOT divided by TP)
DenseW = layers_per_stage · NON_EXPERT_W_PER_LAYER_B / TP / 2     (FP8)
ExpW   = layers_per_stage · EXPERT_W_PER_LAYER_B      / EP / 2    (FP8)
```

Constants:

```
NON_EXPERT_W_PER_LAYER_B (FP16 bytes) = 2 × (
    HIDDEN·Q_LORA_RANK              # q_down
  + Q_LORA_RANK·NUM_HEADS·(QK_NOPE+QK_ROPE)  # q_up
  + HIDDEN·KV_LORA_RANK             # kv_down
  + KV_LORA_RANK·NUM_HEADS·(QK_NOPE+V_HEAD)  # kv_up (still counted; tiny weight)
  + NUM_HEADS·V_HEAD·HIDDEN         # o_proj
  + HIDDEN·INDEXER_DIM              # indexer K proj (128-dim, not 512)
  + HIDDEN·NUM_ROUTED_EXPERTS       # router gate
  + 3·HIDDEN·EXPERT_INTERMEDIATE    # shared expert
) ≈ 450.5 MB FP16

EXPERT_W_PER_LAYER_B (FP16 bytes) = NUM_ROUTED_EXPERTS · 3 · HIDDEN ·
    EXPERT_INTERMEDIATE · 2
  = 256 · 3 · 7168 · 2048 · 2
  ≈ 21.0 GB FP16
```

So at PP=2 TP=2 EP=8 FP8:

```
DenseW = 31 × 445.2 MB / 2 / 2 = 3.37 GB
ExpW   = 31 × 21.0  GB / 8 / 2 = 40.7 GB
```

Per-token cache footprint (per GPU, identical across all 16 GPUs):

```
KV  / token = 31 × 576 B =  17.4 kB
IDX / token = 31 × 128 B =   3.97 kB
total       =              21.4 kB / token / GPU per request
```

At sl=128K: **21.4 kB × 128·1024 = 2.81 GB / request / GPU** (was 4.32 GB
with the old 512-dim indexer; the smaller indexer K cache is ~4× lighter).

---

## 6. Compute model per piece (FP8 BS=4, HBM)


| Piece | Formula | Time (FP8, BS=4) |
|---|---|---:|
| `idx_io` (HBM) | `(BS/4)·sl·128 B / 1384 GB/s` | 0.015 ms (floor — was 0.045 ms at 512 B) |
| `kv_io` (HBM) | `(BS/4)·2560·576 B / 1384 GB/s` | 0.015 ms (floor) |
| `block_a` | `q_down + q_up + rope + kv_down + pre_norm` | 0.070 ms |
| `block_c` (absorbed) | as derived in §4 | 0.139 ms |
| `idx_comp` | `0.005 + max(0.005, 0.005·bs_rank)` | 0.010 ms |
| `moe_shardable` | norm + router + shared_exp + residual | 0.112 ms |
| `moe_expert_gemm` (EP=8) | grouped GEMM on E[unique]≈1 expert | 0.030 ms |
| `moe_ep_comms` | `2 × (nvlink_ms(8) + bw_ms)` | 0.080 ms |

Shardable subset (block_a, block_c, idx_comp, moe_shardable) is divided
by TP; expert_gemm and ep_comms are not.

---

## 7. Pipeline schedule

```
Stage 0 (layers 0..30):
    cum_io  = 0
    end_cmp = 0
    for layer in stage_0:
        cum_io  += total_io(layer)
        end_cmp  = max(end_cmp, cum_io) + total_compute(layer, tp=TP)

Stage 1 (layers 31..60):
    stage_io  = sum(total_io(c) for c in stage_1)         # full prefetch
    stage_cmp = sum(total_compute(c, tp=TP) for c in stage_1)
    end_cmp   = max(end_cmp, stage_io) + stage_cmp

TPOT = end_cmp
```

Each stage has its own independent HBM bus → stage-1's IO can prefetch
in parallel with stage-0's compute.

---

## 8. Memory budget per rank at sl=128K (HBM regime)

Budget = `80 GB × 90% − 3 GB scratch = 69 GB available` for
weights+cache. **BS = cluster-wide batch size** (each of the 4 DP groups
serves BS/4 requests).

| BS | KV | IDX | Dense W | Expert W | Scratch | **Total** | Fits? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
|   4 |  2.18 G |  0.48 G | 3.37 G | 40.7 G | 3.0 G | **49.7 G** | ✓ |
|   8 |  4.36 G |  0.97 G | 3.37 G | 40.7 G | 3.0 G | **52.4 G** | ✓ |
|  12 |  6.54 G |  1.45 G | 3.37 G | 40.7 G | 3.0 G | **55.0 G** | ✓ |
|  16 |  8.72 G |  1.94 G | 3.37 G | 40.7 G | 3.0 G | **57.7 G** | ✓ |
|  20 | 10.90 G |  2.42 G | 3.37 G | 40.7 G | 3.0 G | **60.4 G** | ✓ |
|  24 | 13.08 G |  2.91 G | 3.37 G | 40.7 G | 3.0 G | **63.0 G** | ✓ |
|  32 | 17.44 G |  3.88 G | 3.37 G | 40.7 G | 3.0 G | **68.4 G** | ✓ (just) |
|  48 | 26.16 G |  5.81 G | 3.37 G | 40.7 G | 3.0 G | **79.0 G** | ✗ |
|  64 | 34.88 G |  7.75 G | 3.37 G | 40.7 G | 3.0 G | **89.7 G** | ✗ |
|  96 | 52.31 G | 11.62 G | 3.37 G | 40.7 G | 3.0 G | **111.0 G** | ✗ |

KV+IDX grow as ~0.67 GB per unit of cluster BS (was 1.03 GB with the
old 512-dim indexer). The 24.9 GB cache budget now covers ~32 requests
cluster-wide. **Max-fittable BS in HBM regime = 32.**

---

## 9. Memory budget with KV on SSD

Moving the MLA KV cache to local NVMe SSD removes the `kv` term from
HBM. The DSA indexer K cache **stays in HBM** (it is hot-read every
layer on F layers and is much smaller than the KV in any case).

| BS | KV (SSD) | IDX | Dense W | Expert W | Scratch | **HBM total** | Fits? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
|   4 |  2.18 G |  0.48 G | 3.37 G | 40.7 G | 3.0 G | **47.5 G** | ✓ |
|   8 |  4.36 G |  0.97 G | 3.37 G | 40.7 G | 3.0 G | **48.0 G** | ✓ |
|  12 |  6.54 G |  1.45 G | 3.37 G | 40.7 G | 3.0 G | **48.5 G** | ✓ |
|  16 |  8.72 G |  1.94 G | 3.37 G | 40.7 G | 3.0 G | **49.0 G** | ✓ |
|  20 | 10.90 G |  2.42 G | 3.37 G | 40.7 G | 3.0 G | **49.5 G** | ✓ |
|  24 | 13.08 G |  2.91 G | 3.37 G | 40.7 G | 3.0 G | **50.0 G** | ✓ |
|  32 | 17.44 G |  3.88 G | 3.37 G | 40.7 G | 3.0 G | **50.9 G** | ✓ |
|  48 | 26.16 G |  5.81 G | 3.37 G | 40.7 G | 3.0 G | **52.9 G** | ✓ |
|  64 | 34.88 G |  7.75 G | 3.37 G | 40.7 G | 3.0 G | **54.8 G** | ✓ |
|  96 | 52.31 G | 11.62 G | 3.37 G | 40.7 G | 3.0 G | **58.7 G** | ✓ |
| 128 | 69.75 G | 15.50 G | 3.37 G | 40.7 G | 3.0 G | **62.6 G** | ✓ |
| 192 |104.62 G | 23.25 G | 3.37 G | 40.7 G | 3.0 G | **70.3 G** | ✓ (just) |
| 256 |139.50 G | 31.00 G | 3.37 G | 40.7 G | 3.0 G | **78.1 G** | ✗ |

The KV column shows SSD consumption — the GPU doesn't see this in HBM.
The new HBM bottleneck is the indexer K cache (only 0.12 GB/cluster-BS now);
the ceiling moves from **BS=32 (HBM)** to **BS=192 (SSD)** — a 6× lift in
batch capacity since the indexer cache shrunk 4× along with KV moving off HBM.

---

## 10. TPOT sweep — DSA, both regimes

**TPOT is per-request latency**.
Cluster tok/s = `BS / TPOT × 1000`.

### 10.1 HBM regime (KV + indexer both in HBM)

KV IOPS = (BS/4 · 2560) ops / kv\_io\_s at 576 B/op. HBM peak = 2,580 M/s.
Indexer K saturates HBM at 11,610 M/s (128 B/op) for BS≥8; floor-limited
at lower BS (8,738 M/s at BS=4 cluster).

| BS | TPOT (ms) | per-layer (ms) | KV IOPS | cluster tok/s | fits? |
|---:|---:|---:|---:|---:|:---:|
|   4 | 15.02 | 0.246 |  170.7 M | 266 | ✓ |
|   8 | 17.86 | 0.293 |  341.3 M | 448 | ✓ |
|  12 | 20.96 | 0.344 |  512.0 M | 572 | ✓ |
|  16 | 24.01 | 0.394 |  682.7 M | 666 | ✓ |
|  20 | 27.01 | 0.443 |  853.3 M | 740 | ✓ |
|  24 | 29.99 | 0.492 | 1,024 M | 800 | ✓ |
|  32 | **35.81** | **0.587** | **1,365 M** | **894** | ✓ (just) |
|  48 | 46.90 | 0.769 | 2,048 M | 1023 | ✗ |
|  64 | 57.37 | 0.940 | 2,580 M | 1115 | ✗ |
|  96 | 76.67 | 1.257 | 2,580 M | 1252 | ✗ |

KV reads in HBM are floor-limited up to BS=48 cluster, hitting peak HBM IOPS
(2,580 M/s) at BS=64+. The smaller indexer K (128 B/token vs 512 B) lifts
the HBM ceiling from BS=24 to **BS=32 cluster**.

**HBM peak fit-able throughput: BS=32 → 894 cluster tok/s.**
The **"20–30 ms TPOT" operating window corresponds to BS=12–24 cluster.**

### 10.2 KV-on-SSD regime, 28 GB/s (4× Gen4 RAID)

KV IOPS = (BS/4 · 2560) ops / kv\_io\_s at 576 B/op. SSD peak = 52.2 M/s.
The SSD is always at peak IOPS: floor-limited at BS=4 (51.2 M/s ≈ peak), BW-saturated at BS≥8 (exactly 52.2 M/s).

| BS | TPOT (ms) | per-layer (ms) | KV IOPS | cluster tok/s | fits? |
|---:|---:|---:|---:|---:|:---:|
|   4 |  15.06 | 0.247 | 51.2 M | 266 | ✓ |
|   8 |  17.94 | 0.294 | 52.2 M | 446 | ✓ |
|  12 |  21.09 | 0.346 | 52.2 M | 569 | ✓ |
|  16 |  24.19 | 0.397 | 52.2 M | 661 | ✓ |
|  20 |  27.24 | 0.447 | 52.2 M | 734 | ✓ |
|  24 |  30.27 | 0.496 | 52.2 M | 793 | ✓ |
|  32 |  36.18 | 0.593 | 52.2 M | 884 | ✓ |
|  48 |  47.48 | 0.778 | 52.2 M | 1011 | ✓ |
|  64 |  58.98 | 0.967 | 52.2 M | 1085 | ✓ |
|  96 |  83.71 | 1.372 | 52.2 M | 1147 | ✓ |
| 128 | 107.54 | 1.763 | 52.2 M | 1190 | ✓ |
| 192 | **153.30** | **2.513** | **52.2 M** | **1252** | ✓ (just) |

**SSD peak fit-able throughput: BS=192 → 1,252 cluster tok/s**, a 1.4×
gain over the HBM regime's max (894 at BS=32). The smaller indexer K
extends the SSD ceiling from BS=48 to BS=192 — KV+IDX in HBM together
now only consume ~30 GB at BS=192.

The **"20–30 ms TPOT" window corresponds to BS=12–24 cluster**.

### 10.3 Side-by-side TPOT comparison

| BS | HBM | SSD (28 GB/s) |
|---:|---:|---:|
|   4 |  15.02 ms |  15.06 ms |
|   8 |  17.86 ms |  17.94 ms |
|  12 |  20.96 ms |  21.09 ms |
|  16 |  24.01 ms |  24.19 ms |
|  20 |  27.01 ms |  27.24 ms |
|  24 |  29.99 ms |  30.27 ms |
|  32 |  35.81 ms |  36.18 ms |
|  48 | overflow  |  47.48 ms |
|  64 | overflow  |  58.98 ms |
|  96 | overflow  |  83.71 ms |
| 128 | overflow  | 107.54 ms |
| 192 | overflow  | 153.30 ms |

---

## 11. IndexCache comparison

IndexCache reuses the DSA indexer's top-k selection across adjacent
layers. The F:S:S:S pattern assigns one **F (Full)** layer for every
three **S (Shared)** layers. S layers skip the indexer K cache read
(`idx_io = 0`) and the indexer compute (`idx_comp = 0`), reusing the
most recent F layer's top-k selections instead. Out of 61 layers,
**16 are F and 45 are S**; per PP stage (31 layers): **8 F + 23 S**.

The S-layer savings come entirely from eliminating `idx_io` on 23/31
layers per stage. Everything else — KV gather, block_a, block_c, MoE —
is identical.

### 11.1 HBM regime

At small BS, IndexCache can be marginally **slower** than full DSA. On F
layers, `block_a` compute is hidden behind `idx_io`; on S layers that
overlap disappears and `block_a` becomes exposed latency. At higher BS
the indexer K reads grow (proportional to BS×sl), eventually dominating
and making IC's savings significant.

Per-layer = TPOT / 61. Per-FSSS = TPOT × 4 / 61 (latency of one F+S+S+S group, 4 layers).
**Peak FSSS KV IOPS** = (4 KV reads × (BS/4) × 2560 ops) ÷ (time to load 4 layers of KV).
Model: after the F-layer's top-k is computed, all 4 layers' KV are preloaded in one burst
before any attention compute — peak IOPS is measured during this preload window only.
HBM saturates at 2,580 M/s (peak HBM at 576 B/op); below saturation IOPS scales linearly
with BS while the 0.015 ms floor still gates the read.

| BS | DSA TPOT | DSA per-layer | IC TPOT | IC per-FSSS | Peak FSSS KV IOPS | speedup | fits? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
|   4 | 15.02 ms | 0.246 ms | 15.47 ms | 1.015 ms |   682.7 M | 0.97× | ✓ |
|   8 | 17.86 ms | 0.293 ms | 18.43 ms | 1.209 ms | 1,365.3 M | 0.97× | ✓ |
|  12 | 20.96 ms | 0.344 ms | 21.42 ms | 1.405 ms | 2,048.0 M | 0.98× | ✓ |
|  16 | 24.01 ms | 0.394 ms | 24.36 ms | 1.597 ms | **2,580 M** (peak) | 0.99× | ✓ |
|  20 | 27.01 ms | 0.443 ms | 27.25 ms | 1.787 ms | 2,580 M (peak) | 0.99× | ✓ |
|  24 | 29.99 ms | 0.492 ms | 30.11 ms | 1.975 ms | 2,580 M (peak) | 1.00× | ✓ |
|  32 | 35.81 ms | 0.587 ms | 35.70 ms | 2.341 ms | 2,580 M (peak) | 1.00× | ✓ |
|  48 | 46.90 ms | 0.769 ms | 46.36 ms | 3.040 ms | 2,580 M (peak) | 1.01× | ✗ |
|  64 | 57.37 ms | 0.940 ms | 56.33 ms | 3.694 ms | 2,580 M (peak) | 1.02× | ✗ |
|  96 | 76.67 ms | 1.257 ms | 74.39 ms | 4.878 ms | 2,580 M (peak) | 1.03× | ✗ |

The 4-layer combined KV preload crosses the floor→BW boundary at **BS=16 cluster** —
above that, the burst hits HBM's peak KV IOPS (2,580 M/s). IC's speedup
under the new 128-dim indexer is at most ~1.03× in HBM — the idx_io that
IC eliminates is now floor-limited (0.015 ms/layer) so there's even less
to save.

**In the HBM regime at sl=128K, IndexCache provides no meaningful
speedup** within the fit-able range (BS≤24). The indexer K read per
layer is only 0.045 ms — small relative to compute — and the block_a
overlap loss on S layers nearly cancels the idx_io savings. IC would
win at much longer sequences where idx_io becomes dominant.

### 11.2 KV-on-SSD regime, 28 GB/s

With KV on SSD, the SSD read cost per layer grows with BS and eventually
dominates the schedule. IC's savings on idx_io (HBM, ~0.045 ms/layer at
BS=4) become meaningful once SSD forces the schedule to run longer — the
saved idx_io ms subtract directly from TPOT.

Per-layer = TPOT / 61. Per-FSSS = TPOT × 4 / 61.
**Peak FSSS KV IOPS** = 4 KV reads ÷ time to load 4 layers of KV from SSD.
At 28 GB/s the 4-layer combined read is BW-saturated at every BS (5.9 MB at BS=4 already
takes 196 µs vs the 50 µs floor), so peak IOPS = SSD's peak at 576 B = **52.2 M/s constant**.

| BS | DSA TPOT | DSA per-layer | IC TPOT | IC per-FSSS | Peak FSSS KV IOPS | speedup | fits? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
|   4 |  15.06 ms | 0.247 ms | 15.51 ms | 1.017 ms | 52.2 M (peak) | 0.97× | ✓ |
|   8 |  17.94 ms | 0.294 ms | 18.51 ms | 1.214 ms | 52.2 M (peak) | 0.97× | ✓ |
|  12 |  21.09 ms | 0.346 ms | 21.55 ms | 1.413 ms | 52.2 M (peak) | 0.98× | ✓ |
|  16 |  24.19 ms | 0.397 ms | 24.54 ms | 1.609 ms | 52.2 M (peak) | 0.99× | ✓ |
|  20 |  27.24 ms | 0.447 ms | 27.48 ms | 1.802 ms | 52.2 M (peak) | 0.99× | ✓ |
|  24 |  30.27 ms | 0.496 ms | 30.39 ms | 1.993 ms | 52.2 M (peak) | 1.00× | ✓ |
|  32 |  36.18 ms | 0.593 ms | 36.08 ms | 2.366 ms | 52.2 M (peak) | 1.00× | ✓ |
|  48 |  47.48 ms | 0.778 ms | 46.93 ms | 3.077 ms | 52.2 M (peak) | 1.01× | ✓ |
|  64 |  58.98 ms | 0.967 ms | 57.10 ms | 3.744 ms | 52.2 M (peak) | **1.03×** | ✓ |
|  96 |  83.71 ms | 1.372 ms | 76.40 ms | 5.010 ms | 52.2 M (peak) | **1.10×** | ✓ |
| 128 | 107.54 ms | 1.763 ms | 97.42 ms | 6.388 ms | 52.2 M (peak) | **1.10×** | ✓ |
| 192 | **153.30 ms** | **2.513 ms** | **138.02 ms** | **9.050 ms** | 52.2 M (peak) | **1.11×** | ✓ (just) |

**With the 128-dim indexer, IndexCache's speedup in the SSD regime drops
from the old 1.24× peak to ~1.10× at high BS.** The idx_io savings shrunk
by 4× along with the indexer, so there is much less IO for IC to skip.
IC still pulls ahead modestly at very high BS (BS=96–192) where the
compounding savings across 45 S layers per token become visible:

- At BS=192 (SSD fit ceiling): DSA 153.30 ms → IC 138.02 ms = **1.11×**
- At BS=48 (old fit ceiling): DSA 47.48 ms → IC 46.93 ms = ~1.01×

IC's fit-able peak throughput: `192 / 138.02 × 1000 ≈ 1,391 cluster tok/s`
vs DSA's 1,252 tok/s.

### 11.3 Summary: DSA vs IC across regimes

| Regime | Fit-able BS range | IC wins? | Peak IC speedup |
|---|---|---|---:|
| HBM | BS=4–32 | No (0.97–1.00×) | 1.00× at BS=32 |
| KV-on-SSD 28 GB/s | BS=4–192 | Marginal at BS≥64 | **1.11× at BS=192** |

**Rule of thumb at sl=128K with the 128-dim indexer:** IndexCache offers
only marginal gains (≤1.11×). The 4× shrunk indexer K cache means there
is much less idx_io for IC to skip in the first place — at the floor
(0.015 ms/layer), the absolute savings are ≤0.015 ms × 23 S layers ≈
0.35 ms/stage, which is small relative to overall TPOT.

---

## 12. Takeaways

1. **Removing the kv_up GEMM saves ~9 % of TPOT at BS=4 cluster.** Across
   the sweep this savings holds roughly constant in absolute ms (1.5 ms / token)
   so the relative win shrinks at higher BS.
2. **The 128-dim indexer (vs 512-dim KV-LoRA) shrinks the indexer K cache 4×**
   — from 15.5 kB to 3.97 kB per token per stage. This lifts the HBM ceiling
   from BS=24 to **BS=32 cluster** and the SSD ceiling from BS=48 to
   **BS=192 cluster** (a 4× gain in batch capacity).
3. **HBM regime peak fit-able throughput: BS=32 → 894 cluster tok/s.**
4. **SSD regime peak fit-able throughput: BS=192 → 1,252 cluster tok/s**,
   beating HBM peak by 1.4× thanks to the 6× extra batch capacity.
5. **The "20–30 ms TPOT" target maps to BS=12–24 cluster-wide.**
   Both regimes produce nearly identical TPOT in this window.
6. **IndexCache does not help in the HBM regime at sl=128K** — IC is
   neutral to slightly negative (0.97–1.00×) within the fit-able range.
   The smaller indexer means idx_io is floor-limited (0.015 ms) and
   IC's S-layer savings nearly cancel block_a re-exposure.
7. **IndexCache provides only marginal gains in the KV-on-SSD regime**
   under the 128-dim indexer — peaking at ~1.11× at BS=192 (was 1.24×
   at BS=48 under the old 512-dim indexer). The 4× smaller indexer
   gives IC 4× less idx_io to skip.
8. **For higher BS at sl=128K**, deploy options are:
   (a) **KV-on-SSD with fast RAID** — best ROI: 1.4× throughput vs HBM peak;
   (b) FP4 expert quantization frees ~20 GB → more room for KV+IDX in HBM;
   (c) PP=4 halves layers/stage → halves cache/token but doubles GPU count;
   (d) Indexer-K offload would further raise the BS ceiling but breaks
       the producer/consumer model since indexer K is hot-read.

---

