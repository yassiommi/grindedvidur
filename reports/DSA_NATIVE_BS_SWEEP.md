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
| `MLA_KV_BYTES_PER_TOKEN` | 1152 (FP16) / **576 (FP8)** | (KV_LORA_RANK+QK_ROPE)·elem |
| `INDEXER_K_BYTES_PER_TOKEN` | 512 | KV_LORA_RANK·1B (FP8 always) |

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
| HBM peak IOPS (512 B/op, idx) | 2,902 M/s | 1384 GB/s ÷ 512 B |
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
kv_bytes_per_token = KV_LORA_RANK + QK_ROPE           = 576 B (FP8)

KV     = layers_per_stage · (BS/4) · seq_len · 576 B  (NOT divided by TP; BS/4 = per-rank requests)
IDX    = layers_per_stage · (BS/4) · seq_len · 512 B  (NOT divided by TP)
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
  + HIDDEN·KV_LORA_RANK             # indexer K proj
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
DenseW = 31 × 450.5 MB / 2 / 2 = 3.41 GB
ExpW   = 31 × 21.0  GB / 8 / 2 = 40.7 GB
```

Per-token cache footprint (per GPU, identical across all 16 GPUs):

```
KV  / token = 31 × 576 B = 17.4 kB
IDX / token = 31 × 512 B = 15.5 kB
total       = 32.9 kB / token / GPU per request
```

At sl=128K: **32.9 kB × 128·1024 = 4.32 GB / request / GPU**.

---

## 6. Compute model per piece (FP8 BS=4, HBM)


| Piece | Formula | Time (FP8, BS=4) |
|---|---|---:|
| `idx_io` (HBM) | `(BS/4)·sl·512 B / 1384 GB/s` | 0.045 ms (sl=128K) |
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
|   4 |  2.18 G |  1.94 G | 3.41 G | 40.7 G | 3.0 G | **51.2 G** | ✓ |
|   8 |  4.36 G |  3.88 G | 3.41 G | 40.7 G | 3.0 G | **55.3 G** | ✓ |
|  12 |  6.54 G |  5.81 G | 3.41 G | 40.7 G | 3.0 G | **59.4 G** | ✓ |
|  16 |  8.72 G |  7.75 G | 3.41 G | 40.7 G | 3.0 G | **63.6 G** | ✓ |
|  20 | 10.90 G |  9.69 G | 3.41 G | 40.7 G | 3.0 G | **67.7 G** | ✓ |
|  24 | 13.08 G | 11.62 G | 3.41 G | 40.7 G | 3.0 G | **71.8 G** | ✓ (just) |
|  32 | 17.44 G | 15.50 G | 3.41 G | 40.7 G | 3.0 G | **80.0 G** | ✗ |
|  48 | 26.16 G | 23.25 G | 3.41 G | 40.7 G | 3.0 G | **96.5 G** | ✗ |
|  64 | 34.88 G | 31.00 G | 3.41 G | 40.7 G | 3.0 G | **113.0 G** | ✗ |
|  96 | 52.31 G | 46.50 G | 3.41 G | 40.7 G | 3.0 G | **145.9 G** | ✗ |

KV+IDX grow as ~1.03 GB per unit of cluster BS. The 25 GB cache budget
covers ~24 requests cluster-wide. **Max-fittable BS in HBM regime = 24.**

---

## 9. Memory budget with KV on SSD

Moving the MLA KV cache to local NVMe SSD removes the `kv` term from
HBM. The DSA indexer K cache **stays in HBM** (it is hot-read every
layer on F layers and is much smaller than the KV in any case).

| BS | KV (SSD) | IDX | Dense W | Expert W | Scratch | **HBM total** | Fits? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
|   4 |  2.18 G |  1.94 G | 3.41 G | 40.7 G | 3.0 G | **49.0 G** | ✓ |
|   8 |  4.36 G |  3.88 G | 3.41 G | 40.7 G | 3.0 G | **51.0 G** | ✓ |
|  12 |  6.54 G |  5.81 G | 3.41 G | 40.7 G | 3.0 G | **52.9 G** | ✓ |
|  16 |  8.72 G |  7.75 G | 3.41 G | 40.7 G | 3.0 G | **54.8 G** | ✓ |
|  20 | 10.90 G |  9.69 G | 3.41 G | 40.7 G | 3.0 G | **56.8 G** | ✓ |
|  24 | 13.08 G | 11.62 G | 3.41 G | 40.7 G | 3.0 G | **58.7 G** | ✓ |
|  32 | 17.44 G | 15.50 G | 3.41 G | 40.7 G | 3.0 G | **62.6 G** | ✓ |
|  48 | 26.16 G | 23.25 G | 3.41 G | 40.7 G | 3.0 G | **70.3 G** | ✓ (just) |
|  64 | 34.88 G | 31.00 G | 3.41 G | 40.7 G | 3.0 G | **78.1 G** | ✗ |
|  96 | 52.31 G | 46.50 G | 3.41 G | 40.7 G | 3.0 G | **93.6 G** | ✗ |

The KV column shows SSD consumption — the GPU doesn't see this in HBM.
The new HBM bottleneck is the indexer K cache; the ceiling moves from
**BS=24 (HBM)** to **BS=48 (SSD)**.

---

## 10. TPOT sweep — DSA, both regimes

**TPOT is per-request latency**.
Cluster tok/s = `BS / TPOT × 1000`.

### 10.1 HBM regime (KV + indexer both in HBM)

KV IOPS = (BS/4 · 2560) ops / kv\_io\_s at 576 B/op. HBM peak = 2,580 M/s.
Indexer K always saturates HBM at 2,902 M/s (512 B/op, not shown — constant).

| BS | TPOT (ms) | per-layer (ms) | KV IOPS | cluster tok/s | fits? |
|---:|---:|---:|---:|---:|:---:|
|   4 | 14.74 | 0.242 |  170.7 M | 271 | ✓ |
|   8 | 17.93 | 0.294 |  341.3 M | 446 | ✓ |
|  12 | 21.06 | 0.345 |  512.0 M | 570 | ✓ |
|  16 | 24.15 | 0.396 |  682.7 M | 663 | ✓ |
|  20 | 27.18 | 0.446 |  853.3 M | 736 | ✓ |
|  24 | **30.19** | **0.495** | **1,024 M** | **795** | ✓ |
|  32 | 36.08 | 0.591 | 1,365 M |  887 | ✗ |
|  48 | 47.31 | 0.776 | 2,048 M | 1015 | ✗ |

KV reads in HBM are **always floor-limited** at sl=128K — even BS=48 cluster (12/rank) only
reaches 2,048 M/s vs peak 2,580 M/s. The floor (0.015 ms) masks small KV payloads.

**HBM peak fit-able throughput: BS=24 → 795 cluster tok/s.**
The **"20–30 ms TPOT" operating window corresponds to BS=12–20 cluster.**

### 10.2 KV-on-SSD regime, 28 GB/s (4× Gen4 RAID)

KV IOPS = (BS/4 · 2560) ops / kv\_io\_s at 576 B/op. SSD peak = 52.2 M/s.
The SSD is always at peak IOPS: floor-limited at BS=4 (51.2 M/s ≈ peak), BW-saturated at BS≥8 (exactly 52.2 M/s).

| BS | TPOT (ms) | per-layer (ms) | KV IOPS | cluster tok/s | fits? |
|---:|---:|---:|---:|---:|:---:|
|   4 |  14.77 | 0.242 | 51.2 M | 271 | ✓ |
|   8 |  18.01 | 0.295 | 52.2 M | 444 | ✓ |
|  12 |  21.20 | 0.347 | 52.2 M | 566 | ✓ |
|  16 |  24.33 | 0.399 | 52.2 M | 658 | ✓ |
|  20 |  28.29 | 0.464 | 52.2 M | 707 | ✓ |
|  24 |  32.72 | 0.536 | 52.2 M | 734 | ✓ |
|  32 |  41.51 | 0.680 | 52.2 M | 771 | ✓ |
|  48 | **58.80** | **0.964** | **52.2 M** | **816** | ✓ |
|  64 |  75.78 | 1.242 | 52.2 M | 845 | ✗ |
|  96 | 108.90 | 1.785 | 52.2 M | 882 | ✗ |

**SSD peak fit-able throughput: BS=48 → 816 cluster tok/s**, beating
the HBM regime's max (795) by doubling batch capacity.

The **"20–30 ms TPOT" window corresponds to BS=12–20 cluster**.

### 10.3 Side-by-side TPOT comparison

| BS | HBM | SSD (28 GB/s) |
|---:|---:|---:|
|   4 |  14.74 ms |  14.77 ms |
|   8 |  17.93 ms |  18.01 ms |
|  12 |  21.06 ms |  21.20 ms |
|  16 |  24.15 ms |  24.33 ms |
|  20 |  27.18 ms |  28.29 ms |
|  24 |  30.19 ms |  32.72 ms |
|  32 | overflow  |  41.51 ms |
|  48 | overflow  |  58.80 ms |
|  64 | overflow  |  overflow |

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
**FSSS IOPS** = ops in one FSSS group ÷ per-FSSS wall time. For an FSSS group there are
4 KV reads (every layer) but only **1 idx read** (F only) — so IC drives ~1/4 the idx
IOPS that DSA would over an equivalent 4-layer window.

| BS | DSA TPOT | DSA per-layer | IC TPOT | IC per-FSSS | FSSS KV IOPS | FSSS idx IOPS | speedup | fits? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|   4 | 14.74 ms | 0.242 ms | 15.42 ms | 1.011 ms |  10.1 M | 129.6 M | 0.96× | ✓ |
|   8 | 17.93 ms | 0.294 ms | 18.50 ms | 1.213 ms |  16.9 M | 216.1 M | 0.97× | ✓ |
|  12 | 21.06 ms | 0.345 ms | 21.52 ms | 1.411 ms |  21.8 M | 278.6 M | 0.98× | ✓ |
|  16 | 24.15 ms | 0.396 ms | 24.50 ms | 1.606 ms |  25.5 M | 326.4 M | 0.99× | ✓ |
|  20 | 27.18 ms | 0.446 ms | 27.42 ms | 1.798 ms |  28.5 M | 364.5 M | 0.99× | ✓ |
|  24 | 30.19 ms | 0.495 ms | 30.32 ms | 1.988 ms |  30.9 M | 395.6 M | 1.00× | ✓ |
|  32 | 36.08 ms | 0.591 ms | 35.98 ms | 2.359 ms |  34.7 M | 444.5 M | 1.00× | ✗ |
|  48 | 47.31 ms | 0.776 ms | 46.76 ms | 3.066 ms |  40.1 M | 512.9 M | 1.01× | ✗ |
|  64 | 57.91 ms | 0.950 ms | 56.87 ms | 3.729 ms |  43.9 M | 562.4 M | 1.02× | ✗ |
|  96 | 77.49 ms | 1.270 ms | 75.20 ms | 4.931 ms |  49.8 M | 637.9 M | 1.03× | ✗ |

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
**FSSS KV IOPS** averages over the FSSS wall time; the SSD itself still bursts at peak
52.2 M/s during each KV read (4 reads per cycle). **FSSS idx IOPS** is on HBM (idx never
leaves HBM in this regime) — IC issues 1 idx read per cycle vs DSA's 4.

| BS | DSA TPOT | DSA per-layer | IC TPOT | IC per-FSSS | FSSS KV IOPS | FSSS idx IOPS | speedup | fits? |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|   4 |  14.77 ms | 0.242 ms | 15.46 ms | 1.013 ms |  10.1 M | 129.3 M | 0.96× | ✓ |
|   8 |  18.01 ms | 0.295 ms | 18.58 ms | 1.219 ms |  16.8 M | 215.1 M | 0.97× | ✓ |
|  12 |  21.20 ms | 0.347 ms | 21.66 ms | 1.420 ms |  21.6 M | 276.9 M | 0.98× | ✓ |
|  16 |  24.33 ms | 0.399 ms | 24.68 ms | 1.618 ms |  25.3 M | 324.0 M | 0.99× | ✓ |
|  20 |  28.29 ms | 0.464 ms | 27.65 ms | 1.813 ms |  28.2 M | 361.5 M | **1.02×** | ✓ |
|  24 |  32.72 ms | 0.536 ms | 30.60 ms | 2.006 ms |  30.6 M | 392.0 M | **1.07×** | ✓ |
|  32 |  41.51 ms | 0.680 ms | 36.35 ms | 2.384 ms |  34.4 M | 439.9 M | **1.14×** | ✓ |
|  48 |  58.80 ms | 0.964 ms | 47.34 ms | 3.104 ms |  39.6 M | 506.7 M | **1.24×** | ✓ |
|  64 |  75.78 ms | 1.242 ms | 58.91 ms | 3.863 ms |  42.4 M | 542.9 M | 1.29× | ✗ |
|  96 | 108.90 ms | 1.785 ms | 82.91 ms | 5.437 ms |  45.2 M | 578.6 M | 1.31× | ✗ |

**In the KV-on-SSD regime, IndexCache wins meaningfully at BS≥20.**
At the SSD fit ceiling (BS=48), IC delivers **1.24× speedup**
(47.34 ms vs 58.80 ms). IC's fit-able peak throughput is
`48 / 47.34 × 1000 ≈ 1014 cluster tok/s` vs DSA's 816 tok/s.

### 11.3 Summary: DSA vs IC across regimes

| Regime | Fit-able BS range | IC wins? | Peak IC speedup |
|---|---|---|---:|
| HBM | BS=4–24 | No (0.96–1.00×) | 1.00× at BS=24 |
| KV-on-SSD 28 GB/s | BS=4–48 | Yes at BS≥20 | **1.24× at BS=48** |

**Rule of thumb at sl=128K:** IndexCache pays off only in the SSD regime
at BS≥20 cluster. In pure HBM mode the crossover is above the fit-able
range and IC is not beneficial.

---

## 12. Takeaways

1. **Removing the kv_up GEMM saves ~9 % of TPOT at BS=4 cluster.** Across
   the sweep this savings holds roughly constant in absolute ms (1.5 ms / token)
   so the relative win shrinks at higher BS.
2. **In the HBM regime, memory caps BS at 24 cluster**, giving 795 cluster tok/s.
3. **Moving KV to SSD doubles the BS ceiling to 48 cluster** because the
   KV leaves HBM entirely. The new HBM bottleneck is the indexer K cache.
4. **On a 4× Gen4 NVMe RAID (28 GB/s)**, the SSD regime *beats* the HBM
   regime at peak: **816 cluster tok/s at BS=48 vs 795 at BS=24**.
5. **The "20–30 ms TPOT" target maps to BS=12–20 cluster-wide.**
   Both regimes produce nearly identical TPOT in this window.
6. **IndexCache does not help in the HBM regime at sl=128K** — IC is
   neutral to slightly negative (0.96–1.00×) within the fit-able range.
7. **IndexCache pays off in the KV-on-SSD regime at BS≥20**, reaching
   **1.24× speedup at BS=48**. Cluster throughput: DSA peaks at
   816 tok/s; IC peaks at ~1014 tok/s at the same BS=48.
8. **For higher BS at sl=128K**, deploy options are:
   (a) **KV-on-SSD with fast RAID + IndexCache** — best ROI: ~1.24× at BS=48;
   (b) FP4 expert quantization frees ~20 GB → enables BS≈40 in HBM;
   (c) PP=4 halves layers/stage → halves cache/token but doubles GPU count;
   (d) Indexer-K offload would further raise the BS ceiling but breaks
       the producer/consumer model since indexer K is hot-read.

---

## 13. Validation

All 11 identity checks in `experiments/validate_pp2_tp.py` continue to
pass under `absorb_mla=True` (the new default). The hand-derived values
in the validator cover IO bytes/times, memory bytes, and the pipeline
schedule walk — none of which change under absorption. The validator's
`total_compute_ms` hand-reconstruction self-consistently uses the
model's reported `block_c_ms`, so it tracks the new value automatically.

Reproduce with:

```
python -m experiments.dsa_native_bs_sweep
python -m experiments.validate_pp2_tp
```
