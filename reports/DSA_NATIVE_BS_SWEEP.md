# DSA-Native (no IndexCache) Batch-Size Sweep at sl=128K

**Config:** PP=2, TP=2, EP=8, FP8, sl=128K, HBM-resident, all-F (DSA),
MLA absorption **enabled** (matches DeepSeek reference inference).

Source: `experiments/dsa_native_bs_sweep.py`, layer model in
`experiments/analytical_layer.py` (`absorb_mla=True`).

This report enumerates every variable, derives the memory and compute
costs from first principles, walks the pipeline schedule, and presents
the BS-sweep table.

---

## 1. Architecture & DSA constants

All values from `experiments/analytical_layer.py:72-96`.

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
all TP ranks within a stage (`experiments/analytical_layer.py:356-360`).

---

## 4. MLA absorption — what kv_up is and why we drop it

### 4.1 Pre-absorption (naive)

The model originally (and incorrectly, for inference) computed an
explicit kv-up GEMM in `block_c` at every layer:

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

Function: `per_rank_memory_bytes(pp, tp, ep, seq_len, bs, fp8)` at
`experiments/analytical_layer.py:352`. Formulas (FP8):

```
layers_per_stage = ceil(NUM_LAYERS / PP)              = ceil(61/2) = 31
kv_bytes_per_token = KV_LORA_RANK + QK_ROPE           = 576 B (FP8)

KV     = layers_per_stage · BS · seq_len · 576 B      (NOT divided by TP)
IDX    = layers_per_stage · BS · seq_len · 512 B      (NOT divided by TP)
DenseW = layers_per_stage · NON_EXPERT_W_PER_LAYER_B / TP / 2     (FP8)
ExpW   = layers_per_stage · EXPERT_W_PER_LAYER_B      / EP / 2    (FP8)
```

Constants from `analytical_layer.py:337-349`:

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

## 6. Compute model per piece (FP8 BS=1, HBM)

All other pieces are unchanged by absorption. Values from
`analytical_layer.py:213-329`.

| Piece | Formula | Time (FP8, BS=1) |
|---|---|---:|
| `idx_io` (HBM) | `BS·sl·512 B / 1384 GB/s` | 0.045 ms (sl=128K) |
| `kv_io` (HBM) | `BS·2560·576 B / 1384 GB/s` | 0.015 ms (floor) |
| `block_a` | `q_down + q_up + rope + kv_down + pre_norm` | 0.070 ms |
| `block_c` (absorbed) | as derived in §4 | 0.139 ms |
| `idx_comp` | `0.005 + max(0.005, 0.005·bs)` | 0.010 ms |
| `moe_shardable` | norm + router + shared_exp + residual | 0.112 ms |
| `moe_expert_gemm` (EP=8) | grouped GEMM on E[unique]≈1 expert | 0.030 ms |
| `moe_ep_comms` | `2 × (nvlink_ms(8) + bw_ms)` | 0.080 ms |

Shardable subset (block_a, block_c, idx_comp, moe_shardable) is divided
by TP; expert_gemm and ep_comms are not.

---

## 7. Pipeline schedule

Function: `schedule_pp` (`dsa_native_bs_sweep.py:48`). Same producer/
consumer template used across all reports in this project.

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

## 8. Memory budget per rank at sl=128K

Budget = `80 GB × 90% − 3 GB scratch = 69 GB available` for
weights+cache.

| BS | KV | IDX | Dense W | Expert W | Scratch | **Total** | Fits 72 GB? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
|  1 |  2.18 G |  1.94 G | 3.41 G | 40.7 G | 3.0 G | **51.2 G** | ✓ |
|  2 |  4.36 G |  3.88 G | 3.41 G | 40.7 G | 3.0 G | **55.3 G** | ✓ |
|  4 |  8.72 G |  7.75 G | 3.41 G | 40.7 G | 3.0 G | **63.6 G** | ✓ |
|  6 | 13.08 G | 11.62 G | 3.41 G | 40.7 G | 3.0 G | **71.8 G** | ✓ (just) |
|  8 | 17.44 G | 15.50 G | 3.41 G | 40.7 G | 3.0 G | **80.0 G** | ✗ |
| 12 | 26.16 G | 23.25 G | 3.41 G | 40.7 G | 3.0 G | **96.5 G** | ✗ |
| 16 | 34.88 G | 31.00 G | 3.41 G | 40.7 G | 3.0 G | **113.0 G** | ✗ |
| 24 | 52.31 G | 46.50 G | 3.41 G | 40.7 G | 3.0 G | **145.9 G** | ✗ |

**Max fittable BS at sl=128K = 6 per attention rank.** Notice that
KV+IDX scale together as 4.12 GB/BS (= 4.32 GB minus a small rounding;
the table uses GiB = 2³⁰ B). The 25 GB cache budget covers about
`25 / 4.12 ≈ 6` requests.

---

## 9. TPOT sweep results

All values from `experiments/dsa_native_bs_sweep.py`. Cluster
tokens/s = `DP_attn × BS / TPOT × 1000` with `DP_attn = EP/TP = 4`.

| BS | TPOT (ms) | tok/s / rank | cluster tok/s | fits? |
|---:|---:|---:|---:|:---:|
|  1 | 14.74 |  67.9 |   271.4 | ✓ |
|  2 | 17.93 | 111.6 |   446.2 | ✓ |
|  4 | 24.15 | 165.6 |   662.6 | ✓ |
|  6 | 30.19 | 198.7 |   794.9 | ✓ |
|  8 | 36.08 | 221.8 |   887.0 | ✗ |
| 12 | 47.31 | 253.6 | 1014.6 | ✗ |
| 16 | 57.91 | 276.3 | 1105.2 | ✗ |
| 24 | 77.49 | 309.7 | 1238.9 | ✗ |

**Throughput peaks at the memory wall: BS=6, ~795 cluster tok/s.**
Beyond BS=6 every additional request overflows HBM at sl=128K.

Compare to the pre-absorption model at BS=1 sl=128K (same config):
the older block_c took 0.2353 ms vs the new 0.1390 ms; per-stage savings
under TP=2 = (0.2353−0.1390)/2 × 31 = **1.49 ms / token**, taking BS=1
TPOT from ~16.2 ms (pre-absorption) down to 14.74 ms.

---

## 10. Takeaways

1. **Removing the kv_up GEMM saves ~9 % of TPOT at BS=1** at this
   config — non-trivial but not dramatic. The savings come from
   eliminating a (2560·512·32768) GEMM per layer plus a 80-MB
   decompressed-KV HBM read per layer. The absorbed-weight reads that
   replace them (W^UK, W^UV) are BS-bounded and hit the HBM floor.
2. **Memory is the bottleneck at sl=128K, not compute.** At BS=6 the
   per-rank footprint is already 71.8 GB out of 72 GB usable. There is
   no room to grow BS without an architectural change (longer PP,
   smaller seq_len, FP4 experts, or KV offload).
3. **The earlier "BS=85" hand-calc was off by 3.5×** because it divided
   the MLA latent KV by TP (it doesn't shard) and ignored the indexer K
   cache entirely. The real ceiling matching that math style is BS≈6 per
   attention rank, or ~24 concurrent requests across 4 attention-DP
   groups (still much less than 85).
4. **DP_attn = 4 is the only multiplier you get from this 16-GPU layout.**
   PP=2 splits layers; TP=2 splits non-expert weights but not KV/IDX. So
   concurrent batch capacity = 4 × BS_per_rank.
5. **For higher BS at sl=128K**, deploy options are: (a) FP4 expert
   quantization halves Expert W and frees ~20 GB → enables BS≈10;
   (b) PP=4 halves layers/stage → halves KV/IDX/token → enables BS≈12
   but doubles the GPU count; (c) KV-on-SSD (covered in
   `reports/KV_SSD_OFFLOAD_REPORT.md`) but pays the SSD bandwidth cost.

---

## 11. Validation

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
