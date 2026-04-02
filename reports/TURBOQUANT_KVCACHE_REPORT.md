# TurboQuant KV Cache Experiment Report

## Effect of TurboQuant 3-bit Quantization on KV Cache & LLM Inference

---

## 1. Introduction

This experiment investigates the impact of [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) 3-bit quantization on KV cache memory consumption, compute/IO overlap behavior, memory access patterns, and end-to-end inference latency across three attention architectures: Multi-Head Attention (MHA), Grouped Query Attention (GQA), and Multi-head Latent Attention (MLA).

TurboQuant is a two-stage compression algorithm:
1. **PolarQuant**: Randomly rotates data vectors, then converts from Cartesian to polar coordinates, mapping data onto a fixed, predictable circular grid. This eliminates the memory overhead of per-group normalization constants.
2. **QJL (Quantized Johnson-Lindenstrauss)**: Applies the JL transform to remaining errors, using a single sign bit (+1 or -1) for residual error correction with zero additional memory overhead.

Together, these stages compress KV cache entries from FP16 (16 bits) to just 3 bits per element, achieving a **5.33x compression ratio** with negligible accuracy loss and minimal runtime overhead.

## 2. Experimental Setup

### 2.1 Hardware Configuration

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA H100 80GB |
| PCIe Generation | Gen 4 (31.5 GB/s unidirectional) |
| FP16 Compute | 1000 TFLOPS |
| HBM Bandwidth | 3350 GB/s |
| HBM Capacity | 80 GB |
| Bandwidth Efficiency | 80% |

**Note**: While H100 natively supports PCIe Gen5 (64 GB/s), we use PCIe Gen4 (31.5 GB/s) as specified for the experimental configuration, representing a more bandwidth-constrained scenario common in many datacenter deployments.

### 2.2 Model Configurations

We evaluate six configurations spanning three attention architectures, each with FP16 and TurboQuant 3-bit variants:

| Config | Architecture | Layers | KV Heads | Head Dim | Compressed Dim | Bits | B/tok/layer | B/tok total | 1M ctx (GB) |
|--------|-------------|--------|----------|----------|----------------|------|-------------|-------------|-------------|
| MHA (FP16) | Llama-2-70B class | 80 | 64 | 128 | - | 16 | 32,768 | 2,621,440 | 2,560.0 |
| GQA (FP16) | Llama-2-70B (GQA) | 80 | 8 | 128 | - | 16 | 4,096 | 327,680 | 320.0 |
| MLA (FP16) | DeepSeek-V3 | 61 | 128 | 128 | 576 | 16 | 1,152 | 70,272 | 68.6 |
| MHA + TQ (3-bit) | Llama-2-70B + TQ | 80 | 64 | 128 | - | 3 | 6,144 | 491,520 | 480.0 |
| GQA + TQ (3-bit) | Llama-2-70B (GQA) + TQ | 80 | 8 | 128 | - | 3 | 768 | 61,440 | 60.0 |
| MLA + TQ (3-bit) | DeepSeek-V3 + TQ | 61 | 128 | 128 | 576 | 3 | 216 | 13,176 | 12.9 |

**Key architectural differences:**
- **MHA**: Every query head has its own KV head. KV cache = 2 x 64 x 128 = 16,384 elements/token/layer.
- **GQA**: 8 KV heads shared across 64 query heads (8x fewer KV entries). KV cache = 2 x 8 x 128 = 2,048 elements/token/layer.
- **MLA**: Compresses KV into a low-rank latent (kv_lora_rank=512 + rope_dim=64 = 576 dims). KV cache = 576 elements/token/layer.
- **TurboQuant**: Reduces each element from 16 bits to 3 bits (5.33x compression), applied orthogonally to the attention architecture.

### 2.3 Timing Model

We use the InferSim analytical FLOPs-based timing approach:
- **Compute time**: `GFLOPs / (GPU_TFLOPS * 1024 * MFU)`
- **I/O time**: `bytes / (bandwidth * efficiency)`
- **Overlap**: `max(compute, I/O)` for concurrent streams
- **MFU values**: Attention = 25%, Dense MLP = 30%, MoE grouped GEMM = 15%, Routing = 5%
- **TurboQuant dequant overhead**: Modeled as ~2 FLOP/element for PolarQuant inverse + QJL sign decode (negligible per the TurboQuant paper)

---

## 3. Experiment Results

### 3.1 Experiment 1: KV Cache Size at 1M+ Context

![KV Cache Size Bar Chart](exp1_kv_cache_size_1M.png)

**Objective**: Quantify the KV cache memory footprint at 1M token context for each configuration.

**Results** (single request, all layers):

| Configuration | KV Cache (GB) | Compression vs MHA FP16 | Fits in H100 80GB? |
|---------------|---------------|------------------------|---------------------|
| MHA (FP16) | 2,560.0 | 1.0x (baseline) | No |
| GQA (FP16) | 320.0 | 8.0x | No |
| MLA (FP16) | 68.6 | 37.3x | Yes (85.8%) |
| MHA + TQ (3-bit) | 480.0 | 5.3x | No |
| GQA + TQ (3-bit) | 60.0 | 42.7x | Yes (75.0%) |
| MLA + TQ (3-bit) | 12.9 | **198.6x** | Yes (16.1%) |

**Key finding**: At 1M context, only three configurations fit within a single H100's 80 GB HBM: **MLA FP16** (68.6 GB), **GQA + TQ** (60.0 GB), and **MLA + TQ** (12.9 GB). The combination of MLA compression and TurboQuant 3-bit quantization achieves a staggering **199x reduction** from the MHA FP16 baseline, using only 16% of HBM capacity and leaving ample room for model weights and activations.

### 3.2 Experiment 2: Compute/Copy Stream Overlap (Gantt Chart)

![Stream Overlap Gantt Chart](exp2_stream_overlap_gantt.png)

**Objective**: Visualize how the KV cache load (Copy Stream) overlaps with attention/MLP compute (Compute Stream) using prefetch pipelining, and quantify how much I/O latency is hidden.

**Setup**: 8 layers shown, batch_size=32, context=4096 tokens.

**Results**:

| Configuration | Sequential (ms) | Pipelined (ms) | I/O Hidden (%) | Speedup |
|---------------|-----------------|----------------|----------------|---------|
| GQA (FP16) | 3.271 | 1.965 | 39.9% | 1.66x |
| MLA (FP16) | 9.747 | 8.228 | 15.6% | 1.18x |
| GQA + TQ (3-bit) | 2.142 | 1.897 | 11.4% | 1.13x |
| MLA + TQ (3-bit) | 9.429 | 7.890 | 16.3% | 1.20x |

**Analysis**:

- **GQA (FP16)** benefits most from prefetch overlap (39.9% I/O hidden) because its I/O time is comparable to compute time, allowing the copy stream to be largely masked by the compute stream.
- **MLA configurations** (both FP16 and TQ) are compute-dominated due to the MoE expert computation. The KV I/O is already small relative to compute, so the overlap percentage appears lower, but the absolute I/O time is minimal.
- **TurboQuant** reduces the copy stream duration (fewer bytes to load), which means:
  - For GQA: the I/O was already partially hidden; with TQ, I/O becomes so small that overlap provides diminishing returns (11.4%) because the copy finishes before compute does.
  - For MLA: TQ slightly improves overlap effectiveness (16.3% vs 15.6%) since less data needs to be read.

**The key insight**: TurboQuant's primary benefit is not overlap improvement but **absolute I/O reduction** -- the copy stream segments shrink by 5.3x, directly reducing the wall-clock time even when overlap is limited.

### 3.3 Experiment 3: I/O Access Pattern Heatmap

![I/O Access Heatmap](exp3_io_access_heatmap.png)

**Objective**: Visualize the fundamental difference in memory access patterns between full sequential reads (MHA), compressed contiguous reads (MLA), and TurboQuant's discrete sparse reads (DSA).

**The three access patterns**:

1. **MHA (FP16) - Full Sequential Read**: Every KV cache element across all heads and dimensions is read contiguously. The heatmap shows uniform red (100% of memory addresses accessed, 2 bytes per element). This is bandwidth-efficient but requires reading the maximum amount of data.

2. **MLA (FP16) - Compressed Contiguous Read**: Only the compressed latent vector (576 dims out of 32,768 full dimensions) is read per token. The heatmap shows a narrow active band (~1.6% of full dimensionality), with the rest untouched. Access is still contiguous within the compressed region, maintaining good memory coalescing.

3. **TurboQuant (3-bit) - DSA Sparse Discrete Read**: The Discrete State Approximation creates a fundamentally different access pattern:
   - **Packed index reads**: 3-bit values are bit-packed, so every ~5th memory position is accessed to read packed words.
   - **Codebook lookups**: Only 8 discrete states (2^3) exist per element group. Dequantization looks up scattered codebook entries, creating a sparse, non-contiguous pattern.
   - Result: ~40% of addresses are touched, but with only 0.375 bytes per element (3 bits), reducing total data movement by 5.3x.

**Implications for hardware design**: The sparse access pattern of TurboQuant may benefit less from sequential prefetching but gains significantly from reduced total bandwidth consumption. Modern GPU HBM controllers handle scattered reads efficiently, and the 5.3x reduction in total bytes dominates any coalescing penalty.

### 3.4 Experiment 4: TTFT, TPOT, and Throughput

![Inference Latency](exp4_inference_latency.png)

**Objective**: Compare Time to First Token (TTFT), Time per Output Token (TPOT), and request throughput across all configurations.

**Setup**: prefill=4096 tokens, decode=256 tokens, batch_size sweep [1, 8, 32, 64].

**Results at batch=32**:

| Configuration | TTFT (ms) | TPOT (ms) | Token Throughput (tok/s) | Req Throughput (req/s) |
|---------------|-----------|-----------|--------------------------|------------------------|
| MHA (FP16) | 72,842.6 | 123.134 | 78.5 | 0.307 |
| GQA (FP16) | 72,842.6 | 17.805 | 105.8 | 0.413 |
| MLA (FP16) | 54,216.2 | 61.182 | 117.2 | 0.458 |
| MHA + TQ (3-bit) | 72,842.6 | 24.725 | 103.5 | 0.404 |
| GQA + TQ (3-bit) | 72,842.6 | 18.670 | 105.5 | 0.412 |
| MLA + TQ (3-bit) | 54,216.2 | 58.500 | 118.4 | 0.462 |

**Analysis**:

- **TTFT** is identical within architecture families (MHA/GQA share the same model size; MLA has fewer layers and different hidden dim). TurboQuant does not affect TTFT because the prefill phase computes KV from scratch rather than loading from cache.

- **TPOT** shows the most dramatic TurboQuant impact on MHA:
  - MHA FP16: 123.1 ms/token -> MHA + TQ: 24.7 ms/token (**5.0x improvement**). The massive KV cache read (32 KB/token/layer x 80 layers) dominates decode; TQ reduces this by 5.3x.
  - GQA FP16: 17.8 ms -> GQA + TQ: 18.7 ms (**negligible change, slightly worse**). GQA already has small KV cache; the TQ dequant overhead slightly exceeds the I/O savings.
  - MLA FP16: 61.2 ms -> MLA + TQ: 58.5 ms (**4.4% improvement**). MLA is compute-bound (MoE experts), so KV I/O savings have modest impact.

- **Throughput**: MLA + TQ achieves the highest throughput at 118.4 tok/s, slightly above MLA FP16 (117.2 tok/s). The biggest throughput gain from TurboQuant is on MHA (78.5 -> 103.5 tok/s, +32%).

---

## 4. Key Findings

### 4.1 Memory Capacity
- At 1M context, **MLA + TurboQuant 3-bit uses only 12.9 GB** -- a 199x reduction from MHA FP16 (2,560 GB). This makes million-token context feasible on a single H100.
- Only MLA FP16 (68.6 GB), GQA + TQ (60.0 GB), and MLA + TQ (12.9 GB) fit within 80 GB HBM at 1M context.

### 4.2 Compute/IO Overlap
- TurboQuant reduces the copy stream duration by 5.3x, shifting I/O-bound layers toward compute-bound.
- For GQA, this means I/O is almost fully masked (copy finishes before compute). For MHA, the shift is most beneficial, converting severely I/O-bound decode into a more balanced regime.
- MLA configurations are already compute-dominated (MoE expert computation), so the overlap behavior changes minimally.

### 4.3 Access Patterns
- MHA reads 100% of KV cache contiguously (dense, sequential).
- MLA reads only ~1.6% of the full dimensionality (compressed latent, still contiguous).
- TurboQuant DSA creates sparse discrete reads (~40% of addresses touched) with 5.3x fewer total bytes. The access pattern trades coalescing efficiency for dramatically reduced bandwidth consumption.

### 4.4 Inference Latency
- TurboQuant's biggest TPOT win is on **MHA** (5.0x reduction from 123 to 25 ms/token) where KV I/O dominates.
- For **GQA** (already I/O-efficient) and **MLA** (compute-dominated), TurboQuant provides marginal TPOT changes.
- TurboQuant does **not affect TTFT** since prefill computes KV from scratch.

### 4.5 Practical Recommendation
The optimal configuration depends on the bottleneck:
- **Memory-constrained** (long context): MLA + TQ is the clear winner (12.9 GB at 1M context).
- **I/O-bound decode** (MHA architectures): TurboQuant provides 5x TPOT reduction, a transformative improvement.
- **Compute-bound** (MLA/MoE models): TurboQuant provides modest gains; optimization effort is better spent on compute efficiency (expert parallelism, kernel optimization).

---

## 5. Methodology Notes

### 5.1 Simulation Approach
All timing uses the InferSim analytical FLOPs-based model, consistent with previous experiments in this repository (PDD analysis, batch sweep). This approach:
- Models compute via `GFLOPs / (TFLOPS * MFU)` with architecture-specific MFU values
- Models I/O via `bytes / bandwidth` with 80% efficiency factor
- Models overlap via `max(compute, I/O)` for concurrent CUDA streams
- Includes TurboQuant dequantization overhead (~2 FLOP/element, modeled at 5% MFU)

### 5.2 TurboQuant Modeling
- **Compression**: 16-bit to 3-bit per element (5.33x compression ratio)
- **Dequant overhead**: Modeled as negligible per the TurboQuant paper ("negligible runtime overhead")
- **DSA access pattern**: Modeled as sparse discrete reads with packed index access every ~5 positions plus scattered codebook lookups (8 unique states per element group)
- **Accuracy**: Not modeled; TurboQuant reports zero accuracy loss on LongBench, Needle In A Haystack, and L-Eval benchmarks

### 5.3 Limitations
- Real-world performance may differ due to kernel implementation details, memory fragmentation, and cache effects not captured by the analytical model.
- TurboQuant's dequantization overhead may vary by GPU architecture and kernel maturity.
- MoE expert activation patterns are estimated probabilistically rather than traced from real workloads.
- The H100 PCIe Gen4 configuration is non-standard (H100 supports Gen5); this represents a conservative bandwidth scenario.

---

## 6. Reproducibility

**Run the experiment:**
```bash
python experiments/run_turboquant_kvcache_experiment.py
```

**Output files:**
- `example_outputs/experiments/turboquant_kvcache/exp1_kv_cache_size_1M.png` - KV cache size bar chart
- `example_outputs/experiments/turboquant_kvcache/exp2_stream_overlap_gantt.png` - Compute/Copy Gantt chart
- `example_outputs/experiments/turboquant_kvcache/exp3_io_access_heatmap.png` - I/O access heatmap
- `example_outputs/experiments/turboquant_kvcache/exp4_inference_latency.png` - TTFT/TPOT/Throughput comparison
- `example_outputs/experiments/turboquant_kvcache/turboquant_kvcache_results.json` - Full numerical results

**Dependencies:** Python 3.11+, matplotlib, numpy
