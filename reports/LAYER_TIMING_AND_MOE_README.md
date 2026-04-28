# Layer-Level Timing, GPU Prefetching, and MoE/Sparse Model Support

This document describes the extensions to InferLens that add (1) per-layer execution time breakdowns with GPU-initiated KV cache prefetching, and (2) support for Mixture-of-Experts (MoE) and sparse attention models including DeepSeek-V3.

## Table of Contents

- [Overview](#overview)
- [Feature 1: Layer-Level Timing and GPU Prefetching](#feature-1-layer-level-timing-and-gpu-prefetching)
  - [Background](#background)
  - [Per-Layer Execution Breakdown](#per-layer-execution-breakdown)
  - [GPU-Initiated KV Cache Prefetching](#gpu-initiated-kv-cache-prefetching)
  - [Gantt-Style Visualization](#gantt-style-visualization)
- [Feature 2: MoE and Sparse Model Support](#feature-2-moe-and-sparse-model-support)
  - [Background on MoE](#background-on-moe)
  - [DeepSeek-V3 and Multi-head Latent Attention](#deepseek-v3-and-multi-head-latent-attention)
  - [Supported MoE Models](#supported-moe-models)
  - [MoE Execution Time Prediction](#moe-execution-time-prediction)
  - [Expert Parallelism](#expert-parallelism)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Integration with InferSim](#integration-with-infersim)

---

## Overview

Standard LLM inference simulators model execution at the batch or pipeline-stage level, treating all layers as identical and aggregating their costs. This extension adds two capabilities:

1. **Layer-level granularity**: Each transformer layer reports its own compute, I/O (memory bandwidth), and communication time independently. This enables identification of per-layer bottlenecks and supports optimizations like GPU-initiated KV cache prefetching.

2. **MoE/Sparse model support**: Models like DeepSeek-V3 (256 experts, MLA attention) and Mixtral use sparse Mixture-of-Experts layers where only a subset of parameters are activated per token. This fundamentally changes the compute/I/O balance and requires modeling expert routing, grouped GEMM, expert weight loading, and expert parallelism communication.

Both features use InferSim's FLOPs-based simulation methodology for MoE timing estimation.

---

## Feature 1: Layer-Level Timing and GPU Prefetching

### Background

In transformer inference, each layer performs:
- **Attention**: QKV projection, RoPE, attention core (QK^T softmax V), output projection
- **FFN/MoE**: Up projection, activation, down projection (or expert routing + grouped GEMM)
- **Communication**: Tensor parallel all-reduce (2x per layer), pipeline parallel send/recv

These operations are bounded by different resources:
- **Compute-bound**: GEMM operations (projections, expert compute) limited by GPU FLOPS
- **I/O-bound**: KV cache loading bounded by PCIe bandwidth; expert weight loading bounded by HBM bandwidth
- **Communication-bound**: All-reduce, expert dispatch/combine limited by NVLink/RDMA

### How I/O is Simulated

KV cache I/O is modeled as a PCIe transfer — the KV cache for the next layer is prefetched from host memory to GPU memory over the PCIe bus:

```
kv_load_time_per_layer = kv_bytes / PCIe_bandwidth
```

where:
- `kv_bytes = kv_bytes_per_token * avg_kv_length * decode_batch_size`
- MHA/GQA: `kv_bytes_per_token = 2 * num_kv_heads * head_dim * bytes_per_element`
- MLA (DeepSeek-V3): `kv_bytes_per_token = (kv_lora_rank + qk_rope_head_dim) * bytes_per_element`
- `PCIe_bandwidth = pcie_bandwidth_gb_per_s * 0.80` (80% efficiency)
- PCIe bandwidth values from `device_sku_config` (A40: 31.5 GB/s PCIe4 x16, A100: 31.5 GB/s, H100: 64 GB/s PCIe5 x16)

Expert weight loading (MoE) uses HBM bandwidth instead, since expert weights are already in GPU memory:
- `expert_load_time = expert_params * local_experts / (HBM_bandwidth * 0.80)`

InferLens's sklearn-based predictor trains on real GPU kernel traces, so `attention_decode_execution_time` already implicitly includes KV cache loading from the profiled kernel's wall-clock time. The PCIe-based `kv_load_time` is used for:
- Populating `LayerExecutionTime.kv_cache_load_time` in the per-layer Gantt visualization
- Computing KV prefetch overlap savings: `overlap = min(current_compute, next_kv_load)`

It is **not** added to the base `model_time` (which uses profiled values), avoiding double-counting. When prefetch is enabled, `model_time = base_profiled_time - total_prefetch_savings`.

### Per-Layer Execution Breakdown

Each layer now tracks via `LayerExecutionTime`:

| Component | Description | Bound |
|-----------|-------------|-------|
| `attention_compute_time` | All attention ops (projections + core + norm) | Compute |
| `mlp_compute_time` | FFN/MoE compute (projections or expert GEMM) | Compute |
| `kv_cache_load_time` | KV cache transfer time from host via PCIe (for Gantt/prefetch) | I/O (PCIe) |
| `weight_load_time` | Loading expert weights from HBM (MoE only) | I/O (HBM) |
| `tensor_parallel_comm_time` | TP all-reduce communication | Communication |
| `expert_parallel_comm_time` | EP dispatch/combine (MoE only) | Communication |
| `prefetch_overlap_savings` | Time saved by KV prefetching | Optimization |

### GPU-Initiated KV Cache Prefetching

During decode, loading the KV cache for attention is a major I/O bottleneck. GPU-initiated prefetching overlaps this I/O with computation:

```
Without prefetch:
Layer N:  [===Compute===][===KV Load===][===Comm===]
Layer N+1:                                           [===Compute===][===KV Load===][===Comm===]

With prefetch:
Layer N:  [===Compute===][===Comm===]
          [--KV Prefetch N+1--]        <- DMA engine
Layer N+1:                     [===Compute===][=Remaining KV=][===Comm===]
```

**Rules:**
- At the start of layer N, a DMA transfer begins prefetching KV cache for layer N+1
- Layer N+1 cannot start computing until all I/O and comm from layer N completes
- Overlap = min(current_layer_compute_time, next_layer_kv_cache_load_time)
- The last layer gets no prefetch benefit (no layer N+1)

**Enable with:** `--replica_config_enable_kv_prefetch`

### Gantt-Style Visualization

When `--metrics_config_store_layer_metrics` is set, the simulator produces:

1. **Per-batch timeline Gantt** (`plots/layer_gantt_batch_N.png`): Each layer is one row. X-axis is wall-clock time (ms). Three hardware streams are drawn as overlapping bars within each row:
   - **SM (green)**: Compute — dark green for attention, light green for MLP/MoE
   - **DMA (blue)**: KV cache PCIe transfer — drawn below compute, temporally overlapping
   - **NCCL (orange)**: All-reduce communication — drawn above compute, starts after SM finishes

   When prefetch is enabled, the DMA bar for layer N starts at the same time as layer N's compute, showing the I/O overlap. The layer cadence is determined by `max(compute + comm, dma_remaining)`.

2. **Summary plot** (`plots/layer_timing_summary.png`): Stacked bar chart averaging all batches, showing the time distribution across layers.

3. **Raw data** (`layer_timings.json`, `layer_timings.csv`): Full per-layer timing data including per-stream start/end offsets for custom analysis.

---

## Feature 2: MoE and Sparse Model Support

### Background on MoE

Mixture-of-Experts (MoE) replaces the dense FFN in each transformer layer with a set of "expert" sub-networks. A gating/routing network selects which experts process each token:

```
Input token -> Router -> Top-K experts selected -> Experts compute in parallel -> Combine outputs
```

Key parameters:
- **num_routed_experts**: Total experts (e.g., 256 for DeepSeek-V3)
- **num_experts_per_tok**: Active experts per token (e.g., 8)
- **num_shared_experts**: Experts applied to all tokens (e.g., 1)

MoE changes the performance characteristics:
- **Compute**: Only K out of N experts fire, reducing per-token FLOPs
- **I/O**: All expert weights must reside in HBM, creating memory pressure. Weight loading can become the bottleneck.
- **Communication**: With Expert Parallelism, tokens must be dispatched to expert-owning GPUs and results combined back

### DeepSeek-V3 and Multi-head Latent Attention

DeepSeek-V3 uses two key architectural innovations:

**Multi-head Latent Attention (MLA):**
Instead of standard multi-head attention with separate K/V projections, MLA compresses KV into a low-rank latent space:
- KV down-projection: `hidden_size -> kv_lora_rank` (512 for DeepSeek-V3)
- Much smaller KV cache: Only stores compressed latent + RoPE component
- Absorbed attention: Fuses K/V up-projections into attention weights

This reduces KV cache size by ~10x compared to standard MHA with the same number of heads.

**Sparse MoE with 256 experts:**
- 256 routed experts + 1 shared expert per layer
- Top-8 routing: each token activates 8 of 256 experts
- Expert intermediate size: 2048 (much smaller than dense FFN)
- Supports FP8 quantization for both GEMM and KV cache

### Supported MoE Models

| Model | Experts | Active/Token | Shared | Attention | Config Name |
|-------|---------|-------------|--------|-----------|-------------|
| DeepSeek-V3 | 256 | 8 | 1 | MLA | `deepseek-ai/DeepSeek-V3` |
| Qwen3-30B-A3B | 128 | 8 | 1 | GQA | `Qwen/Qwen3-30B-A3B` |
| Mixtral-8x7B | 8 | 2 | 0 | GQA | `mistralai/Mixtral-8x7B-v0.1` |

### MoE Execution Time Prediction

Following InferSim's methodology, MoE layer time is computed as:

```
MoE_time = max(expert_compute_time, expert_load_time) + shared_expert_time + routing_time

Where:
  expert_compute_time = FLOPs / (GPU_TFLOPS * MFU)
  FLOPs = 3 * hidden_size * intermediate_size * num_experts_per_tok * batch_size
  expert_load_time = expert_params * local_experts / memory_bandwidth
  routing_time = router_GEMM_time (hidden -> num_experts)
```

The `max()` reflects that expert computation and weight loading from HBM can be overlapped (compute is bound by tensor cores, loading by memory controller).

### Expert Parallelism

For large MoE models (256+ experts), Expert Parallelism (EP) distributes experts across GPUs:
- Each GPU owns `num_experts / EP_size` experts
- **Dispatch**: Router sends each token to the GPUs owning its selected experts
- **Combine**: Expert outputs are gathered back to the original token's GPU
- Communication uses NVLink (intra-node) or RDMA (inter-node)

DeepEP is the optimized implementation used for DeepSeek-V3 at scale (128 GPUs).

---

## Configuration

### New Config Fields

**ReplicaConfig:**
```
--replica_config_enable_kv_prefetch          # GPU-initiated KV prefetching (flag, no value)
--replica_config_expert_parallel_size N      # Expert parallel degree
--replica_config_model_name "deepseek-ai/DeepSeek-V3"  # MoE model
```

**MetricsConfig:**
```
--metrics_config_store_layer_metrics         # Enable per-layer Gantt plots (flag, no value)
```

Note: Boolean flags use `--flag` to enable and `--no-flag` to disable (no `true`/`false` value).

**Model Config (BaseModelConfig) new fields:**
- `is_moe`: Whether the model uses MoE
- `num_routed_experts`: Total number of routed experts
- `num_experts_per_tok`: Active experts per token
- `num_shared_experts`: Shared experts (applied to all tokens)
- `moe_intermediate_size`: Expert FFN hidden dimension
- `attention_type`: "MHA", "GQA", or "MLA"
- MLA-specific: `kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`

---

## Usage Examples

### Dense model with layer timing and prefetching
```bash
python -m vidur.main \
    --replica_config_model_name "meta-llama/Llama-2-7b-hf" \
    --replica_config_device a100 \
    --replica_config_enable_kv_prefetch \
    --metrics_config_store_layer_metrics
```

### DeepSeek-V3 MoE simulation
```bash
python -m vidur.main \
    --replica_config_model_name "deepseek-ai/DeepSeek-V3" \
    --replica_config_device a100 \
    --replica_config_network_device a100_dgx \
    --replica_config_tensor_parallel_size 8 \
    --replica_config_expert_parallel_size 8 \
    --replica_config_enable_kv_prefetch \
    --metrics_config_store_layer_metrics
```

Note: TP=8 requires `a100_dgx` network device (DGX has 8-GPU NVSwitch topology).
MoE models use profiling data from a similar dense model (e.g., Meta-Llama-3-70B for DeepSeek-V3)
with MoE-specific timing computed analytically.

### Mixtral-8x7B
```bash
python -m vidur.main \
    --replica_config_model_name "mistralai/Mixtral-8x7B-v0.1" \
    --replica_config_device a100 \
    --replica_config_tensor_parallel_size 2 \
    --metrics_config_store_layer_metrics
```

---

## Example Outputs

The `../example_outputs/` directory contains actual simulation results from running the examples above.

### Llama-2-7b (Dense, A100, TP=1, Prefetch=ON)

**Command:**
```bash
python -m vidur.main \
    --replica_config_model_name "meta-llama/Llama-2-7b-hf" \
    --replica_config_device a100 \
    --replica_config_enable_kv_prefetch \
    --metrics_config_store_layer_metrics
```

**Request-Level Results** (128 requests, 2048 prefill + 512 decode tokens each):

| Metric | Value |
|--------|-------|
| Simulation time | 235.08s |
| Mean E2E latency | 337.4 ms |
| Mean TTFT (time to first token) | 139.3 ms |
| Mean TPOT (time per output token) | 0.39 ms/token |
| P99 E2E latency | 368.0 ms |

**Per-Layer Timing** (averaged across decode batches):

| Component | Avg Time (ms) | Description |
|-----------|--------------|-------------|
| `attention_compute_time` | 0.14 | Attention projections + core |
| `mlp_compute_time` | 0.18 | FFN up/down/gate projections |
| `kv_cache_load_time` | 1.58 | KV cache PCIe transfer (31.5 GB/s) |
| `tensor_parallel_comm_time` | 0.00 | No TP (single GPU) |
| `prefetch_overlap_savings` | 0.31 | DMA overlap with compute |
| **total_time** | **1.59** | Per-layer decode step |

**Gantt Charts** (`../example_outputs/llama_2_7b/`):

The timeline Gantt shows each layer as a row with overlapping stream bars:
- `layer_gantt_batch_0_prefill.png`: Prefill batch — all 32 layers show only compute (dark green = attention, light green = MLP). With TP=1, no communication bars. Time progresses left-to-right, ~1.14 ms per layer.
- `layer_gantt_batch_4_decode.png`: Decode batch — now I/O (blue, KV DMA over PCIe) is visible overlapping with compute. The blue bars extend past the green compute, showing the system is **I/O-bound**. The DMA prefetch for layer N+1 runs concurrently with layer N's compute. Layer 31 has no I/O bar (last layer, nothing to prefetch).

**Summary Plot** (`../example_outputs/llama_2_7b/layer_timing_summary.png`):

The stacked bar summary (averaged across all batches including decode) shows that Llama-2-7b on a single A100 is I/O-bound during decode: KV cache loading from host via PCIe (blue, ~1.58 ms) dominates over compute (green, ~0.33 ms). The red dashed prefetch savings line shows ~0.31 ms saved per layer by overlapping KV DMA with compute. Layer 0 has higher effective I/O since it gets no prefetch benefit (no preceding layer to overlap with).

### DeepSeek-V3 (MoE, A100 DGX, TP=8, EP=8, Prefetch=ON)

**Command:**
```bash
python -m vidur.main \
    --replica_config_model_name "deepseek-ai/DeepSeek-V3" \
    --replica_config_device a100 \
    --replica_config_network_device a100_dgx \
    --replica_config_tensor_parallel_size 8 \
    --replica_config_expert_parallel_size 8 \
    --replica_config_enable_kv_prefetch \
    --metrics_config_store_layer_metrics
```

**Request-Level Results** (128 requests, 2048 prefill + 512 decode tokens each):

| Metric | Value |
|--------|-------|
| Simulation time | 242.38s |
| Mean E2E latency | 4977.0 ms |
| Mean TTFT (time to first token) | 237.5 ms |
| Mean TPOT (time per output token) | 9.26 ms/token |
| P99 E2E latency | 6825.4 ms |

**Per-Layer Timing** (averaged across decode batches):

| Component | Avg Time (ms) | Description |
|-----------|--------------|-------------|
| `attention_compute_time` | 0.08 | MLA attention (compressed KV) |
| `mlp_compute_time` | 0.14 | MoE expert GEMM (8 of 256 experts) |
| `kv_cache_load_time` | 0.29 | KV cache PCIe transfer (MLA: smaller KV) |
| `tensor_parallel_comm_time` | 0.13 | TP=8 all-reduce over NVLink |
| `prefetch_overlap_savings` | 0.18 | DMA overlap with compute |
| **total_time** | **0.46** | Per-layer decode step |

**Gantt Charts** (`../example_outputs/deepseek_v3/`):

The timeline Gantt shows all three hardware streams per layer:
- `layer_gantt_batch_0_prefill.png`: Prefill, TP=8 — green compute blocks followed by orange all-reduce communication. The compute-then-comm pattern is clearly visible, with the staircase showing how communication serializes after compute.
- `layer_gantt_batch_4_decode.png`: Decode, TP=8, prefetch=ON — all three streams visible: green compute, blue I/O (DMA, nearly invisible since MLA compresses KV), and orange comm. The I/O is almost fully hidden behind compute, showing that MLA's small KV cache makes DMA prefetch very effective (savings=5.23ms).

**Summary Plot** (`../example_outputs/deepseek_v3/layer_timing_summary.png`):

The stacked bar summary shows a balanced profile across compute (green, ~0.22 ms), I/O (blue, ~0.10 ms effective after prefetch), and communication (orange, ~0.13 ms). DeepSeek-V3's MLA attention significantly reduces KV cache size (kv_lora_rank=512 vs full multi-head), leading to much smaller I/O costs compared to Llama-2-7b. The prefetch savings line (red dashed, ~0.18 ms) nearly eliminates the I/O bottleneck.

### Output Files

Each simulation run produces in `simulator_output/<timestamp>/`:

| File | Description |
|------|-------------|
| `config.json` | Full simulation configuration |
| `request_metrics.csv` | Per-request E2E latency, TTFT, TPOT, scheduling delay |
| `layer_timings.csv` | Per-layer, per-batch timing breakdown (can be very large) |
| `layer_timings.json` | Same data in JSON format |
| `plots/layer_gantt_batch_N.png` | Per-batch Gantt chart |
| `plots/layer_timing_summary.png` | Averaged stacked bar summary |
| `chrome_trace.json` | Chrome trace viewer format (open in `chrome://tracing`) |

The `../example_outputs/` directory contains sample outputs from both runs above, including request metrics CSV, config JSON, layer timing CSV sample (first 200 rows), and Gantt/summary plots.

---

## Architecture

### New/Modified Files

**New entities:**
- `vidur/entities/layer_execution_time.py` - Per-layer timing dataclass
- `vidur/metrics/gantt_plotter.py` - Gantt chart generation and layer timing storage
- `vidur/execution_time_predictor/moe_execution_time_predictor.py` - MoE-aware predictor

**Modified:**
- `vidur/entities/execution_time.py` - Extended with per-layer breakdowns, MoE fields, prefetch logic, bandwidth-based KV I/O
- `vidur/config/model_config.py` - Added MoE/MLA fields and DeepSeek/Mixtral/Qwen configs
- `vidur/config/config.py` - Added `enable_kv_prefetch`, `expert_parallel_size`, `store_layer_metrics`
- `vidur/config/device_sku_config.py` - Added `memory_bandwidth_gb_per_s` and `pcie_bandwidth_gb_per_s` to all device SKU configs
- `vidur/execution_time_predictor/base_execution_time_predictor.py` - MoE method hooks, PCIe-based `_get_kv_cache_load_time()`
- `vidur/execution_time_predictor/sklearn_execution_time_predictor.py` - Uses `get_profiling_name()` for profiling data lookup
- `vidur/metrics/metrics_store.py` - Layer timing integration
- `vidur/metrics/constants.py` - MoE operation metrics

### Data Flow

```
Batch arrives at pipeline stage
    |
    v
ExecutionTimePredictor.get_execution_time()
    |-- Computes attention times (MHA/GQA/MLA projections)
    |-- Computes MoE times (routing + expert compute + weight load)
    |-- Computes communication (TP all-reduce + EP dispatch/combine)
    |
    v
ExecutionTime object created
    |-- Builds per-layer LayerExecutionTime list
    |-- Applies KV prefetch overlap if enabled
    |
    v
MetricsStore.on_replica_stage_schedule()
    |-- Records layer timings to LayerTimingStore
    |
    v
MetricsStore.plot()
    |-- Generates Gantt charts and CSV/JSON exports
```

---

## Sparse and IO Profilers (`vidur/profiling/sparse/`)

The `vidur/profiling/sparse/` package provides on-device profiling for MoE-specific operations that cannot be captured by the standard dense-model profiling pipeline. It has three components, each with an impl (the PyTorch model) and a wrapper (the profiling harness).

### MoE MLP Profiler (`sparse_mlp_impl.py`, `sparse_mlp_wrapper.py`)

Builds a single MoE transformer block on-GPU and measures each sub-operation with CUDA event timers:

1. **Router/gating** — a `Linear(hidden, num_experts)` projection followed by softmax and top-K selection. Three separate timers capture the gate GEMM, softmax, and top-K.
2. **Expert dispatch** — reorganises tokens into per-expert batches (expand + index).
3. **Grouped expert GEMM** — loops over active experts, running each `ExpertMLP` (gate\_proj \* up\_proj → SiLU → down\_proj) on its assigned token subset. Only `min(num_routed_experts, 64)` expert modules are instantiated to cap GPU memory; overflow tokens reuse expert 0, preserving the GEMM shapes that matter for timing.
4. **Expert combine** — weighted scatter-add back to token space.
5. **Shared expert** — a dense MLP applied to all tokens, timed separately.

The wrapper (`SparseMlpWrapper`) runs 2 warmup + 10 active iterations, collects `CudaTimer` stats (or `RecordFunctionTracer` traces), and returns a dict of per-operation median times keyed by token count.

### MLA Attention Profiler (`mla_attention_impl.py`, `mla_attention_wrapper.py`)

Profiles the MLA-specific projection GEMMs that differ from standard MHA/GQA:

- **Q compression**: `q_down_proj` (hidden → q\_lora\_rank) then `q_up_proj` (q\_lora\_rank → heads \* (nope + rope))
- **KV compression**: `kv_down_proj` (hidden → kv\_lora\_rank + rope\_dim)
- **KV decompression**: `kv_up_proj` (kv\_lora\_rank → heads \* (nope + v\_head\_dim)), measured separately even though production fuses this into attention
- **RoPE**: element-wise ops on the rope portions of Q and K
- **Output projection**: `o_proj` (heads \* v\_head\_dim → hidden)

The actual attention kernel is stubbed (a `torch.randn` of the correct shape) because FlashAttention is profiled by the existing attention profiler; MLA profiling targets only the surrounding projection costs.

### IO Profiler (`io_profiler.py`)

Measures real data-transfer bandwidth using CUDA-event-timed `tensor.copy_()` operations (5 warmup + 20 active, median reported). Three transfer types:

| Method | Source → Dest | What it models |
|--------|--------------|----------------|
| `profile_hbm_read` | GPU → GPU (same device) | Expert weight fetch from HBM |
| `profile_pcie_h2d` | Pinned host → GPU | CPU-offloaded expert weights or KV prefetch |
| `profile_pcie_d2h` | GPU → pinned host | KV cache offloading to host |

Two higher-level helpers compose these primitives:
- `profile_expert_weight_load(hidden, intermediate, num_local_experts, source)` — computes the byte count for 3 matrices × N experts and calls the appropriate HBM or PCIe primitive.
- `profile_kv_cache_transfer(kv_bytes_per_token, num_tokens, batch_size, source)` — likewise for a KV cache batch.

`run_bandwidth_sweep(sizes_bytes)` runs all three transfer types across a list of payload sizes, producing `IOProfileResult` rows with `(transfer_type, size_bytes, latency_ms, bandwidth_gb_per_s)`.

### Orchestration (`main.py`)

The entry point `python -m vidur.profiling.sparse.main` runs all three profilers in sequence for each requested model. It supports Ray for multi-GPU parallelism (round-robin token counts across workers) or single-GPU mode (`--disable_ray`). Outputs are timestamped CSVs (`sparse_mlp.csv`, `mla_attention.csv`, `io_bandwidth.csv`) plus a config YAML, all under `profiling_outputs/sparse/<timestamp>/<model>/`.

Model configs come from a built-in preset dict (DeepSeek-V3, Mixtral, Qwen3-30B, Engram-27B) or are loaded dynamically from `BaseModelConfig` for any registered model name.

---

## Integration with InferSim

InferSim provides the analytical framework for MoE timing and KV I/O estimation:

- **FLOPs calculation**: `2*M*N*K` per GEMM, with separate counts for routed and shared experts
- **MFU (Model FLOPS Utilization)**: Empirical measurements from GPU kernel benchmarks (DeepGEMM, FlashAttention-3, FlashInfer)
- **KV cache I/O modeling**: `kv_load_time = kv_bytes / PCIe_bandwidth`. The KV cache for the next layer is transferred from host memory over PCIe. InferLens uses PCIe bandwidth from `device_sku_config` (with 80% efficiency) for the per-layer Gantt visualization and prefetch savings, while keeping its sklearn-profiled base time for the overall model_time.
- **Expert weight I/O**: `load_time = expert_params * local_experts / HBM_bandwidth`, using HBM bandwidth since expert weights reside in GPU memory. Captures whether expert layers are compute-bound or I/O-bound via `max(compute, load)`.
- **Communication**: Bandwidth-delay model for NVLink and RDMA, with DeepEP-specific dispatch/combine patterns

InferLens's `MoEExecutionTimePredictor` implements the MoE calculations, and `base_execution_time_predictor` implements the PCIe-based KV cache load estimation, allowing InferLens's event-driven simulation to model MoE inference and KV prefetching at scale.
