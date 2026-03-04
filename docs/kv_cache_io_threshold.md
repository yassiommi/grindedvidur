# KV Cache I/O Threshold: When I/O Dominates Layer Execution Time

## The Question

When offloading KV cache to host memory (CPU DRAM) and uploading it back to GPU each decode step over PCIe, at what point does that I/O take longer than the GPU compute for the same layer?

## The Formula

Per-layer KV cache load time:
```
kv_load_time = kv_bytes_per_token × avg_kv_len × batch_size / PCIe_bandwidth
```

Layer compute time (for decode, attention-dominated at long context):
```
compute_time ≈ attention_flops_per_token × avg_kv_len × batch_size / GPU_TFLOPS
```

Both scale identically with `avg_kv_len × batch_size`, so their **ratio is a hardware/architecture constant** — it does not change with batch size or context length on its own. The crossover is determined by:

```
kv_bytes_per_token / PCIe_BW  vs.  attention_flops_per_token / GPU_TFLOPS
```

IO dominates when:
```
kv_bytes_per_token / PCIe_BW > attention_flops_per_token / GPU_TFLOPS
```

At short context, MLP compute (which scales with batch_size only, not kv_len) matters and slightly favors IO being sub-dominant. The full crossover context length where IO first equals compute is:

```
kv_len_crossover = mlp_flops_per_token / (kv_bytes_per_token × GPU_TFLOPS/PCIe_BW - attention_flops_per_token)
```

## Measured Results on A100 (PCIe Gen4, 31.5 GB/s)

| Model | Architecture | kv_bytes/token/layer | IO/Compute ratio | IO-bound? |
|---|---|---|---|---|
| Llama-2-7B | MHA, dense | 16,384 B | **4.94×** | Always (100% of batches) |
| DeepSeek-V3 | MLA, MoE | 1,152 B | **1.47×** | 60.3% of batches |

DeepSeek-V3's MLA reduces KV size by **14.2×** vs MHA, which is why it sits near the boundary rather than being always IO-bound.

The analytical crossover context length for DeepSeek-V3 on A100 PCIe Gen4: **~38,480 tokens**.

Below that length, MLP compute pushes the total compute time above the KV load time. Above it, IO dominates every decode step.

## PCIe Generation Impact

PCIe bandwidth directly scales kv_load_time:

| PCIe Gen | Bandwidth | DeepSeek IO/Compute ratio |
|---|---|---|
| Gen4 | 31.5 GB/s | 1.47× |
| Gen3 | 16.0 GB/s | 2.81× |

Since compute doesn't change, halving PCIe bandwidth nearly doubles the ratio. The crossover context length drops accordingly on Gen3.

## GPU Prefetching

With prefetching enabled (`--replica_config_enable_kv_prefetch`), the DMA engine loads the next layer's KV cache while the GPU computes the current layer:

```
prefetch_savings = min(compute_time, next_layer_kv_load_time)
```

This partially masks IO overhead but the savings are **capped at compute time**. For models where IO/Compute >> 1 (like Llama-2-7B at 4.94×), most of the KV load still serializes after compute. For DeepSeek-V3 near the boundary, prefetch hides close to half the KV load time.

## Key Takeaways

1. **Dense MHA models are always IO-bound** when offloading KV — the KV size is simply too large relative to compute for any practical context length.
2. **MLA/MoE models live near the boundary** — roughly 60% IO-bound in typical workloads.
3. **Batch size doesn't move the threshold** — both IO and compute scale identically with `batch_size × kv_len`, so the ratio is fixed by architecture and hardware alone.
4. **Context length determines per-batch IO vs compute**, not batch size: longer sequences push each batch deeper into IO-bound territory.
5. **PCIe bandwidth is the primary lever** — upgrading from Gen3 to Gen4 halves kv_load_time directly.
