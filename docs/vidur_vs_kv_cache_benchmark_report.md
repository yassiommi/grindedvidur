# Vidur vs. MLCommons KV Cache Benchmark: Comparison Report

## 1. Executive Summary

**Vidur** is a high-fidelity LLM inference system simulator (Microsoft) that models end-to-end serving performance — scheduling, batching, KV cache allocation, and latency prediction — without requiring GPU hardware. The **MLCommons KV Cache Benchmark** is a storage-focused diagnostic tool that measures real I/O performance of tiered storage systems (GPU/CPU/NVMe) under realistic KV cache offloading workloads.

They are **complementary, not competing** tools. Vidur simulates the *logical* behavior of LLM serving (what gets scheduled, how memory is allocated, what latencies result), while the KV Cache Benchmark measures *physical* storage performance (actual read/write bandwidth, I/O latency, tier utilization). However, Vidur has significant potential to serve as an **alternative or augmentation** to the benchmark for scenarios where real hardware is unavailable or where system-level analysis is needed.

---

## 2. Architecture Comparison

| Dimension | Vidur | MLCommons KV Cache Benchmark |
|-----------|-------|------------------------------|
| **Core approach** | Event-driven discrete simulation | Real workload execution on hardware |
| **Hardware required** | None (uses profiled execution models) | Real storage hardware (NVMe, RAM, optionally GPU) |
| **Primary focus** | End-to-end inference serving (scheduling, batching, latency) | Storage subsystem I/O performance |
| **KV cache model** | Logical block allocation with radix-tree prefix cache | Physical 3-tier storage with waterfall LRU eviction |
| **Storage tiers** | Single tier (GPU HBM) + prefix cache region | Three tiers: GPU VRAM → CPU RAM → NVMe |
| **Execution model** | Simulated (ML-predicted execution times) | Real I/O operations (mmap, fsync, CUDA memcpy) |
| **Output** | TTFT, TPOT, throughput, scheduling efficiency, cache hit rates | Storage bandwidth, I/O latency percentiles, tier utilization |
| **Language** | Python (simulation) | Python (with optional CUDA/CuPy) |

---

## 3. KV Cache Management: Feature-by-Feature

### 3.1 Memory Hierarchy

**Vidur** models a single GPU HBM tier:
- Block-based allocation: 16-token blocks (configurable via `block_size`)
- Total blocks computed from: `(available_memory - model_params) / kv_cache_per_request`
- Watermark reservation: 1% of blocks kept free (`watermark_blocks_fraction = 0.01`)
- Prefix cache carved from total blocks: up to 20% (`max_blocks_fraction = 0.2`)

**KV Cache Benchmark** models a three-tier hierarchy:
- **GPU tier**: Sub-ms latency, CUDA-synchronized, pinned memory transfers
- **CPU tier**: Tens of ms, NumPy arrays, dictionary-based cache
- **NVMe tier**: Hundreds of ms, memory-mapped `.npy` files, explicit `fsync()`, page cache management
- Waterfall LRU: eviction cascades entries downward (GPU → CPU → NVMe)
- Capacity thresholds: 80% for non-terminal tiers, 100% for terminal (NVMe)

**Gap**: Vidur has **no CPU or NVMe offloading model**. All KV cache lives in GPU HBM. When capacity is exceeded, requests are preempted (re-queued), not offloaded to a lower tier.

### 3.2 Eviction Policies

**Vidur**:
- Prefix cache: LRU eviction on leaf nodes of radix tree (`OrderedDict`-based)
- Non-leaf nodes (shared prefixes) are preserved during eviction
- Scheduler-level: requests preempted when memory is full (vLLM scheduler)
- No data movement on eviction — blocks are simply freed

**KV Cache Benchmark**:
- LRU with snapshot-based optimized eviction (O(n) vs O(n^2))
- Non-terminal tiers: evicted entries **demote** to next tier (data movement)
- Terminal tier (NVMe): entries deleted on eviction
- Thread-safe locking for concurrent access

**Gap**: Vidur eviction is **delete-only**; the benchmark models **demotion** (data cascading to slower tiers), which is the production-realistic pattern for KV offloading.

### 3.3 Prefix Caching

**Vidur** (radix tree implementation, 443 lines):
- `RadixTreeNode`: stores `token_segment`, `children`, `num_blocks`, LRU tracking
- `match_prefix(token_ids)` → walks tree, returns block-aligned cached tokens
- `insert(token_ids)` → inserts with automatic eviction and node splitting
- Statistics: `hit_rate`, `token_hit_rate`, `total_evictions`, `blocks_evicted`
- Config: `num_shared_prefixes = 5`, `shared_prefix_length_fraction = 0.3`

**KV Cache Benchmark**:
- Probabilistic system prompt detection (20% default hit probability)
- `min_prefix_length = 50` tokens threshold
- Caches usage metadata: frequency, user count, timestamps
- Less sophisticated tree structure, but combined with multi-turn conversation reuse

**Vidur advantage**: More sophisticated radix-tree prefix matching with node splitting and merging. The benchmark's prefix cache is simpler but integrated into a richer multi-turn conversation model.

### 3.4 Multi-Turn Conversation Modeling

**Vidur**: No explicit multi-turn conversation modeling. Requests are independent. Prefix cache provides implicit conversation reuse if token sequences overlap.

**KV Cache Benchmark**:
- Dedicated `ConversationManager` with stateful tracking
- Turn-specific cache keys: `"conv_id_turn_2"`
- LRU eviction for conversations (1,000 conversations, 50 turns max)
- Context accumulation across turns

**Gap**: Vidur lacks multi-turn conversation state, which is critical for modeling real chat workloads where KV cache reuse across turns is a major optimization.

### 3.5 KV Cache Transfer Modeling

**Vidur** (Prefill-Decode Disaggregation):
- `KvCacheTransferEvent`: models inter-GPU transfer for PDD architecture
- Transfer time: `total_kv_bytes / (pcie_bandwidth_gbps * 1e9)`
- Supports compute-transfer overlap (`prefetch_overlap_ms`)
- Device bandwidths: A100 PCIe=31.5 GB/s, H100 PCIe=64 GB/s, HBM up to 3350 GB/s

**KV Cache Benchmark**:
- Real I/O operations with actual measured latencies
- Per-tier bandwidth measurement (not modeled, measured)
- Supports `bpftrace` block-layer tracing for deep I/O stack analysis

**Vidur advantage**: Analytical transfer modeling allows what-if analysis across hardware configs. **Benchmark advantage**: Measures real-world performance including OS/driver/firmware effects.

---

## 4. Workload Generation Comparison

### 4.1 Request Patterns

| Feature | Vidur | KV Cache Benchmark |
|---------|-------|-------------------|
| **Synthetic generation** | Poisson, Gamma, Static arrival; Zipf, Uniform, Fixed token lengths | User personas (chatbot/coding/document) with think times |
| **Trace replay** | CSV trace replay (`TraceRequestGeneratorConfig`) | BurstGPT (Azure production), ShareGPT (real conversations) |
| **Arrival distribution** | Configurable QPS with Poisson/Gamma | Think-time based with QoS priority levels |
| **Token distribution** | Zipf (theta configurable), Uniform, Fixed | Per-persona ranges (e.g., chatbot: 512-4096 prefill) |
| **QoS levels** | None | 3 levels: Interactive (15%), Responsive (35%), Batch (50%) |
| **RAG simulation** | None | Zipfian document retrieval, configurable chunk size |

**Vidur advantage**: More flexible statistical distributions for arrival patterns and token lengths. **Benchmark advantage**: Richer real-world workload types (multi-turn, RAG, QoS priorities).

### 4.2 Model Support

**Vidur**: Llama-2 (7B/13B/70B), Llama-3 (8B/70B), CodeLlama-34B, Qwen-72B, InternLM-20B, DeepSeek-V3 (MLA)

**KV Cache Benchmark**: llama2-7b, llama3.1-8b, mistral-7b, deepseek-v3 — with configurable KV bytes/token for any model

Both compute KV cache size from model architecture (layers, KV heads, head dim), with Vidur additionally supporting MLA compressed KV dimensions.

---

## 5. Metrics Comparison

### 5.1 Serving Performance Metrics (Vidur only)

- **TTFT** (Time to First Token)
- **TPOT** (Time Per Output Token)
- **End-to-end request latency**
- **Throughput** (tokens/sec, requests/sec)
- **Batch utilization and scheduling efficiency**
- **Queue wait times**
- **Preemption rates**
- Chrome trace visualization

### 5.2 KV Cache Metrics

| Metric | Vidur | KV Cache Benchmark |
|--------|-------|-------------------|
| Cache hit rate | Yes (prefix cache) | Yes (all tiers) |
| Token hit rate | Yes | Yes |
| Eviction count/blocks | Yes | Yes |
| Per-tier bandwidth | No (computed analytically) | Yes (measured) |
| Per-tier latency (P50/P95/P99) | No | Yes |
| I/O volume per tier (GB) | No | Yes |
| Multi-turn reuse rate | No | Yes |
| QoS SLA compliance | No | Yes |
| Block-layer I/O stack tracing | No | Yes (bpftrace) |

### 5.3 Storage Metrics (Benchmark only)

- **Tier-specific read/write bandwidth** (GB/s)
- **Storage P95 latency** (ms) — primary MLPerf metric
- **Decode bytes read** (GB)
- **IOPS per tier**
- **Queue depth monitoring**
- **Block-layer latency histograms** (D2C, Q2D, VFS syscall)
- **Spatial LBA heatmaps**

**Gap**: Vidur has no real storage I/O metrics. It models bandwidth analytically but cannot capture real-world effects (OS scheduling, driver overhead, SSD firmware, thermal throttling, GC pauses).

---

## 6. Gap Analysis

### 6.1 Gaps in Vidur (relative to the benchmark)

| Gap | Severity | Description |
|-----|----------|-------------|
| **No multi-tier storage** | **High** | Vidur only models GPU HBM. No CPU RAM or NVMe offloading. Cannot simulate KV cache spillover. |
| **No demotion on eviction** | **High** | Evicted KV blocks are deleted, not demoted to a lower tier. Misses the write amplification and latency of tier cascading. |
| **No real I/O measurement** | **High** | All bandwidth/latency is analytical. Cannot capture real storage device behavior (SSD wear, thermal throttling, contention). |
| **No multi-turn conversations** | **Medium** | Requests are independent. Cannot model the KV cache reuse patterns of multi-turn chat, which dominates production workloads. |
| **No QoS differentiation** | **Medium** | All requests treated equally. Cannot model priority-based scheduling or SLA compliance. |
| **No RAG workload** | **Medium** | No retrieval-augmented generation simulation. Cannot model the mixed read patterns of RAG-heavy workloads. |
| **No NVMe-specific modeling** | **Medium** | No SSD preconditioning, steady-state behavior, or GC impact simulation. |
| **No autoscaling simulation** | **Low** | Cannot discover maximum user count while maintaining latency targets. |
| **No bpftrace/block-layer visibility** | **Low** | Cannot provide I/O stack breakdown (this requires real hardware anyway). |

### 6.2 Gaps in the Benchmark (relative to Vidur)

| Gap | Severity | Description |
|-----|----------|-------------|
| **No scheduling simulation** | **High** | Doesn't model batch scheduling, continuous batching, or scheduler algorithms (vLLM, Orca, Sarathi). |
| **No execution time modeling** | **High** | Doesn't predict compute latency (attention, MLP). Only measures storage I/O. |
| **No pipeline/tensor parallelism** | **High** | Cannot model multi-GPU configurations (TP, PP, PDD). |
| **No end-to-end latency prediction** | **High** | Cannot predict TTFT or TPOT — only storage contribution to latency. |
| **Simpler prefix cache** | **Medium** | Probabilistic prefix detection vs. Vidur's full radix tree with node splitting. |
| **No capacity planning** | **Medium** | Cannot sweep configurations to find optimal deployment parameters. |
| **Requires real hardware** | **Medium** | Cannot run what-if analyses without physical storage devices. |
| **No preemption modeling** | **Low** | Cannot simulate request preemption and re-queuing under memory pressure. |

---

## 7. How Vidur Can Serve as an Alternative

### 7.1 Where Vidur Can Replace the Benchmark

1. **Hardware-free capacity planning**: Vidur can estimate how many concurrent users a given GPU config can support, without buying hardware. The benchmark requires real NVMe/RAM to test.

2. **Scheduling algorithm comparison**: Vidur can compare vLLM vs. Orca vs. Sarathi scheduling under identical workloads. The benchmark has no scheduling model.

3. **Configuration sweeps**: Vidur can sweep batch sizes, block sizes, parallelism strategies, and model choices in minutes. The benchmark requires hours of real I/O testing per configuration.

4. **Prefix cache effectiveness**: Vidur's radix-tree prefix cache is more sophisticated and can predict hit rates under various shared-prefix scenarios without running real workloads.

### 7.2 Where Vidur Cannot Replace the Benchmark

1. **Real storage device evaluation**: Comparing NVMe SSDs, measuring actual bandwidth under load, detecting thermal throttling, GC pauses — these require real hardware.

2. **Tiered offloading performance**: Until Vidur adds multi-tier storage modeling, it cannot predict the latency impact of KV cache spillover to CPU/NVMe.

3. **I/O stack analysis**: bpftrace, block-layer tracing, LBA heatmaps — these are inherently hardware-dependent measurements.

4. **Production validation**: The benchmark can validate against real traces with <5% error. Vidur's predictions depend on pre-profiled execution models.

### 7.3 Where Vidur Can Augment the Benchmark

1. **Trace generation**: Run Vidur simulations with realistic scheduling → export traces → feed into the benchmark for real I/O testing. This gives the benchmark more realistic access patterns than its built-in synthetic workloads.

2. **Latency decomposition**: Use Vidur for compute latency + the benchmark for storage latency → combine for full end-to-end prediction.

3. **What-if analysis**: Use Vidur to narrow down promising configurations, then validate the top candidates with the benchmark on real hardware.

---

## 8. Roadmap: Bridging the Gaps

To make Vidur a more complete alternative (or complement) to the KV Cache Benchmark, the following extensions are recommended, ordered by impact:

### Phase 1: Multi-Tier KV Cache (High Impact)

**Goal**: Model GPU → CPU → NVMe KV cache offloading with waterfall eviction.

- Add `StorageTierConfig` with per-tier capacity and bandwidth parameters
- Extend `BaseReplicaScheduler` block management to track which tier each block resides in
- Implement demotion (evict from GPU → CPU → NVMe) instead of delete
- Model promotion latency on cache hits from lower tiers
- Use benchmark-measured latencies as default parameters

### Phase 2: Multi-Turn Conversation Support (Medium Impact)

**Goal**: Model KV cache reuse across conversation turns.

- Add `ConversationManager` entity tracking active conversations
- Extend `Request` with conversation ID and turn number
- Modify prefix cache to use conversation-aware cache keys
- Add conversation-based request generators (think-time, context accumulation)

### Phase 3: Trace Export for Benchmark Integration (Medium Impact)

**Goal**: Generate traces that can drive the KV Cache Benchmark.

- Export Vidur simulation traces in BurstGPT CSV format (timestamp, model, request_tokens, response_tokens)
- Include per-request KV cache size, tier assignments, and access patterns
- Enable round-trip validation: Vidur trace → benchmark run → compare predicted vs. measured I/O latency

### Phase 4: QoS and RAG Workloads (Lower Impact)

**Goal**: Richer workload modeling for production realism.

- Add QoS levels to requests with priority-based scheduling
- Add RAG document retrieval simulation (Zipfian access, configurable chunk sizes)
- Add SLA compliance metrics (P95 latency targets per QoS level)

---

## 9. Conclusion

Vidur and the MLCommons KV Cache Benchmark occupy different but overlapping positions in the LLM inference tooling landscape:

- **Vidur** excels at **system-level simulation**: scheduling, parallelism, batching, and capacity planning — all without hardware. Its KV cache model is logically sophisticated (radix-tree prefix cache, PDD transfers) but physically limited to a single GPU tier.

- **The KV Cache Benchmark** excels at **storage-level measurement**: real I/O bandwidth, tiered offloading, and hardware diagnostics. It lacks any model of compute, scheduling, or multi-GPU systems.

**The strongest path forward** is bidirectional integration: use Vidur to generate realistic workload traces and narrow configuration spaces, then validate storage performance with the benchmark on real hardware. Extending Vidur with multi-tier storage modeling would further close the gap, enabling fully simulated end-to-end analysis of KV cache offloading strategies.
