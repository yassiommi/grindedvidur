#!/usr/bin/env python3
"""Build a Word document for the KV Cache blog post with embedded figures."""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, "blog", "figures")
OUT = os.path.join(REPO, "blog", "kv_cache_blog.docx")

doc = Document()

# ── Styles ────────────────────────────────────────────────────
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)
style.font.color.rgb = RGBColor(0x24, 0x29, 0x2E)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

for level in range(1, 4):
    hs = doc.styles[f"Heading {level}"]
    hs.font.name = "Calibri"
    hs.font.color.rgb = RGBColor(0x24, 0x29, 0x2E)

def heading(text, level=1):
    doc.add_heading(text, level=level)

def para(text, bold=False, italic=False, size=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return p

def caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x6A, 0x73, 0x7D)

def img(filename, width=6.2):
    path = os.path.join(FIG, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        para(f"[Missing figure: {filename}]", italic=True)

def table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Light Shading Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)

def spacer():
    doc.add_paragraph()

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
title = doc.add_heading("KV Cache and the IO Wall", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run(
    "How Memory Bandwidth Became the Bottleneck in LLM Inference\n"
    "— and What We're Doing About It"
)
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(0x6A, 0x73, 0x7D)
run.italic = True

spacer()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("Based on experiments run in the Vidur / InferSim simulation framework")
run.font.size = Pt(10)
run.font.color.rgb = RGBColor(0x95, 0x9D, 0xA5)
run.italic = True

spacer()

# ══════════════════════════════════════════════════════════════
# TL;DR
# ══════════════════════════════════════════════════════════════
heading("TL;DR", 1)
para(
    "Every token an LLM generates requires loading the entire KV cache accumulated so far — "
    "and that load dominates everything else. For a vanilla Llama-2-7B serving decode requests, "
    "the KV cache IO takes 4.94× longer than the actual compute. 100% of decode batches are IO-bound. "
    "The field has responded with a sequence of architectural innovations — GQA, MLA, sparse attention, "
    "Engram, TurboQuant — each attacking the same bottleneck from a different angle. But even with a 199× "
    "compressed cache, system-level scheduling determines whether any of it matters. When concurrent agentic "
    "sessions thrash the cache, 81–84% of prefill compute is wasted, TTFT inflates by 11.6× at p95, and a "
    "monitoring system that only looks at utilization sees nothing wrong."
)

spacer()
doc.add_page_break()

print("Writing Part I...")

# ══════════════════════════════════════════════════════════════
# PART I: THE IO WALL
# ══════════════════════════════════════════════════════════════
heading("Part I: The IO Wall", 1)

heading("Why Decode Is Different", 2)
para(
    "Prefill and decode look similar on the surface — both run transformer layers over tokens — "
    "but their compute profiles are completely different."
)
para(
    "During prefill, you process a batch of prompt tokens together. The attention GEMM is a fat "
    "matrix multiply. GPU utilization is high. Compute dominates."
)
para(
    "During decode, you generate one new token per request per step. The attention operation becomes "
    "a thin vector-times-matrix: one query vector attended against a KV cache of all prior tokens. "
    "The GEMM is trivially small. But you still have to load the full KV cache for every token generated."
)
para(
    "For a model with 32 layers, 32 KV heads, head dimension 128, and FP16 precision, that's "
    "2 × 32 × 128 × 2 bytes = 16,384 bytes per token per layer. At 2,048 tokens of context and "
    "32 layers: 1 GB of KV data to load from HBM for a single decode step."
)

heading("Measuring the Bottleneck", 2)
para(
    "Our layer-timing experiments on Llama-2-7B quantify this precisely:"
)
table(
    ["Component", "Time per layer"],
    [
        ["Attention compute", "0.462 ms"],
        ["MLP compute", "0.681 ms"],
        ["KV cache load", "1.583 ms"],
        ["IO/Compute ratio", "4.94×"],
    ]
)
spacer()
para(
    "The model is IO-bound in 100% of decode batches. Every single one. Adding more compute "
    "(bigger GPU) doesn't help. The bottleneck is reading data from memory."
)
spacer()
img("fig01_io_wall_layer_breakdown.png")
caption("Figure 1: Per-layer decode timing. Llama-2-7B (left) is dominated by KV cache IO. "
        "DeepSeek-V3 with MLA (right) achieves near-balance.")
spacer()

heading("The Three-Stream Hardware Model", 2)
para(
    "Modern GPUs have three independent execution units: SM (compute), DMA engine "
    "(memory transfers), and NCCL (communication). You can overlap the next layer's KV load "
    "with the current layer's compute (\"prefetch\"). But savings are bounded by "
    "min(compute_time, next_kv_load_time). When IO >> compute — the MHA regime — "
    "the GPU sits idle waiting for data."
)
img("fig09_three_stream_scheduling.png")
caption("Figure 2: Without prefetch (top), IO runs sequentially. With prefetch (bottom), "
        "DMA overlaps with compute — but savings are capped by compute time.")
spacer()

heading("PDD Exposes the Problem Further", 2)
para(
    "Prefill-Decode Disaggregation (PDD) separates prefill and decode onto different GPU pools. "
    "The KV cache must transfer between pools over PCIe. For MHA models, this transfer dwarfs "
    "the actual decode compute."
)
img("fig08_pdd_transfer_dominance.png")
caption("Figure 3: KV transfer/compute ratios in PDD. MHA: 64.7× — the transfer is 65× "
        "the compute it's trying to support.")
spacer()
para(
    "And a cruel irony: faster GPUs make this worse. H100 delivers 3.2× more compute TFLOPS "
    "than A100, but PCIe Gen5 is only 2× faster than Gen4. The IO-to-compute gap grows each generation."
)

doc.add_page_break()
print("Writing Part II...")

# ══════════════════════════════════════════════════════════════
# PART II: ARCHITECTURAL RESPONSE
# ══════════════════════════════════════════════════════════════
heading("Part II: The Architectural Response — Compressing the Cache", 1)

heading("MHA → GQA → MLA: The Ladder of Compression", 2)
para(
    "Multi-Head Attention (MHA) is the baseline. Every query head has its own KV head. "
    "For Llama-2-7B: 32 KV heads × 128 dims × 2 bytes × 2 (K+V) = 16,384 bytes per token per layer."
)
para(
    "Grouped Query Attention (GQA), introduced in Llama-2-70B, shares KV heads across groups of "
    "query heads. With 8 KV heads serving 64 query heads: 4,096 bytes/token/layer — a 4× reduction."
)
para(
    "Multi-head Latent Attention (MLA), DeepSeek-V3's invention, compresses KV into a low-rank "
    "latent vector of dimension 576, rather than the full tensor. It reconstructs K and V via "
    "learned up-projections during decode: (512 + 64) × 2 = 1,152 bytes/token/layer — a 14.2× "
    "reduction from MHA."
)
img("fig02_kv_cache_size_landscape.png")
caption("Figure 4: Left — KV cache bytes per token per layer (log scale). MLA is a qualitative jump. "
        "Right — Total KV at 1M context. Only MLA-based configs fit in a single H100.")
spacer()

para(
    "The impact on IO/compute balance is dramatic. DeepSeek-V3 with MLA achieves an IO/Compute "
    "ratio of 1.47× (vs 4.94× for MHA). Only 60.3% of decode batches are IO-bound, compared to "
    "100% for MHA."
)
img("fig03_io_compute_shift.png")
caption("Figure 5: Left/Center — IO-bound fraction: 100% for MHA vs 60.3% for MLA. "
        "Right — Context length (not batch size) drives IO-boundedness.")
spacer()

heading("DeepSeek's Sparse Attention", 2)
para(
    "While MLA compresses the KV representation, DeepSeek also explored native sparse attention — "
    "selectively attending to only the most relevant KV entries rather than the full context. "
    "The two techniques compose: sparse selection over a compressed representation yields a "
    "doubly-reduced IO footprint. The key system insight: the scheduler must now track which KV "
    "entries are hot for each active request, not just total cache space consumed."
)

heading("Engram: Conditional Memory as a New Axis of Sparsity", 2)
para(
    "Engram (DeepSeek, arXiv:2601.07372) separates language modeling into two workloads: "
    "dynamic reasoning (handled by MoE transformer layers) and static pattern recall (handled by "
    "O(1) lookup tables indexed by N-gram hashes). By replacing 17 routed experts with Engram lookup, "
    "you load 24% fewer expert weight tensors from HBM per layer — directly reducing the IO bottleneck."
)
para(
    "The killer property: Engram addresses are deterministic — they depend only on input token IDs, "
    "not activations. DMA transfers begin before any layer executes. The GPU compute of preceding "
    "layers fully hides the host DRAM lookup. The DMA never stalls (9–64× prefetch headroom)."
)
img("fig04_engram_pareto.png")
caption("Figure 6: Left — U-shaped quality curve: optimal split at ρ ≈ 0.74. Center — Engram is "
        "21–31% faster at batch 32–64. Right — Prefetch headroom: the DMA transfer is always "
        "hidden behind compute, regardless of batch size.")
spacer()
para(
    "At V3 scale, a 100B-parameter Engram table lives entirely in host DRAM (~186 GB at ~$5/GB), "
    "freeing that HBM for KV cache or batch capacity. O(1) access means table size doesn't affect "
    "per-token latency — a 200B table has the same cost as a 1B table."
)

heading("TurboQuant: Quantizing the Cache to 3 Bits", 2)
para(
    "TurboQuant (Google, arXiv:2504.19874) compresses KV entries from FP16 to 3 bits per element "
    "via PolarQuant (rotational grid mapping) + QJL (sign-bit error correction), achieving 5.33× "
    "compression with negligible accuracy loss. The compression is orthogonal to the attention "
    "architecture — it stacks on top of MHA, GQA, or MLA."
)

table(
    ["Configuration", "KV at 1M ctx", "vs MHA FP16", "Fits H100?"],
    [
        ["MHA FP16", "2,560 GB", "1×", "No"],
        ["GQA FP16", "320 GB", "8×", "No"],
        ["MLA FP16", "68.6 GB", "37×", "Yes (86%)"],
        ["MHA + TQ 3-bit", "480 GB", "5.3×", "No"],
        ["GQA + TQ 3-bit", "60 GB", "42.7×", "Yes (75%)"],
        ["MLA + TQ 3-bit", "12.9 GB", "199×", "Yes (16%)"],
    ]
)
spacer()

img("fig05_turboquant_impact.png")
caption("Figure 7: Left — TPOT improvement: 5.3× on MHA (IO-bound), negligible on GQA (already compact). "
        "Right — Memory access patterns: MHA reads everything, MLA reads 1.6%, TQ reads sparse discrete.")
spacer()

doc.add_page_break()
print("Writing Part III...")

# ══════════════════════════════════════════════════════════════
# PREFIX CACHING
# ══════════════════════════════════════════════════════════════
heading("Prefix Caching: Exploiting Workload Structure", 1)
para(
    "Architecture reduces cache size per token. Prefix caching reduces how many tokens you need "
    "to process. If multiple requests share a common prefix (system prompt, few-shot examples), "
    "cache the KV entries and reuse them. A radix tree with LRU eviction handles lookup."
)
img("fig11_prefix_caching.png")
caption("Figure 8: Left — Token hit rate scales linearly with shared prefix fraction. "
        "Right — Eviction pressure drops 17× at 90% sharing.")
spacer()

table(
    ["Shared Fraction", "Token Hit Rate", "Prefill Reduction", "Blocks Evicted"],
    [
        ["0%", "0%", "0%", "6,016"],
        ["10%", "9.05%", "9.05%", "5,249"],
        ["50%", "48.5%", "48.5%", "2,896"],
        ["90%", "85.3%", "85.3%", "347"],
    ]
)
spacer()
para(
    "For chat applications with shared system prompts, prefix caching can eliminate 50–85% of "
    "all prefill computation. But there is a failure mode that neither architecture nor prefix "
    "caching can prevent."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# PART III: THRASHING
# ══════════════════════════════════════════════════════════════
heading("Part III: When Everything Breaks — The Thrashing Cliff", 1)

heading("Agentic Workloads and Growing Contexts", 2)
para(
    "In agentic inference — tool-use loops, chain-of-thought planning, multi-step code generation — "
    "each step appends to a growing, unique context. Step 0 might be 300 tokens; by step 12, it's "
    "2,700 tokens. Each step shares the full prefix of all prior steps."
)
para(
    "With N concurrent agent sessions, the aggregate working set grows until it exceeds cache capacity. "
    "At that point, inserting new blocks for session A evicts blocks belonging to session B — but "
    "session B needs exactly those blocks on its very next step. This is mid-phase thrashing: "
    "the cache is full of the wrong data, and the system never recovers."
)

heading("The Cliff Is Binary", 2)
para(
    "We simulated this across concurrent sessions (2–12) and cache sizes (200–1,200 blocks). "
    "The result is not a gradient — it's a cliff."
)
img("fig06_thrashing_cliff.png")
caption("Figure 9: Left — Thrashing boundary heatmap. The transition from ~80% to ~20% hit rate "
        "is nearly instantaneous. Right — Compute overhead: below the threshold, no penalty. "
        "Above it, immediate 5× overhead with 81% of prefill compute wasted.")
spacer()

table(
    ["Config", "WS/Cache", "Compute Overhead", "Throughput", "TTFT p95", "Wasted"],
    [
        ["2 conc, 400 blk", "0.8×", "1.00×", "100%", "1.0×", "0%"],
        ["4 conc, 400 blk", "1.7×", "1.48×", "68%", "4.9×", "32%"],
        ["6 conc, 400 blk", "2.5×", "5.19×", "19%", "11.5×", "81%"],
        ["8 conc, 400 blk", "3.4×", "5.38×", "19%", "11.6×", "81%"],
        ["12 conc, 400 blk", "5.1×", "5.38×", "19%", "11.6×", "81%"],
    ]
)
spacer()

heading("Utilization Is a Lie", 2)
para(
    "The most dangerous property of thrashing: standard utilization metrics are completely "
    "uninformative. During severe thrashing (12 concurrent, 400 blocks), cache utilization is "
    "~83% while token hit rate is ~18%. A monitoring system that only tracks utilization would "
    "report the cache as healthy."
)
para(
    "The correct diagnostic triad: high utilization + high eviction rate + low hit rate = thrashing. "
    "All three signals are required."
)

heading("Heterogeneous Agents: Asymmetric Destruction", 2)
para(
    "Real deployments mix short (4-step), medium (12-step), and long (24-step) agents. "
    "When these share the same cache, long agents' massive insertions evict short agents' "
    "cached prefixes. Short agents lose 35 percentage points of hit rate. Long agents barely notice."
)
img("fig07_utilization_lies_hetero.png")
caption("Figure 10: Left — Cache utilization stays high (~83%) while hit rate collapses to 18%. "
        "Right — Mixing short+long agents (21% hit rate) is worse than all-medium (56%).")
spacer()
para(
    "The capacity planning rule: provision cache for the largest agent type's full working set × "
    "its concurrent count. A pool with 25% long agents behaves as if it were 100% long agents "
    "from a cache-pressure perspective."
)

doc.add_page_break()
print("Writing Part IV...")

# ══════════════════════════════════════════════════════════════
# PART IV: IO-AWARE SCHEDULER
# ══════════════════════════════════════════════════════════════
heading("Part IV: The IO-Aware Scheduler", 1)

para(
    "Architectural compression (MLA, Engram, TurboQuant) reduces KV footprint per token. "
    "But it doesn't prevent thrashing — it just shifts the boundary to higher concurrency "
    "or longer contexts. The system-level scheduler is the last line of defense."
)

heading("What Makes a Scheduler IO-Aware", 2)

para("1. Working-Set-Aware Admission Control", bold=True)
para(
    "Before admitting a new session, check: sum(active_session_blocks) + new_session_max_blocks "
    "≤ α × cache_capacity, where α < 1.0 provides a safety margin. The thrashing cliff is a hard "
    "phase boundary — partial overcommit immediately delivers the full 5× penalty."
)

para("2. Replace Utilization Monitoring With the Diagnostic Triad", bold=True)
para(
    "Stop monitoring cache utilization as a health metric. The actionable signal is: "
    "high utilization + high eviction rate + low hit rate = thrashing. "
    "Utilization is near 100% in both healthy and thrashing regimes."
)

para("3. Session-Aware Eviction Over Pure LRU", bold=True)
para(
    "LRU is adversarial for agentic workloads: the least-recently-used block is always the one "
    "from the session that will request it next. A session-aware policy protects active sessions' "
    "blocks and only evicts cold (completed session) blocks."
)

para("4. Type-Aware Scheduling and Cache Partitioning", bold=True)
para(
    "When agent types are known at dispatch time, partition cache space by agent type to prevent "
    "cross-type eviction. Short agents should have a dedicated segment that long agents cannot "
    "pollute, eliminating the asymmetric fairness failure."
)

para("5. Dynamic Concurrency Limits", bold=True)
para(
    "Rather than a fixed batch size cap, dynamically adjust concurrent session count based on "
    "measured working set. The capacity planning rule:"
)
para("Required cache blocks ≥ N_concurrent × max_blocks_per_session_type", bold=True)
spacer()

table(
    ["Agent Type", "Max Blocks/Session", "8 Concurrent Requires", "Safe Cache"],
    [
        ["Short (4 steps)", "39", "312 blocks", "~400 blocks"],
        ["Medium (12 steps)", "169", "1,352 blocks", "~1,400 blocks"],
        ["Long (24 steps)", "469", "3,752 blocks", "~4,000 blocks"],
    ]
)
spacer()
para(
    "For mixed pools, dimension against the largest type present, not the average."
)

heading("The Memory Hierarchy Extends Beyond GPU", 2)
para(
    "Our Qwen3-Coder-Next experiment (80B total / 3B active) reveals a counterintuitive result: "
    "CPU-resident expert weights can beat GPU transfer by 30% per layer (0.81 ms vs 1.16 ms). "
    "Reading 30 MB from DDR5 at 200 GB/s (0.15 ms) beats pushing the same data over PCIe at "
    "25.2 GB/s (1.16 ms). The future IO-aware scheduler needs to reason about this extended "
    "hierarchy: GPU HBM for hot KV blocks, host DRAM for Engram tables and cold KV, "
    "PCIe as a constrained interconnect."
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# CONCLUSION
# ══════════════════════════════════════════════════════════════
heading("Conclusion: The Full Stack", 1)

img("fig10_full_compression_stack.png")
caption("Figure 11: The full compression stack compounds: MHA FP16 (2,560 GB) → MLA + TQ + "
        "Prefix Cache (1.9 GB effective). Each layer is necessary; none alone is sufficient.")
spacer()

table(
    ["Layer", "Technique", "What It Reduces", "Compression"],
    [
        ["Architecture", "MHA → GQA → MLA", "KV bytes per token", "14.2×"],
        ["Attention sparsity", "Sparse attention", "Tokens attended per query", "variable"],
        ["Conditional memory", "Engram", "HBM expert loads", "24% per layer"],
        ["Quantization", "TurboQuant", "KV bits per element", "5.33×"],
        ["Cache management", "Prefix caching", "Redundant prefill compute", "up to 85%"],
        ["Scheduling", "IO-aware scheduler", "Thrashing-wasted compute", "up to 81%"],
    ]
)
spacer()

para(
    "None of these are redundant. MLA without scheduling still thrashes. Prefix caching without "
    "admission control still thrashes. TurboQuant on top of GQA still doesn't fit a million-token "
    "context in a single H100. The stack composes — MLA + TurboQuant + prefix caching + IO-aware "
    "scheduling is qualitatively better than any subset."
)
para(
    "The uncomfortable trajectory: faster GPUs widen the IO gap. H100 delivers 3.2× more FP16 "
    "TFLOPS than A100, but only 2× more PCIe bandwidth. Each GPU generation, compute-to-IO "
    "improves, and KV IO becomes relatively more of the bottleneck. The architectural and "
    "system-level techniques described here will become more important, not less."
)
para(
    "The problem is fundamental: autoregressive generation reads an amount of memory proportional "
    "to context length to produce a single token. Until the architecture of decoding itself changes, "
    "the memory wall will keep moving. Every technique here buys time. The scheduler is what "
    "determines whether that time is well spent."
)

spacer()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run(
    "Experiments run using the Vidur + InferSim simulation framework. "
    "All numerical results are from simulation."
)
run.font.size = Pt(9)
run.font.color.rgb = RGBColor(0x95, 0x9D, 0xA5)
run.italic = True

# ── Save ──────────────────────────────────────────────────────
doc.save(OUT)
print(f"\nSaved: {OUT}")
print("Done!")
