#!/usr/bin/env python3
"""Qwen3-Coder-Next 80B-A3B FP8: CPU-Resident vs Dynamic GPU Transfer experiment.

Models two strategies for handling expert weights when VRAM < model weights:

Strategy A — CPU-Resident (n_cpu_moe = N):
  First N layers keep expert weights in CPU RAM and *compute the expert
  FFN on the CPU*.  Weights never cross PCIe — only the result tensor
  (~4 KB) is sent back to GPU.  Per-layer CPU time is dominated by CPU
  FLOPS and CPU memory bandwidth (DDR5 ~200–400 GB/s).
  Calibrated: ~0.81 ms/layer (from user's 37 layers ≈ 30 ms total).

Strategy B — Dynamic GPU Transfer:
  All layers run on GPU, but when VRAM cannot hold all expert weights,
  deficit layers' *active* expert weights (only num_experts_per_tok, not
  all 512) are loaded over PCIe per forward pass.  PCIe DMA and GPU SM
  can overlap: per-layer = max(PCIe_transfer, GPU_compute).
  Calibrated: ~1.16 ms/layer for PCIe (30 MB / 25.2 GB/s).

Key insight: CPU-resident (0.81 ms/layer) < Dynamic transfer (1.16 ms/layer)
because the CPU accesses weights from local DRAM at full memory bandwidth,
avoiding the PCIe bottleneck entirely.  But when VRAM can hold ALL layers
(deficit=0), dynamic transfer wins since all layers run at GPU speed.

Sweeps:
  - n_cpu_moe: 0, 5, 10, 15, 20, 25, 30, 37, 40, 48
  - VRAM levels: 24, 32, 40, 48, 64, 80 GB
  - Also runs simulator sweeps for validation
"""

import csv
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Model Architecture (Qwen3-Coder-Next 80B-A3B) ────────────────
NUM_LAYERS = 48
HIDDEN_DIM = 2048
EXPERT_INTERMEDIATE = 512  # MoE expert FFN intermediate dim
NUM_ROUTED_EXPERTS = 512
NUM_EXPERTS_PER_TOK = 10
NUM_SHARED_EXPERTS = 1
BYTES_PER_PARAM = 1  # FP8

# ── Hardware (A100 80GB, PCIe Gen4) ───────────────────────────────
GPU_FP16_TFLOPS = 312
HBM_BW_RAW = 2039  # GB/s
PCIE_BW_RAW = 31.5  # GB/s
BW_EFFICIENCY = 0.8
CPU_MEM_BW_GBS = 200  # DDR5 (conservative)

HBM_BW_EFF = HBM_BW_RAW * BW_EFFICIENCY   # 1631.2 GB/s
PCIE_BW_EFF = PCIE_BW_RAW * BW_EFFICIENCY  # 25.2 GB/s

# ── Derived Constants ─────────────────────────────────────────────
# Expert weight size: 3 matrices (gate, up, down) of hidden x intermediate
EXPERT_BYTES = 3 * HIDDEN_DIM * EXPERT_INTERMEDIATE * BYTES_PER_PARAM
EXPERT_MB = EXPERT_BYTES / (1024**2)

# Per-layer expert weight totals
ALL_EXPERTS_PER_LAYER_BYTES = NUM_ROUTED_EXPERTS * EXPERT_BYTES
ALL_EXPERTS_PER_LAYER_GB = ALL_EXPERTS_PER_LAYER_BYTES / (1024**3)

ACTIVE_EXPERTS_PER_LAYER_BYTES = NUM_EXPERTS_PER_TOK * EXPERT_BYTES
ACTIVE_EXPERTS_PER_LAYER_MB = ACTIVE_EXPERTS_PER_LAYER_BYTES / (1024**2)

TOTAL_EXPERT_WEIGHT_GB = NUM_LAYERS * ALL_EXPERTS_PER_LAYER_GB

# Non-expert model weights (attention, norms, embeddings) — estimated
NON_EXPERT_WEIGHT_GB = 8.0  # ~8 GB for attention, embedding, norms at FP8
KV_CACHE_OVERHEAD_GB = 5.0  # typical KV cache + runtime overhead
TOTAL_MODEL_GB = TOTAL_EXPERT_WEIGHT_GB + NON_EXPERT_WEIGHT_GB

# ── Per-layer timing (calibrated to user's real-world numbers) ────
# User reports: n_cpu_moe=37 → 37 CPU layers ≈ 30ms, 11 GPU layers ≈ 5ms

# GPU layer time: includes attention + MoE expert compute + norms
# 5ms / 11 layers ≈ 0.45ms per layer
GPU_LAYER_MS = 0.455

# CPU layer time: CPU computes expert FFN from local DRAM
# 30ms / 37 layers ≈ 0.81ms per layer
CPU_LAYER_MS = 0.811

# PCIe transfer time for active expert weights (Dynamic Transfer strategy)
# 10 active experts × 3.0 MB each = 30 MB per layer
# 30 MB / 25.2 GB/s = 1.16 ms
PCIE_ACTIVE_XFER_MS = (ACTIVE_EXPERTS_PER_LAYER_BYTES / (PCIE_BW_EFF * 1024**3)) * 1e3

# Dynamic transfer per deficit layer: max(PCIe, GPU_compute) due to DMA/SM overlap
DYNAMIC_DEFICIT_LAYER_MS = max(PCIE_ACTIVE_XFER_MS, GPU_LAYER_MS)

# ── Sweep Configuration ──────────────────────────────────────────
N_CPU_MOE_VALUES = [0, 5, 10, 15, 20, 25, 30, 37, 40, 48]
VRAM_LEVELS_GB = [24, 32, 40, 48, 64, 80]

OUT_DIR = "example_outputs/experiments/qwen3_cpumoe"
os.makedirs(OUT_DIR, exist_ok=True)


def cpu_resident_tps(n_cpu_moe: int) -> dict:
    """Compute TPS for CPU-resident strategy.

    CPU layers: expert FFN computed on CPU (weights in host DRAM).
    GPU layers: everything on GPU (weights in HBM).
    """
    cpu_total_ms = n_cpu_moe * CPU_LAYER_MS
    gpu_total_ms = (NUM_LAYERS - n_cpu_moe) * GPU_LAYER_MS
    total_ms = cpu_total_ms + gpu_total_ms
    tps = 1000.0 / total_ms if total_ms > 0 else float('inf')

    return {
        "n_cpu_moe": n_cpu_moe,
        "strategy": "cpu_resident",
        "cpu_layers_ms": round(cpu_total_ms, 3),
        "gpu_layers_ms": round(gpu_total_ms, 3),
        "total_ms_per_token": round(total_ms, 3),
        "tps": round(tps, 2),
    }


def dynamic_transfer_tps(vram_gb: float) -> dict:
    """Compute TPS for dynamic GPU transfer strategy.

    When VRAM < expert weights, deficit layers' ACTIVE expert weights
    are loaded over PCIe each pass.  PCIe DMA overlaps with GPU compute:
    deficit_layer_time = max(PCIe_transfer, GPU_compute).
    """
    available_for_experts = max(0, vram_gb - NON_EXPERT_WEIGHT_GB - KV_CACHE_OVERHEAD_GB)
    layers_that_fit = min(NUM_LAYERS, int(available_for_experts / ALL_EXPERTS_PER_LAYER_GB))
    deficit_layers = NUM_LAYERS - layers_that_fit

    # Deficit layers: active expert PCIe transfer overlapped with GPU compute
    deficit_ms = deficit_layers * DYNAMIC_DEFICIT_LAYER_MS
    # Resident layers: pure GPU compute
    resident_ms = layers_that_fit * GPU_LAYER_MS
    total_ms = deficit_ms + resident_ms

    # Transfer data: only active experts for deficit layers
    xfer_mb = deficit_layers * ACTIVE_EXPERTS_PER_LAYER_MB
    xfer_ms = (deficit_layers * ACTIVE_EXPERTS_PER_LAYER_BYTES / (PCIE_BW_EFF * 1024**3)) * 1e3

    tps = 1000.0 / total_ms if total_ms > 0 else float('inf')

    return {
        "vram_gb": vram_gb,
        "strategy": "dynamic_transfer",
        "layers_in_vram": layers_that_fit,
        "deficit_layers": deficit_layers,
        "xfer_mb": round(xfer_mb, 1),
        "pcie_xfer_ms": round(xfer_ms, 3),
        "deficit_ms": round(deficit_ms, 3),
        "resident_ms": round(resident_ms, 3),
        "total_ms_per_token": round(total_ms, 3),
        "tps": round(tps, 2),
    }


def run_simulator_sweep():
    """Run actual simulator sweeps for validation."""
    sim_results = []
    sim_n_values = [0, 10, 20, 37, 48]

    for n in sim_n_values:
        cmd = [
            sys.executable, "-m", "vidur.main",
            "--replica_config_model_name", "Qwen/Qwen3-Coder-Next-80B-A3B",
            "--replica_config_device", "a100",
            "--replica_config_network_device", "a100_pairwise_nvlink",
            "--replica_config_tensor_parallel_size", "1",
            "--replica_config_expert_parallel_size", "1",
            "--replica_config_enable_kv_prefetch",
            "--replica_config_n_cpu_moe", str(n),
            "--replica_config_weight_bytes_per_param", "1",
            "--metrics_config_store_layer_metrics",
            "--synthetic_request_generator_config_num_requests", "32",
        ]

        print(f"\n{'='*60}")
        print(f"  SIMULATOR: n_cpu_moe={n} (Qwen3-Coder-Next FP8)")
        print(f"{'='*60}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                err = result.stderr[-500:] if result.stderr else "unknown"
                print(f"  ERROR: {err}")
                sim_results.append({"n_cpu_moe": n, "sim_status": "error"})
                continue

            output_dirs = sorted(
                [d for d in os.listdir("simulator_output") if d.startswith("2026")],
                reverse=True,
            )
            if not output_dirs:
                sim_results.append({"n_cpu_moe": n, "sim_status": "no_output"})
                continue

            sim_dir = os.path.join("simulator_output", output_dirs[0])

            req_csv = os.path.join(sim_dir, "request_metrics.csv")
            e2e_times = []
            with open(req_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    e2e_times.append(float(row["request_e2e_time"]) * 1000)

            layer_csv = os.path.join(sim_dir, "layer_timings.csv")
            cpu_wl, gpu_wl, computes = [], [], []
            with open(layer_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["batch_id"]) < 2:
                        continue
                    li = int(row["layer_index"])
                    wl = float(row["weight_load_time"])
                    ct = float(row["compute_time"])
                    computes.append(ct)
                    if li < n:
                        cpu_wl.append(wl)
                    else:
                        gpu_wl.append(wl)

            sim_results.append({
                "n_cpu_moe": n,
                "sim_status": "ok",
                "sim_mean_e2e_ms": round(np.mean(e2e_times), 2),
                "sim_avg_cpu_wl_ms": round(np.mean(cpu_wl), 4) if cpu_wl else 0,
                "sim_avg_gpu_wl_ms": round(np.mean(gpu_wl), 4) if gpu_wl else 0,
                "sim_avg_compute_ms": round(np.mean(computes), 4),
            })
            print(f"  OK: e2e={np.mean(e2e_times):.1f}ms")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            sim_results.append({"n_cpu_moe": n, "sim_status": str(e)})

    return sim_results


def plot_results(cpu_results, dynamic_results, sim_results):
    """Generate all comparison plots."""

    n_vals = [r["n_cpu_moe"] for r in cpu_results]
    cpu_tps = [r["tps"] for r in cpu_results]

    # ── Figure 1: CPU-Resident TPS vs n_cpu_moe ──────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(n_vals, cpu_tps, "o-", color="#2ecc71", linewidth=2.5, markersize=9,
            label="CPU-Resident (CPU computes experts)", zorder=5)

    for r in cpu_results:
        if r["n_cpu_moe"] in (0, 37, 48):
            offset_y = 8 if r["n_cpu_moe"] != 48 else -12
            ax.annotate(f"{r['tps']:.0f} TPS\n({r['total_ms_per_token']:.1f} ms)",
                        xy=(r["n_cpu_moe"], r["tps"]),
                        xytext=(r["n_cpu_moe"] + 2.5, r["tps"] + offset_y),
                        fontsize=9, fontweight="bold", color="#27ae60",
                        arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.2))

    ax.axhline(y=30, color="#e67e22", linestyle=":", linewidth=1.5,
               label="User-reported: ~30 TPS (n=37)")

    ax.set_xlabel("Number of MoE Layers on CPU (n_cpu_moe)", fontsize=12)
    ax.set_ylabel("Tokens Per Second (TPS)", fontsize=12)
    ax.set_title("Qwen3-Coder-Next 80B-A3B FP8: CPU-Resident Strategy\n"
                 "(CPU computes expert FFN from local DRAM — no PCIe weight transfer)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(cpu_tps) * 1.15)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cpu_resident_tps.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ── Figure 2: Dynamic Transfer TPS vs VRAM ───────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    vram_vals = [r["vram_gb"] for r in dynamic_results]
    dyn_tps = [r["tps"] for r in dynamic_results]
    deficit_layers = [r["deficit_layers"] for r in dynamic_results]

    ax.plot(vram_vals, dyn_tps, "s-", color="#e74c3c", linewidth=2.5, markersize=9,
            label="Dynamic GPU Transfer", zorder=5)

    ax2 = ax.twinx()
    ax2.bar(vram_vals, deficit_layers, width=5, alpha=0.15, color="#3498db", zorder=1)
    ax2.set_ylabel("Deficit Layers", fontsize=11, color="#3498db")
    ax2.tick_params(axis='y', labelcolor="#3498db")

    ax.axhline(y=20, color="#e67e22", linestyle=":", linewidth=1.5,
               label="User-reported: ~20 TPS (dynamic)")

    for r in dynamic_results:
        ax.annotate(f"{r['tps']:.0f} TPS\n{r['deficit_layers']} deficit",
                    xy=(r["vram_gb"], r["tps"]),
                    xytext=(r["vram_gb"] - 4, r["tps"] + max(dyn_tps) * 0.07),
                    fontsize=8, fontweight="bold", color="#c0392b", ha="center")

    ax.set_xlabel("Available VRAM (GB)", fontsize=12)
    ax.set_ylabel("Tokens Per Second (TPS)", fontsize=12)
    ax.set_title("Qwen3-Coder-Next 80B-A3B FP8: Dynamic GPU Transfer Strategy\n"
                 "(active expert weights loaded over PCIe Gen4 for deficit layers)",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(dyn_tps) * 1.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dynamic_transfer_tps.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ── Figure 3: Head-to-Head Comparison (2 panels) ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: sweep n_cpu_moe, compare to dynamic at deficit=n (matched)
    ax = axes[0]
    matched_dyn_tps = []
    for n in n_vals:
        # What VRAM gives deficit=n? deficit=n means layers_fit=48-n
        # available = (48-n) * 1.5 GB, so VRAM = available + 13
        needed_vram = (NUM_LAYERS - n) * ALL_EXPERTS_PER_LAYER_GB + NON_EXPERT_WEIGHT_GB + KV_CACHE_OVERHEAD_GB
        d = dynamic_transfer_tps(needed_vram)
        matched_dyn_tps.append(d["tps"])

    ax.plot(n_vals, cpu_tps, "o-", color="#2ecc71", linewidth=2.5, markersize=8,
            label="CPU-Resident")
    ax.plot(n_vals, matched_dyn_tps, "s--", color="#e74c3c", linewidth=2.5, markersize=8,
            label="Dynamic Transfer (same deficit)")

    # Shade winner regions
    for i in range(len(n_vals)):
        winner_color = "#2ecc71" if cpu_tps[i] > matched_dyn_tps[i] else "#e74c3c"
        if i < len(n_vals) - 1:
            ax.axvspan(n_vals[i], n_vals[i+1], alpha=0.04, color=winner_color)

    ax.axhline(y=30, color="#e67e22", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(y=20, color="#e67e22", linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlabel("Deficit / n_cpu_moe (layers on CPU)", fontsize=12)
    ax.set_ylabel("TPS", fontsize=12)
    ax.set_title("Matched Deficit: CPU-Resident vs Dynamic", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(max(cpu_tps), max(matched_dyn_tps)) * 1.1)

    # Right: Fix n_cpu_moe=37, sweep VRAM for dynamic
    ax = axes[1]
    cpu_37 = cpu_resident_tps(37)

    vram_sweep = np.arange(20, 86, 1)
    dyn_sweep = [dynamic_transfer_tps(v)["tps"] for v in vram_sweep]

    ax.plot(vram_sweep, dyn_sweep, "-", color="#e74c3c", linewidth=2.5,
            label="Dynamic Transfer")
    ax.axhline(y=cpu_37["tps"], color="#2ecc71", linestyle="--", linewidth=2,
               label=f"CPU-Resident (N=37): {cpu_37['tps']:.0f} TPS")

    # Find VRAM crossover
    vram_crossover = None
    for v, t in zip(vram_sweep, dyn_sweep):
        if t >= cpu_37["tps"]:
            vram_crossover = v
            break

    if vram_crossover is not None:
        ax.axvline(x=vram_crossover, color="#8e44ad", linestyle=":", linewidth=1.5)
        ax.annotate(f"Crossover:\nVRAM={vram_crossover}GB",
                    xy=(vram_crossover, cpu_37["tps"]),
                    xytext=(vram_crossover - 12, cpu_37["tps"] + 30),
                    fontsize=10, fontweight="bold", color="#8e44ad",
                    arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5))
        ax.fill_betweenx([0, max(dyn_sweep) * 1.1], 20, vram_crossover,
                          alpha=0.06, color="#2ecc71")
        ax.fill_betweenx([0, max(dyn_sweep) * 1.1], vram_crossover, 86,
                          alpha=0.06, color="#e74c3c")
        ax.text((20 + vram_crossover) / 2, max(dyn_sweep) * 0.03, "CPU-Resident\nwins",
                ha="center", fontsize=9, color="#27ae60", fontweight="bold")
        ax.text((vram_crossover + 86) / 2, max(dyn_sweep) * 0.03, "Dynamic\nTransfer wins",
                ha="center", fontsize=9, color="#c0392b", fontweight="bold")

    ax.set_xlabel("Available VRAM (GB)", fontsize=12)
    ax.set_ylabel("TPS", fontsize=12)
    ax.set_title("n_cpu_moe = 37: VRAM Crossover", fontweight="bold", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(dyn_sweep) * 1.1)

    plt.suptitle("Qwen3-Coder-Next 80B-A3B FP8: CPU-Resident vs Dynamic GPU Transfer",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "strategy_comparison.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ── Figure 4: Per-Layer Timing Comparison ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Per-layer time comparison
    ax = axes[0]
    categories = [
        "GPU Layer\n(HBM-resident)",
        "CPU Layer\n(CPU-Resident)",
        "Deficit Layer\n(Dynamic Transfer)",
    ]
    times = [GPU_LAYER_MS, CPU_LAYER_MS, DYNAMIC_DEFICIT_LAYER_MS]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    bars = ax.bar(categories, times, 0.5, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f} ms", ha="center", fontweight="bold", fontsize=11)

    ax.set_ylabel("Time per Layer (ms)", fontsize=11)
    ax.set_title("Per-Layer Time: Three Execution Modes", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Annotate the key insight
    ax.annotate("CPU avoids PCIe\nbottleneck!",
                xy=(1, CPU_LAYER_MS), xytext=(1.5, CPU_LAYER_MS + 0.15),
                fontsize=10, color="#2980b9", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.2))

    # Right: Data movement comparison
    ax = axes[1]
    cat2 = [
        "CPU-Resident\n(result only)",
        "Dynamic Transfer\n(active experts)",
        "Full Layer Swap\n(all 512 experts)",
    ]
    data_kb = [
        HIDDEN_DIM * BYTES_PER_PARAM / 1024,  # ~2 KB result tensor
        ACTIVE_EXPERTS_PER_LAYER_MB * 1024,     # ~30 MB in KB
        ALL_EXPERTS_PER_LAYER_GB * 1024 * 1024,  # ~1.5 GB in KB
    ]
    colors2 = ["#2ecc71", "#e74c3c", "#95a5a6"]

    bars2 = ax.bar(cat2, data_kb, 0.5, color=colors2, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yscale("log")
    for bar, val in zip(bars2, data_kb):
        if val < 1024:
            label = f"{val:.1f} KB"
        elif val < 1024 * 1024:
            label = f"{val/1024:.1f} MB"
        else:
            label = f"{val/(1024*1024):.2f} GB"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                label, ha="center", fontweight="bold", fontsize=10)

    ax.set_ylabel("PCIe Data per Layer (KB, log scale)", fontsize=11)
    ax.set_title("PCIe Data Movement per Layer", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_layer_breakdown.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ── Figure 5: Heatmap — Optimal Strategy (n_cpu_moe × VRAM) ──
    fig, ax = plt.subplots(figsize=(12, 7))

    n_range = np.arange(0, 49, 1)
    vram_range = np.arange(20, 86, 1)
    # TPS difference: positive = CPU-resident wins
    tps_diff = np.zeros((len(vram_range), len(n_range)))

    for vi, v in enumerate(vram_range):
        dyn = dynamic_transfer_tps(v)
        for ni, n in enumerate(n_range):
            cpu = cpu_resident_tps(n)
            tps_diff[vi, ni] = cpu["tps"] - dyn["tps"]

    im = ax.imshow(tps_diff, aspect="auto", origin="lower", cmap="RdYlGn",
                   extent=[0, 48, 20, 85], vmin=-50, vmax=50)
    plt.colorbar(im, ax=ax, label="TPS advantage (green = CPU-Resident, red = Dynamic)")

    # Zero contour
    ax.contour(n_range, vram_range, tps_diff, levels=[0], colors=["black"],
               linewidths=2, linestyles="--")

    ax.set_xlabel("n_cpu_moe (CPU-Resident Layers)", fontsize=12)
    ax.set_ylabel("Available VRAM (GB)", fontsize=12)
    ax.set_title("Optimal Strategy Map: TPS Advantage\n"
                 "Qwen3-Coder-Next 80B-A3B FP8 on A100 PCIe Gen4",
                 fontweight="bold", fontsize=13)

    ax.plot(37, 80, "k*", markersize=15, zorder=10)
    ax.annotate("User config\n(n=37, VRAM=80GB)",
                xy=(37, 80), xytext=(25, 72),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "strategy_heatmap.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # ── Figure 6: Time Breakdown Stacked ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: CPU-Resident across n_cpu_moe
    ax = axes[0]
    cpu_ms = [r["cpu_layers_ms"] for r in cpu_results]
    gpu_ms = [r["gpu_layers_ms"] for r in cpu_results]

    ax.stackplot(n_vals, gpu_ms, cpu_ms,
                 labels=["GPU Layers (HBM)", "CPU Layers (DRAM)"],
                 colors=["#2ecc71", "#3498db"], alpha=0.8)
    ax.set_xlabel("n_cpu_moe", fontsize=11)
    ax.set_ylabel("Time per Token (ms)", fontsize=11)
    ax.set_title("CPU-Resident: Time Breakdown", fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    # Right: Dynamic Transfer across VRAM
    ax = axes[1]
    deficit_ms = [r["deficit_ms"] for r in dynamic_results]
    resident_ms = [r["resident_ms"] for r in dynamic_results]
    vram_labels = [r["vram_gb"] for r in dynamic_results]

    ax.stackplot(vram_labels, resident_ms, deficit_ms,
                 labels=["Resident Layers (HBM)", "Deficit Layers (PCIe)"],
                 colors=["#2ecc71", "#e74c3c"], alpha=0.8)
    ax.set_xlabel("VRAM (GB)", fontsize=11)
    ax.set_ylabel("Time per Token (ms)", fontsize=11)
    ax.set_title("Dynamic Transfer: Time Breakdown", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.suptitle("Per-Token Time Breakdown by Strategy",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "time_breakdown_stacked.png"), bbox_inches="tight", dpi=150)
    plt.close()

    print(f"\nAll plots saved to {OUT_DIR}/")


def main():
    print("=" * 70)
    print("  Qwen3-Coder-Next 80B-A3B FP8: CPU-Resident vs Dynamic Transfer")
    print("=" * 70)

    print(f"\n  Model: Qwen3-Coder-Next 80B-A3B (FP8)")
    print(f"  Layers: {NUM_LAYERS}")
    print(f"  Experts: {NUM_ROUTED_EXPERTS} routed, {NUM_EXPERTS_PER_TOK} active/token, "
          f"{NUM_SHARED_EXPERTS} shared")
    print(f"  Expert size (FP8): {EXPERT_MB:.2f} MB ({HIDDEN_DIM}x{EXPERT_INTERMEDIATE}, gated)")
    print(f"  Active expert data/layer: {ACTIVE_EXPERTS_PER_LAYER_MB:.1f} MB "
          f"({NUM_EXPERTS_PER_TOK} experts)")
    print(f"  All expert data/layer: {ALL_EXPERTS_PER_LAYER_GB:.3f} GB "
          f"({NUM_ROUTED_EXPERTS} experts)")
    print(f"  Total expert weights: {TOTAL_EXPERT_WEIGHT_GB:.1f} GB")
    print(f"  Total model weight (est.): {TOTAL_MODEL_GB:.1f} GB")
    print(f"\n  Hardware: A100 80GB, PCIe Gen4")
    print(f"  HBM BW (eff): {HBM_BW_EFF:.1f} GB/s")
    print(f"  PCIe BW (eff): {PCIE_BW_EFF:.1f} GB/s")
    print(f"  CPU Mem BW: {CPU_MEM_BW_GBS} GB/s (DDR5)")
    print(f"\n  Per-layer timing (calibrated to user data):")
    print(f"    GPU layer: {GPU_LAYER_MS:.3f} ms  (weights in HBM)")
    print(f"    CPU layer: {CPU_LAYER_MS:.3f} ms  (computed on CPU from DRAM)")
    print(f"    Dynamic deficit layer: {DYNAMIC_DEFICIT_LAYER_MS:.3f} ms  "
          f"(PCIe active-expert xfer + GPU compute, overlapped)")
    print(f"    PCIe active-expert xfer: {PCIE_ACTIVE_XFER_MS:.3f} ms  "
          f"({ACTIVE_EXPERTS_PER_LAYER_MB:.0f} MB / {PCIE_BW_EFF:.1f} GB/s)")

    # ── Strategy A: CPU-Resident sweep ────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Strategy A: CPU-Resident (expert FFN computed on CPU)")
    print(f"{'─'*70}")

    cpu_results = []
    for n in N_CPU_MOE_VALUES:
        r = cpu_resident_tps(n)
        cpu_results.append(r)
        print(f"  N={n:3d}: cpu={r['cpu_layers_ms']:7.2f}ms  "
              f"gpu={r['gpu_layers_ms']:6.2f}ms  "
              f"total={r['total_ms_per_token']:7.2f}ms  "
              f"TPS={r['tps']:7.1f}")

    # ── Strategy B: Dynamic Transfer sweep ────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Strategy B: Dynamic GPU Transfer (active-expert PCIe load)")
    print(f"{'─'*70}")

    dynamic_results = []
    for vram in VRAM_LEVELS_GB:
        r = dynamic_transfer_tps(vram)
        dynamic_results.append(r)
        print(f"  VRAM={vram:3d}GB: fit={r['layers_in_vram']:2d}  "
              f"deficit={r['deficit_layers']:2d} ({r['xfer_mb']:.0f}MB)  "
              f"total={r['total_ms_per_token']:7.2f}ms  "
              f"TPS={r['tps']:7.1f}")

    # ── Head-to-head: user's scenario ─────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  User's Scenario Validation")
    print(f"{'─'*70}")

    cpu_37 = cpu_resident_tps(37)
    print(f"  CPU-Resident (N=37): {cpu_37['total_ms_per_token']:.2f} ms → "
          f"{cpu_37['tps']:.1f} TPS  (user: ~30 TPS)")

    # Find the VRAM that gives deficit=37 (same layers offloaded)
    vram_for_deficit_37 = (NUM_LAYERS - 37) * ALL_EXPERTS_PER_LAYER_GB + NON_EXPERT_WEIGHT_GB + KV_CACHE_OVERHEAD_GB
    dyn_37 = dynamic_transfer_tps(vram_for_deficit_37)
    print(f"  Dynamic Transfer (deficit=37, VRAM≈{vram_for_deficit_37:.0f}GB): "
          f"{dyn_37['total_ms_per_token']:.2f} ms → "
          f"{dyn_37['tps']:.1f} TPS  (user: ~20 TPS)")

    # Also test at 80GB VRAM
    dyn_80 = dynamic_transfer_tps(80)
    print(f"  Dynamic Transfer (VRAM=80GB): "
          f"{dyn_80['total_ms_per_token']:.2f} ms → "
          f"{dyn_80['tps']:.1f} TPS  (deficit={dyn_80['deficit_layers']} layers)")

    delta_ms = DYNAMIC_DEFICIT_LAYER_MS - CPU_LAYER_MS
    print(f"\n  Per-layer advantage: CPU-Resident is {delta_ms:.3f} ms/layer faster")
    print(f"  Reason: CPU reads weights from local DRAM ({CPU_MEM_BW_GBS} GB/s), "
          f"Dynamic must cross PCIe ({PCIE_BW_EFF:.1f} GB/s)")
    print(f"  But when deficit=0 (all in VRAM), Dynamic wins: "
          f"{NUM_LAYERS * GPU_LAYER_MS:.1f}ms → {1000/(NUM_LAYERS*GPU_LAYER_MS):.0f} TPS")

    # ── Crossover analysis ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Crossover Analysis")
    print(f"{'─'*70}")

    for v in range(20, 86):
        dyn_r = dynamic_transfer_tps(v)
        if dyn_r["tps"] >= cpu_37["tps"]:
            print(f"  Dynamic Transfer beats CPU-Resident (N=37) at VRAM >= {v} GB")
            print(f"    Dynamic: {dyn_r['tps']:.1f} TPS (deficit={dyn_r['deficit_layers']})")
            print(f"    CPU-Resident: {cpu_37['tps']:.1f} TPS")
            break

    # For each VRAM level, find optimal n_cpu_moe
    print(f"\n  Optimal n_cpu_moe per VRAM level:")
    for vram in [24, 32, 40, 48, 64, 80]:
        dyn = dynamic_transfer_tps(vram)
        best_n, best_tps = 0, 0
        for n in range(0, 49):
            r = cpu_resident_tps(n)
            if r["tps"] > best_tps:
                best_tps = r["tps"]
                best_n = n
        winner = "CPU-Resident" if best_tps > dyn["tps"] else "Dynamic"
        winner_tps = max(best_tps, dyn["tps"])
        print(f"    VRAM={vram:3d}GB: best CPU-Resident N={best_n} ({best_tps:.0f} TPS) vs "
              f"Dynamic ({dyn['tps']:.0f} TPS) → {winner} wins ({winner_tps:.0f} TPS)")

    # ── Run simulator for validation ──────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Simulator Validation")
    print(f"{'─'*70}")

    sim_results = run_simulator_sweep()

    # ── Save all results ──────────────────────────────────────────
    all_results = {
        "description": "Qwen3-Coder-Next 80B-A3B FP8: CPU-Resident vs Dynamic GPU Transfer",
        "model": {
            "name": "Qwen/Qwen3-Coder-Next-80B-A3B",
            "layers": NUM_LAYERS,
            "routed_experts": NUM_ROUTED_EXPERTS,
            "active_experts_per_tok": NUM_EXPERTS_PER_TOK,
            "shared_experts": NUM_SHARED_EXPERTS,
            "hidden_dim": HIDDEN_DIM,
            "expert_intermediate": EXPERT_INTERMEDIATE,
            "bytes_per_param": BYTES_PER_PARAM,
            "total_expert_weight_gb": round(TOTAL_EXPERT_WEIGHT_GB, 2),
            "total_model_weight_gb": round(TOTAL_MODEL_GB, 2),
        },
        "hardware": {
            "device": "A100 80GB",
            "hbm_bw_eff_gbs": HBM_BW_EFF,
            "pcie_bw_eff_gbs": PCIE_BW_EFF,
            "cpu_mem_bw_gbs": CPU_MEM_BW_GBS,
        },
        "per_layer_timing": {
            "gpu_layer_ms": GPU_LAYER_MS,
            "cpu_layer_ms": CPU_LAYER_MS,
            "dynamic_deficit_layer_ms": DYNAMIC_DEFICIT_LAYER_MS,
            "pcie_active_xfer_ms": round(PCIE_ACTIVE_XFER_MS, 4),
            "expert_bytes_fp8": EXPERT_BYTES,
            "active_experts_per_layer_mb": round(ACTIVE_EXPERTS_PER_LAYER_MB, 2),
            "all_experts_per_layer_gb": round(ALL_EXPERTS_PER_LAYER_GB, 3),
        },
        "cpu_resident_results": cpu_results,
        "dynamic_transfer_results": dynamic_results,
        "simulator_results": sim_results,
    }

    with open(os.path.join(OUT_DIR, "qwen3_experiment_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(OUT_DIR, "cpu_resident_results.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=cpu_results[0].keys())
        writer.writeheader()
        writer.writerows(cpu_results)

    with open(os.path.join(OUT_DIR, "dynamic_transfer_results.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=dynamic_results[0].keys())
        writer.writeheader()
        writer.writerows(dynamic_results)

    plot_results(cpu_results, dynamic_results, sim_results)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"  Results saved to: {OUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
