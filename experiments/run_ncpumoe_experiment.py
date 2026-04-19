#!/usr/bin/env python3
"""Sweep n_cpu_moe from 0 to num_layers and find the threshold where
CPU-offloading becomes too expensive.

For each value of N, the first N MoE layers load weights over PCIe (slow)
while the remaining layers load from GPU HBM (fast).  We measure:
  - Per-batch model execution time
  - Mean E2E latency and TPOT
  - Per-layer weight load times and total overhead
"""

import csv
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Configuration ──────────────────────────────────────────────────
MODEL = "deepseek-ai/DeepSeek-V3"
DEVICE = "a100"
NETWORK = "a100_dgx"
TP = 8
EP = 8
NUM_REQUESTS = 64
NUM_LAYERS = 61

# Sweep values: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 61
SWEEP_VALUES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 61]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "example_outputs", "experiments", "ncpumoe_sweep")
os.makedirs(OUT_DIR, exist_ok=True)


def run_simulation(n_cpu_moe: int) -> dict:
    """Run a single simulation with the given n_cpu_moe value."""
    cmd = [
        sys.executable, "-m", "vidur.main",
        "--replica_config_model_name", MODEL,
        "--replica_config_device", DEVICE,
        "--replica_config_network_device", NETWORK,
        "--replica_config_tensor_parallel_size", str(TP),
        "--replica_config_expert_parallel_size", str(EP),
        "--replica_config_enable_kv_prefetch",
        "--replica_config_n_cpu_moe", str(n_cpu_moe),
        "--metrics_config_store_layer_metrics",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
    ]

    print(f"\n{'='*60}")
    print(f"  RUNNING: n_cpu_moe={n_cpu_moe}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=_ROOT)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return None

    # Find the output directory (most recent)
    sim_out = os.path.join(_ROOT, "simulator_output")
    output_dirs = sorted(
        [d for d in os.listdir(sim_out) if d.startswith("20")],
        reverse=True,
    )
    if not output_dirs:
        print("  ERROR: No output directory found")
        return None

    sim_dir = os.path.join(sim_out, output_dirs[0])

    # Parse layer timings
    layer_csv = os.path.join(sim_dir, "layer_timings.csv")
    cpu_layers_weight_load = []
    gpu_layers_weight_load = []
    all_compute = []
    all_total = []

    with open(layer_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = int(row["batch_id"])
            if bid < 2:  # Skip prefill batches
                continue
            li = int(row["layer_index"])
            wl = float(row["weight_load_time"])
            ct = float(row["compute_time"])
            tt = float(row["total_time"])
            all_compute.append(ct)
            all_total.append(tt)
            if li < n_cpu_moe:
                cpu_layers_weight_load.append(wl)
            else:
                gpu_layers_weight_load.append(wl)

    # Parse request metrics
    req_csv = os.path.join(sim_dir, "request_metrics.csv")
    e2e_times = []
    with open(req_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            e2e_times.append(float(row["request_e2e_time"]) * 1000)  # to ms

    avg_cpu_wl = np.mean(cpu_layers_weight_load) if cpu_layers_weight_load else 0
    avg_gpu_wl = np.mean(gpu_layers_weight_load) if gpu_layers_weight_load else 0
    avg_compute = np.mean(all_compute)
    avg_total = np.mean(all_total)

    # Total overhead from CPU layers per forward pass
    cpu_overhead_per_pass = avg_cpu_wl * n_cpu_moe  # ms total for CPU layers
    gpu_time_per_pass = avg_gpu_wl * (NUM_LAYERS - n_cpu_moe)

    stats = {
        "n_cpu_moe": n_cpu_moe,
        "avg_cpu_weight_load_ms": round(avg_cpu_wl, 4),
        "avg_gpu_weight_load_ms": round(avg_gpu_wl, 4),
        "avg_compute_ms": round(avg_compute, 4),
        "avg_layer_total_ms": round(avg_total, 4),
        "cpu_overhead_per_pass_ms": round(cpu_overhead_per_pass, 2),
        "gpu_time_per_pass_ms": round(gpu_time_per_pass, 2),
        "mean_e2e_latency_ms": round(np.mean(e2e_times), 2),
        "p99_e2e_latency_ms": round(np.percentile(e2e_times, 99), 2),
        "num_requests": len(e2e_times),
    }

    print(f"  n_cpu_moe={n_cpu_moe}: "
          f"cpu_wl={avg_cpu_wl:.2f}ms  gpu_wl={avg_gpu_wl:.2f}ms  "
          f"compute={avg_compute:.2f}ms  "
          f"cpu_overhead={cpu_overhead_per_pass:.1f}ms  "
          f"e2e={np.mean(e2e_times):.1f}ms")

    return stats


def plot_results(results: list):
    """Generate all plots from the sweep results."""

    n_vals = [r["n_cpu_moe"] for r in results]
    e2e = [r["mean_e2e_latency_ms"] for r in results]
    cpu_overhead = [r["cpu_overhead_per_pass_ms"] for r in results]
    avg_total = [r["avg_layer_total_ms"] for r in results]

    baseline_e2e = e2e[0]  # n_cpu_moe=0

    # ── Figure 1: E2E Latency vs n_cpu_moe ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_vals, e2e, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.axhline(y=baseline_e2e, color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Baseline (all GPU): {baseline_e2e:.0f} ms")

    # Find threshold (e.g. 10% degradation)
    threshold_10pct = baseline_e2e * 1.10
    ax.axhline(y=threshold_10pct, color="#f39c12", linestyle=":", linewidth=1.5,
               label=f"10% degradation: {threshold_10pct:.0f} ms")

    # Find max N where e2e < threshold
    max_n_10pct = 0
    for r in results:
        if r["mean_e2e_latency_ms"] <= threshold_10pct:
            max_n_10pct = r["n_cpu_moe"]

    ax.set_xlabel("Number of MoE Layers on CPU (n_cpu_moe)")
    ax.set_ylabel("Mean E2E Latency (ms)")
    ax.set_title(f"DeepSeek-V3: E2E Latency vs CPU-Offloaded MoE Layers\n"
                 f"(A100, TP={TP}, EP={EP}, {NUM_REQUESTS} requests)",
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    if max_n_10pct > 0:
        ax.annotate(f"Max N={max_n_10pct} within 10%",
                    xy=(max_n_10pct, threshold_10pct),
                    xytext=(max_n_10pct + 5, threshold_10pct + (e2e[-1] - e2e[0]) * 0.15),
                    fontsize=10, color="#e67e22", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "e2e_vs_ncpumoe.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 2: Per-layer overhead breakdown ──
    fig, ax = plt.subplots(figsize=(10, 5))

    cpu_wl = [r["avg_cpu_weight_load_ms"] for r in results]
    gpu_wl = [r["avg_gpu_weight_load_ms"] for r in results]
    compute = [r["avg_compute_ms"] for r in results]

    ax.bar(n_vals, cpu_wl, width=3.5, label="CPU Weight Load (PCIe)", color="#e74c3c",
           edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.bar(n_vals, gpu_wl, width=3.5, bottom=cpu_wl, label="GPU Weight Load (HBM)",
           color="#3498db", edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.plot(n_vals, compute, "s--", color="#2ecc71", linewidth=2, markersize=8,
            label="Compute Time", zorder=5)

    ax.set_xlabel("Number of MoE Layers on CPU (n_cpu_moe)")
    ax.set_ylabel("Time per Layer (ms)")
    ax.set_title("Per-Layer Timing: CPU-Offloaded vs GPU-Resident Expert Weights",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "layer_timing_vs_ncpumoe.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 3: Total CPU overhead per forward pass ──
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(n_vals, 0, cpu_overhead, alpha=0.3, color="#e74c3c",
                     label="CPU Weight Load Overhead")
    ax.plot(n_vals, cpu_overhead, "o-", color="#e74c3c", linewidth=2, markersize=8)

    gpu_total = [r["gpu_time_per_pass_ms"] for r in results]
    ax.fill_between(n_vals, 0, gpu_total, alpha=0.2, color="#3498db",
                     label="GPU Weight Load Time")
    ax.plot(n_vals, gpu_total, "s--", color="#3498db", linewidth=2, markersize=6)

    ax.set_xlabel("Number of MoE Layers on CPU (n_cpu_moe)")
    ax.set_ylabel("Total Weight Load Time per Forward Pass (ms)")
    ax.set_title("Weight Loading Overhead: CPU vs GPU Layers per Forward Pass",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "overhead_vs_ncpumoe.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 4: Latency breakdown stacked area ──
    fig, ax = plt.subplots(figsize=(10, 5))

    # Compute the components per forward pass
    cpu_wl_total = [r["avg_cpu_weight_load_ms"] * r["n_cpu_moe"] for r in results]
    gpu_wl_total = [r["avg_gpu_weight_load_ms"] * (NUM_LAYERS - r["n_cpu_moe"]) for r in results]
    compute_total = [r["avg_compute_ms"] * NUM_LAYERS for r in results]

    ax.stackplot(n_vals, compute_total, gpu_wl_total, cpu_wl_total,
                 labels=["Compute", "GPU Weight Load (HBM)", "CPU Weight Load (PCIe)"],
                 colors=["#2ecc71", "#3498db", "#e74c3c"], alpha=0.8)
    ax.set_xlabel("Number of MoE Layers on CPU (n_cpu_moe)")
    ax.set_ylabel("Total Time per Forward Pass (ms)")
    ax.set_title("Forward Pass Time Breakdown by Component",
                 fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "stacked_breakdown_vs_ncpumoe.png"), bbox_inches="tight")
    plt.close()

    # ── Figure 5: Memory savings estimate ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Expert params per layer per GPU
    hidden = 7168
    expert_inter = 2048
    local_experts = 256 // EP  # 32
    expert_bytes_per_layer = 3 * hidden * expert_inter * 2 * local_experts  # FP16
    expert_gb_per_layer = expert_bytes_per_layer / (1024**3)

    mem_saved = [n * expert_gb_per_layer for n in n_vals]
    latency_increase_pct = [(e - baseline_e2e) / baseline_e2e * 100 for e in e2e]

    ax = axes[0]
    ax.plot(n_vals, mem_saved, "o-", color="#9b59b6", linewidth=2, markersize=8)
    ax.set_xlabel("Number of MoE Layers on CPU")
    ax.set_ylabel("GPU Memory Saved (GB)")
    ax.set_title("GPU Memory Freed by CPU Offloading", fontweight="bold")
    ax.grid(alpha=0.3)
    for n, m in zip(n_vals[::2], mem_saved[::2]):
        ax.annotate(f"{m:.1f} GB", xy=(n, m), xytext=(n+1, m+0.3),
                   fontsize=8, color="#9b59b6")

    ax = axes[1]
    ax.plot(mem_saved, latency_increase_pct, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.set_xlabel("GPU Memory Saved (GB)")
    ax.set_ylabel("Latency Increase (%)")
    ax.set_title("Memory-Latency Tradeoff", fontweight="bold")
    ax.axhline(y=10, color="#f39c12", linestyle=":", linewidth=1.5, label="10% threshold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("DeepSeek-V3: Memory vs Latency Tradeoff (CPU-Offloaded Expert Weights)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "memory_latency_tradeoff.png"), bbox_inches="tight")
    plt.close()

    print(f"\nAll plots saved to {OUT_DIR}/")


def main():
    results = []

    for n in SWEEP_VALUES:
        stats = run_simulation(n)
        if stats:
            results.append(stats)

    # Save results
    with open(os.path.join(OUT_DIR, "ncpumoe_sweep_results.json"), "w") as f:
        json.dump({
            "description": "DeepSeek-V3: n_cpu_moe sweep to find CPU-offload threshold",
            "model": MODEL,
            "device": f"{DEVICE} (TP={TP}, EP={EP})",
            "num_requests": NUM_REQUESTS,
            "num_layers": NUM_LAYERS,
            "results": results,
        }, f, indent=2)

    with open(os.path.join(OUT_DIR, "ncpumoe_sweep_results.csv"), "w") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  n_cpu_moe SWEEP SUMMARY")
    print(f"{'='*70}")
    baseline = results[0]["mean_e2e_latency_ms"]
    print(f"  Baseline (n_cpu_moe=0): {baseline:.1f} ms E2E")
    print(f"  CPU weight load:  {results[-1]['avg_cpu_weight_load_ms']:.2f} ms/layer (PCIe)")
    print(f"  GPU weight load:  {results[0]['avg_gpu_weight_load_ms']:.2f} ms/layer (HBM)")
    print(f"  Ratio:            {results[-1]['avg_cpu_weight_load_ms'] / max(results[0]['avg_gpu_weight_load_ms'], 0.001):.1f}x slower")
    print()
    for r in results:
        pct = (r["mean_e2e_latency_ms"] - baseline) / baseline * 100
        flag = " ***" if pct > 10 else ""
        print(f"  N={r['n_cpu_moe']:3d}: e2e={r['mean_e2e_latency_ms']:8.1f}ms  "
              f"(+{pct:5.1f}%)  "
              f"cpu_overhead={r['cpu_overhead_per_pass_ms']:8.1f}ms{flag}")

    # Find threshold
    for r in results:
        if r["mean_e2e_latency_ms"] > baseline * 1.10:
            print(f"\n  >>> 10% degradation threshold: n_cpu_moe < {r['n_cpu_moe']}")
            break
    else:
        print(f"\n  >>> All values within 10% of baseline!")

    # Generate plots
    if len(results) >= 3:
        plot_results(results)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
