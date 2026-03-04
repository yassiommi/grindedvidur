#!/usr/bin/env python3
"""Experiment 2: DeepSeek-V3 batch size sweep.

Directly vary batch_size_cap with a saturating QPS so the scheduler
always forms batches up to the cap. Analyzes per-batch IO vs compute
at each batch size to show how the IO/compute ratio changes.
"""
import json
import os
import subprocess
import sys
import pandas as pd

RESULTS_DIR = "example_outputs/experiments/deepseek_batch_sweep"
os.makedirs(RESULTS_DIR, exist_ok=True)

# High enough QPS to keep the scheduler saturated at all batch size caps
SATURATING_QPS = 100.0
# Enough requests to fill multiple full batches at every cap value
NUM_REQUESTS = 1024


def run_sim(extra_args: list, label: str) -> str:
    """Run simulation and return the output dir created by this run."""
    sim_output = 'simulator_output'
    os.makedirs(sim_output, exist_ok=True)

    # Snapshot existing dirs so we can identify the new one after the run
    before = set(os.listdir(sim_output))

    cmd = [sys.executable, "-m", "vidur.main",
        "--replica_config_model_name", "deepseek-ai/DeepSeek-V3",
        "--replica_config_device", "a100",
        "--replica_config_network_device", "a100_dgx",
        "--replica_config_tensor_parallel_size", "8",
        "--replica_config_expert_parallel_size", "8",
        "--replica_config_enable_kv_prefetch",
        "--metrics_config_store_layer_metrics",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", str(NUM_REQUESTS),
        "--request_interval_generator_config_type", "poisson",
        "--poisson_request_interval_generator_config_qps", str(SATURATING_QPS),
    ] + extra_args

    print(f"\n  RUNNING: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    after = set(os.listdir(sim_output))
    new_dirs = sorted(after - before)
    if new_dirs:
        return os.path.join(sim_output, new_dirs[-1])

    raise RuntimeError(f"Sim produced no new output dir: {result.stderr[-300:]}")


def analyze_run(output_dir: str) -> dict:
    """Analyze per-batch IO vs compute crossover."""
    csv_path = os.path.join(output_dir, "layer_timings.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    decode = df[df['kv_cache_load_time'] > 0]
    if len(decode) == 0:
        return {'error': 'no decode batches'}

    # Per-batch stats: one row per batch (layer 0 has the same timing as all
    # other layers within a batch, so 'first' gives the correct per-batch value)
    batch_stats = decode.groupby('batch_id').agg({
        'kv_cache_load_time': 'first',
        'compute_time': 'first',
    }).reset_index()
    batch_stats['io_gt_compute'] = batch_stats['kv_cache_load_time'] > batch_stats['compute_time']
    batch_stats['ratio'] = batch_stats['kv_cache_load_time'] / batch_stats['compute_time'].clip(lower=1e-9)

    n_total = len(batch_stats)
    n_io_bound = batch_stats['io_gt_compute'].sum()

    result = {
        'total_decode_batches': int(n_total),
        'io_bound_batches': int(n_io_bound),
        'io_bound_pct': round(100 * n_io_bound / max(n_total, 1), 1),
        'avg_kv_load_ms': round(batch_stats['kv_cache_load_time'].mean(), 4),
        'avg_compute_ms': round(batch_stats['compute_time'].mean(), 4),
        'max_kv_load_ms': round(batch_stats['kv_cache_load_time'].max(), 4),
        'max_compute_ms': round(batch_stats['compute_time'].max(), 4),
        'median_ratio': round(batch_stats['ratio'].median(), 3),
        'p90_ratio': round(batch_stats['ratio'].quantile(0.90), 3),
        'p95_ratio': round(batch_stats['ratio'].quantile(0.95), 3),
    }
    return result


# ── Main sweep ────────────────────────────────────────────────────────
# Sweep batch_size_cap directly. Saturating QPS ensures the scheduler
# always has enough queued requests to fill batches up to the cap.
batch_size_caps = [1, 4, 16, 64, 128, 256]
all_results = []

for cap in batch_size_caps:
    try:
        out_dir = run_sim([
            "--sarathi_scheduler_config_batch_size_cap", str(cap),
        ], f"DeepSeek-V3 batch_size_cap={cap}")

        stats = analyze_run(out_dir)
        entry = {'batch_size_cap': cap, **stats}
        all_results.append(entry)

        print(f"  cap={cap:>4d}: {stats.get('total_decode_batches', 0)} decode batches, "
              f"{stats.get('io_bound_pct', 0)}% IO-bound, "
              f"avg kv_load={stats.get('avg_kv_load_ms', 0):.4f}ms, "
              f"median ratio={stats.get('median_ratio', 0):.3f}")

    except Exception as e:
        print(f"  cap={cap} FAILED: {e}")
        all_results.append({'batch_size_cap': cap, 'error': str(e)})

# Save results
sweep_df = pd.DataFrame(all_results)
sweep_df.to_csv(os.path.join(RESULTS_DIR, "batch_sweep_results.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "batch_sweep_summary.json"), 'w') as f:
    json.dump({
        'description': (
            'DeepSeek-V3 IO vs Compute: varying batch_size_cap at saturating QPS '
            f'({SATURATING_QPS} req/s, {NUM_REQUESTS} requests)'
        ),
        'model': 'deepseek-ai/DeepSeek-V3',
        'device': 'A100 (PCIe Gen4 31.5 GB/s)',
        'config': 'TP=8, EP=8, prefetch=ON, Sarathi scheduler',
        'sweep_results': all_results,
    }, f, indent=2, default=str)

# Print summary table
print("\n" + "="*90)
print("DeepSeek-V3 Batch Size Sweep - IO vs Compute Crossover")
print("="*90)
print(f"{'Cap':>5s} {'Batches':>8s} {'IO-bound%':>10s} "
      f"{'Avg IO(ms)':>11s} {'Avg Compute(ms)':>16s} {'Median Ratio':>13s}")
print("-"*90)
for r in all_results:
    if 'error' not in r:
        print(f"{r['batch_size_cap']:>5d} "
              f"{r.get('total_decode_batches', 0):>8d} "
              f"{r.get('io_bound_pct', 0):>9.1f}% "
              f"{r.get('avg_kv_load_ms', 0):>11.4f} "
              f"{r.get('avg_compute_ms', 0):>16.4f} "
              f"{r.get('median_ratio', 0):>13.3f}")
    else:
        print(f"{r['batch_size_cap']:>5d} {'FAILED':>8s}")
print(f"\nResults saved to {RESULTS_DIR}/")
