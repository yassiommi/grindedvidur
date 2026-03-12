#!/usr/bin/env python3
"""Experiment 2: DeepSeek-V3 batch size sweep.

Vary QPS (request arrival rate) to produce different decode batch sizes,
then analyze per-batch IO vs compute to find the crossover.
"""
import json
import os
import subprocess
import sys
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "example_outputs", "experiments", "deepseek_batch_sweep")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_sim(extra_args: list, label: str) -> str:
    """Run simulation and return most recent output dir."""
    cmd = [sys.executable, "-m", "vidur.main",
        "--replica_config_model_name", "deepseek-ai/DeepSeek-V3",
        "--replica_config_device", "a100",
        "--replica_config_network_device", "a100_dgx",
        "--replica_config_tensor_parallel_size", "8",
        "--replica_config_expert_parallel_size", "8",
        "--replica_config_enable_kv_prefetch",
        "--metrics_config_store_layer_metrics",
        "--request_generator_config_type", "synthetic",
        "--synthetic_request_generator_config_num_requests", "64",
    ] + extra_args

    print(f"\n  RUNNING: {label}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=_ROOT)
    sim_out = os.path.join(_ROOT, 'simulator_output')
    dirs = sorted([d for d in os.listdir(sim_out) if d.startswith('20')])
    if dirs:
        return os.path.join(sim_out, dirs[-1])
    raise RuntimeError(f"Sim failed: {result.stderr[-300:]}")


def analyze_run(output_dir: str) -> dict:
    """Analyze per-batch IO vs compute crossover."""
    csv_path = os.path.join(output_dir, "layer_timings.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    decode = df[df['kv_cache_load_time'] > 0]
    if len(decode) == 0:
        return {'error': 'no decode batches'}

    # Per-batch stats (layer 0 values - same for all layers in a batch)
    batch_stats = decode.groupby('batch_id').agg({
        'kv_cache_load_time': 'first',
        'compute_time': 'first',
    }).reset_index()
    batch_stats['io_gt_compute'] = batch_stats['kv_cache_load_time'] > batch_stats['compute_time']
    batch_stats['ratio'] = batch_stats['kv_cache_load_time'] / batch_stats['compute_time'].clip(lower=1e-9)

    n_total = len(batch_stats)
    n_io_bound = batch_stats['io_gt_compute'].sum()

    return {
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


# ── Main sweep ────────────────────────────────────────────────────────
# Use different Poisson QPS to get different batch saturation levels
qps_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
all_results = []

for qps in qps_values:
    try:
        out_dir = run_sim([
            "--request_interval_generator_config_type", "poisson",
            "--poisson_request_interval_generator_config_qps", str(qps),
        ], f"DeepSeek-V3 QPS={qps}")

        stats = analyze_run(out_dir)
        entry = {'qps': qps, **stats}
        all_results.append(entry)

        print(f"  QPS={qps}: {stats.get('total_decode_batches', 0)} decode batches, "
              f"{stats.get('io_bound_pct', 0)}% IO-bound, "
              f"median ratio={stats.get('median_ratio', 0):.3f}")

    except Exception as e:
        print(f"  QPS={qps} FAILED: {e}")
        all_results.append({'qps': qps, 'error': str(e)})

# Save results
sweep_df = pd.DataFrame(all_results)
sweep_df.to_csv(os.path.join(RESULTS_DIR, "batch_sweep_results.csv"), index=False)

with open(os.path.join(RESULTS_DIR, "batch_sweep_summary.json"), 'w') as f:
    json.dump({
        'description': 'DeepSeek-V3 IO vs Compute: varying QPS to change decode batch sizes',
        'model': 'deepseek-ai/DeepSeek-V3',
        'device': 'A100 (PCIe Gen4 31.5 GB/s)',
        'config': 'TP=8, EP=8, prefetch=ON',
        'sweep_results': all_results,
    }, f, indent=2, default=str)

# Print summary table
print("\n" + "="*80)
print("DeepSeek-V3 Batch Size Sweep - IO vs Compute Crossover")
print("="*80)
print(f"{'QPS':>6s} {'Batches':>8s} {'IO-bound%':>10s} {'Avg IO(ms)':>11s} {'Avg Compute(ms)':>16s} {'Median Ratio':>13s}")
print("-"*80)
for r in all_results:
    if 'error' not in r:
        print(f"{r['qps']:>6.1f} {r.get('total_decode_batches', 0):>8d} "
              f"{r.get('io_bound_pct', 0):>9.1f}% "
              f"{r.get('avg_kv_load_ms', 0):>11.4f} "
              f"{r.get('avg_compute_ms', 0):>16.4f} "
              f"{r.get('median_ratio', 0):>13.3f}")
    else:
        print(f"{r['qps']:>6.1f} {'FAILED':>8s}")
print(f"\nResults saved to {RESULTS_DIR}/")
