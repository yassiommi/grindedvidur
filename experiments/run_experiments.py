#!/usr/bin/env python3
"""Run PCIe3 experiments and batch-size sweep for DeepSeek.

Experiment 1: PCIe Gen3 (16 GB/s) with Llama-2-7b and DeepSeek-V3
Experiment 2: DeepSeek batch-size sweep to find IO > compute threshold
"""
import json
import os
import subprocess
import sys
import pandas as pd
import shutil

RESULTS_DIR = "example_outputs/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────
def patch_pcie_bw(bw: float):
    """Monkey-patch A100 PCIe bandwidth in device_sku_config.py."""
    path = "vidur/config/device_sku_config.py"
    with open(path) as f:
        src = f.read()
    # Replace the A100 pcie line
    old = '    pcie_bandwidth_gb_per_s: float = 31.5  # PCIe Gen4 x16'
    new = f'    pcie_bandwidth_gb_per_s: float = {bw}  # PCIe Gen3 x16 (experiment)'
    if old not in src and 'experiment' in src:
        # already patched, replace experiment line
        import re
        src = re.sub(
            r'    pcie_bandwidth_gb_per_s: float = [\d.]+  # PCIe Gen3 x16 \(experiment\)',
            new, src, count=1)
    else:
        src = src.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(src)


def restore_pcie_bw():
    """Restore A100 PCIe bandwidth to Gen4 default."""
    path = "vidur/config/device_sku_config.py"
    with open(path) as f:
        src = f.read()
    import re
    src = re.sub(
        r'    pcie_bandwidth_gb_per_s: float = [\d.]+  # PCIe Gen3 x16 \(experiment\)',
        '    pcie_bandwidth_gb_per_s: float = 31.5  # PCIe Gen4 x16',
        src, count=1)
    with open(path, 'w') as f:
        f.write(src)


def run_sim(args: list, label: str) -> str:
    """Run simulation and return output dir."""
    cmd = [sys.executable, "-m", "vidur.main"] + args
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # Find output dir from logs
    for line in result.stderr.split('\n') + result.stdout.split('\n'):
        if 'layer_timings.csv' in line or 'request_metrics' in line:
            # extract path
            for part in line.split():
                if 'simulator_output/' in part:
                    return os.path.dirname(part)
    # fallback: find most recent simulator_output dir
    dirs = sorted([d for d in os.listdir('simulator_output') if d.startswith('20')])
    if dirs:
        return os.path.join('simulator_output', dirs[-1])
    raise RuntimeError(f"Simulation failed:\nSTDOUT: {result.stdout[-500:]}\nSTDERR: {result.stderr[-500:]}")


def extract_layer_stats(output_dir: str) -> dict:
    """Extract average per-layer stats from layer_timings.csv."""
    csv_path = os.path.join(output_dir, "layer_timings.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    decode = df[df['kv_cache_load_time'] > 0]
    if len(decode) == 0:
        decode = df  # fallback to all
    stats = {
        'avg_compute_time': decode['compute_time'].mean(),
        'avg_io_time': decode['io_time'].mean(),
        'avg_effective_io_time': decode['effective_io_time'].mean(),
        'avg_comm_time': decode['comm_time'].mean(),
        'avg_prefetch_savings': decode['prefetch_overlap_savings'].mean(),
        'avg_total_time': decode['total_time'].mean(),
        'avg_kv_cache_load_time': decode['kv_cache_load_time'].mean(),
        'io_exceeds_compute': bool(decode['kv_cache_load_time'].mean() > decode['compute_time'].mean()),
        'io_to_compute_ratio': decode['kv_cache_load_time'].mean() / max(decode['compute_time'].mean(), 1e-9),
        'num_decode_batches': len(decode['batch_id'].unique()),
    }
    return stats


def extract_request_stats(output_dir: str) -> dict:
    """Extract request-level summary."""
    csv_path = os.path.join(output_dir, "request_metrics.csv")
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {
        'num_requests': len(df),
        'mean_e2e_latency_ms': df['request_e2e_time'].mean() * 1000,
        'p99_e2e_latency_ms': df['request_e2e_time'].quantile(0.99) * 1000,
        'mean_ttft_ms': df['prefill_e2e_time'].mean() * 1000,
        'mean_tpot_ms': df['decode_time_execution_plus_preemption_normalized'].mean() * 1000,
    }


# ── Experiment 1: PCIe Gen3 (16 GB/s) ───────────────────────────────
def experiment_1():
    print("\n" + "="*70)
    print("  EXPERIMENT 1: PCIe Gen3 (16 GB/s) vs Gen4 (31.5 GB/s)")
    print("="*70)

    results = {}

    # 1a: Llama-2-7b with PCIe3
    patch_pcie_bw(16.0)
    try:
        out_dir = run_sim([
            "--replica_config_model_name", "meta-llama/Llama-2-7b-hf",
            "--replica_config_device", "a100",
            "--replica_config_enable_kv_prefetch",
            "--metrics_config_store_layer_metrics",
        ], "Llama-2-7b @ PCIe3 16GB/s")

        layer_stats = extract_layer_stats(out_dir)
        req_stats = extract_request_stats(out_dir)
        results['llama_2_7b_pcie3'] = {
            'pcie_bw_gb_s': 16.0,
            'layer_stats': layer_stats,
            'request_stats': req_stats,
            'output_dir': out_dir,
        }

        # Copy key outputs
        dest = os.path.join(RESULTS_DIR, "pcie3_llama_2_7b")
        os.makedirs(dest, exist_ok=True)
        for f in ['request_metrics.csv', 'config.json']:
            src = os.path.join(out_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, dest)
        for f in os.listdir(os.path.join(out_dir, 'plots')):
            if 'gantt_batch_4' in f or 'summary' in f:
                shutil.copy2(os.path.join(out_dir, 'plots', f), dest)
        # layer_timings sample
        csv_path = os.path.join(out_dir, 'layer_timings.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.head(200).to_csv(os.path.join(dest, 'layer_timings_sample.csv'), index=False)

        print(f"\n  Llama PCIe3 layer stats: {json.dumps(layer_stats, indent=2)}")
        print(f"  Llama PCIe3 request stats: {json.dumps(req_stats, indent=2)}")

    finally:
        restore_pcie_bw()

    # 1b: DeepSeek-V3 with PCIe3
    patch_pcie_bw(16.0)
    try:
        out_dir = run_sim([
            "--replica_config_model_name", "deepseek-ai/DeepSeek-V3",
            "--replica_config_device", "a100",
            "--replica_config_network_device", "a100_dgx",
            "--replica_config_tensor_parallel_size", "8",
            "--replica_config_expert_parallel_size", "8",
            "--replica_config_enable_kv_prefetch",
            "--metrics_config_store_layer_metrics",
        ], "DeepSeek-V3 @ PCIe3 16GB/s")

        layer_stats = extract_layer_stats(out_dir)
        req_stats = extract_request_stats(out_dir)
        results['deepseek_v3_pcie3'] = {
            'pcie_bw_gb_s': 16.0,
            'layer_stats': layer_stats,
            'request_stats': req_stats,
            'output_dir': out_dir,
        }

        dest = os.path.join(RESULTS_DIR, "pcie3_deepseek_v3")
        os.makedirs(dest, exist_ok=True)
        for f in ['request_metrics.csv', 'config.json']:
            src = os.path.join(out_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, dest)
        for f in os.listdir(os.path.join(out_dir, 'plots')):
            if 'gantt_batch_4' in f or 'summary' in f:
                shutil.copy2(os.path.join(out_dir, 'plots', f), dest)
        csv_path = os.path.join(out_dir, 'layer_timings.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df.head(200).to_csv(os.path.join(dest, 'layer_timings_sample.csv'), index=False)

        print(f"\n  DeepSeek PCIe3 layer stats: {json.dumps(layer_stats, indent=2)}")
        print(f"  DeepSeek PCIe3 request stats: {json.dumps(req_stats, indent=2)}")

    finally:
        restore_pcie_bw()

    return results


# ── Experiment 2: DeepSeek batch size sweep ──────────────────────────
def experiment_2():
    print("\n" + "="*70)
    print("  EXPERIMENT 2: DeepSeek-V3 Batch Size Sweep (IO vs Compute)")
    print("="*70)

    batch_sizes = [16, 32, 64, 128, 256, 512]
    sweep_results = []

    for bs in batch_sizes:
        print(f"\n--- Batch size cap = {bs} ---")
        try:
            out_dir = run_sim([
                "--replica_config_model_name", "deepseek-ai/DeepSeek-V3",
                "--replica_config_device", "a100",
                "--replica_config_network_device", "a100_dgx",
                "--replica_config_tensor_parallel_size", "8",
                "--replica_config_expert_parallel_size", "8",
                "--replica_config_enable_kv_prefetch",
                "--metrics_config_store_layer_metrics",
                "--vllm_scheduler_config_batch_size_cap", str(bs),
            ], f"DeepSeek-V3 batch_size_cap={bs}")

            layer_stats = extract_layer_stats(out_dir)
            req_stats = extract_request_stats(out_dir)

            entry = {
                'batch_size_cap': bs,
                **layer_stats,
                **req_stats,
            }
            sweep_results.append(entry)

            io_gt_compute = layer_stats.get('io_exceeds_compute', False)
            ratio = layer_stats.get('io_to_compute_ratio', 0)
            print(f"  compute={layer_stats.get('avg_compute_time', 0):.4f}ms "
                  f"io={layer_stats.get('avg_kv_cache_load_time', 0):.4f}ms "
                  f"ratio={ratio:.2f} "
                  f"{'*** IO > COMPUTE ***' if io_gt_compute else ''}")

        except Exception as e:
            print(f"  FAILED: {e}")
            sweep_results.append({'batch_size_cap': bs, 'error': str(e)})

    # Save sweep results
    dest = os.path.join(RESULTS_DIR, "deepseek_batch_sweep")
    os.makedirs(dest, exist_ok=True)
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(os.path.join(dest, "batch_sweep_results.csv"), index=False)

    # Save a Gantt from an interesting batch size
    # Find the threshold entry
    threshold_bs = None
    for entry in sweep_results:
        if entry.get('io_exceeds_compute', False):
            threshold_bs = entry['batch_size_cap']
            break

    with open(os.path.join(dest, "batch_sweep_summary.json"), 'w') as f:
        json.dump({
            'sweep_results': sweep_results,
            'threshold_batch_size': threshold_bs,
        }, f, indent=2, default=str)

    return sweep_results, threshold_bs


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    exp1_results = experiment_1()

    # Save exp1 summary
    with open(os.path.join(RESULTS_DIR, "pcie3_comparison.json"), 'w') as f:
        # Also include PCIe4 baseline numbers for comparison
        json.dump({
            'description': 'PCIe Gen3 (16 GB/s) vs Gen4 (31.5 GB/s) impact on KV cache I/O',
            'baseline_pcie4_bw_gb_s': 31.5,
            'experiment_pcie3_bw_gb_s': 16.0,
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'output_dir'}
                       for k, v in exp1_results.items()},
        }, f, indent=2, default=str)

    exp2_results, threshold = experiment_2()

    # Final summary
    print("\n" + "="*70)
    print("  EXPERIMENT SUMMARY")
    print("="*70)
    print("\nExperiment 1 - PCIe3 (16 GB/s):")
    for model, data in exp1_results.items():
        ls = data['layer_stats']
        print(f"  {model}:")
        print(f"    KV load time: {ls.get('avg_kv_cache_load_time', 0):.4f} ms")
        print(f"    Compute time: {ls.get('avg_compute_time', 0):.4f} ms")
        print(f"    IO/Compute ratio: {ls.get('io_to_compute_ratio', 0):.2f}x")
        print(f"    IO > Compute: {ls.get('io_exceeds_compute', False)}")

    print(f"\nExperiment 2 - DeepSeek Batch Size Sweep:")
    print(f"  Threshold where IO > Compute: batch_size_cap = {threshold}")
    for entry in exp2_results:
        if 'error' not in entry:
            print(f"  BS={entry['batch_size_cap']:>4d}: "
                  f"compute={entry.get('avg_compute_time', 0):.4f}ms "
                  f"io={entry.get('avg_kv_cache_load_time', 0):.4f}ms "
                  f"ratio={entry.get('io_to_compute_ratio', 0):.2f}x "
                  f"{'*** THRESHOLD ***' if entry.get('io_exceeds_compute') else ''}")

    print(f"\nResults saved to {RESULTS_DIR}/")
