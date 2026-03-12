#!/usr/bin/env python3
"""Test and characterize Prefill-Decode Disaggregation (PDD) with KV cache I/O.

This script runs simulations to measure the impact of disaggregated inference
on KV cache transfer overhead and end-to-end latency.

Test Scenarios:
1. PDD with different PCIe bandwidths (Gen3, Gen4, Gen5)
2. PDD vs baseline (non-disaggregated)
3. Varying batch sizes and prefetch configurations
"""
import json
import os
import subprocess
import sys
import tempfile
import csv
import yaml
from typing import Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_ROOT, "example_outputs", "pdd_tests")
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_pdd_config(
    scheduler_type: str = "pdd",
    pcie_bandwidth_gbps: float = 50.0,
    kv_cache_bytes_per_token: float = 0.256,
    enable_kv_prefetch: bool = True,
    batch_size: int = 32,
    num_requests: int = 100,
    qps: float = 2.0,
) -> str:
    """Create a test configuration for PDD simulation.

    Args:
        scheduler_type: Scheduler type ('pdd' or 'vllm' for comparison)
        pcie_bandwidth_gbps: PCIe bandwidth in GB/s
        kv_cache_bytes_per_token: KV cache size per token
        enable_kv_prefetch: Whether to enable KV prefetch overlap
        batch_size: Maximum batch size
        num_requests: Number of requests to simulate
        qps: Queries per second

    Returns:
        Path to created YAML config file
    """
    config = {
        "cluster_config": {
            "num_replicas": 1,
            "replica_config": {
                "model_name": "llama-2-7b",
                "num_pipeline_stages": 2,
                "tensor_parallel_size": 1,
            },
        },
        "request_generator_config": {
            "type": "poisson",
            "qps": qps,
            "num_requests": num_requests,
            "seed": 42,
            "length_config": {
                "type": "zipf",
                "min_tokens": 64,
                "max_tokens": 512,
                "zipf_factor": 1.5,
            },
        },
        "replica_scheduler_config": {
            "name": scheduler_type,
            "batch_size_cap": batch_size,
            "block_size": 16,
            "watermark_blocks_fraction": 0.01,
        },
        "metrics_config": {
            "write_metrics": True,
            "write_json_trace": False,
            "store_plots": False,
            "store_utilization_metrics": False,
        },
    }

    # Add scheduler-specific config
    if scheduler_type == "pdd":
        config["replica_scheduler_config"].update({
            "pcie_bandwidth_gbps": pcie_bandwidth_gbps,
            "kv_cache_bytes_per_token": kv_cache_bytes_per_token,
            "enable_kv_prefetch": enable_kv_prefetch,
            "max_tokens_in_batch": batch_size * 256,
        })
    elif scheduler_type == "vllm":
        config["replica_scheduler_config"]["max_tokens_in_batch"] = batch_size * 256

    # Create temp file
    fd, path = tempfile.mkstemp(suffix=".yaml", dir=RESULTS_DIR)
    with os.fdopen(fd, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return path


def run_simulation(config_path: str, label: str) -> Tuple[bool, str]:
    """Run a simulation with the given config.

    Args:
        config_path: Path to YAML config file
        label: Label for this simulation (for logging)

    Returns:
        Tuple of (success, output_dir)
    """
    cmd = [sys.executable, "-m", "vidur.main", "--config", config_path]
    print(f"\n{'='*70}")
    print(f"  Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=_ROOT)

        # Find output directory
        for line in result.stderr.split("\n") + result.stdout.split("\n"):
            if "simulator_output/" in line:
                for part in line.split():
                    if "simulator_output/" in part:
                        output_dir = os.path.join(_ROOT, os.path.dirname(part))
                        if os.path.exists(output_dir):
                            print(f"  ✓ Simulation completed: {output_dir}")
                            return True, output_dir

        # Fallback: find most recent output directory
        sim_out = os.path.join(_ROOT, "simulator_output")
        if os.path.exists(sim_out):
            dirs = sorted([d for d in os.listdir(sim_out) if d.startswith("20")])
            if dirs:
                output_dir = os.path.join(sim_out, dirs[-1])
                print(f"  ✓ Simulation completed: {output_dir}")
                return True, output_dir

        print(f"  ✗ Simulation failed")
        print(f"  STDERR: {result.stderr[-500:]}")
        return False, ""

    except subprocess.TimeoutExpired:
        print(f"  ✗ Simulation timeout")
        return False, ""
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, ""


def extract_metrics(output_dir: str) -> Dict:
    """Extract KV cache I/O and performance metrics.

    Args:
        output_dir: Output directory from simulation

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Layer timings
    layer_csv = os.path.join(output_dir, "layer_timings.csv")
    if os.path.exists(layer_csv):
        rows = []
        with open(layer_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if rows:
            try:
                compute_times = []
                io_times = []
                total_times = []
                prefetch_savings = []
                kv_transfer_times = []
                batch_ids = set()
                stage_ids = set()

                for row in rows:
                    batch_ids.add(row.get("batch_id", ""))
                    stage_ids.add(row.get("stage_id", ""))
                    compute_times.append(float(row.get("compute_time", 0)))
                    io_times.append(float(row.get("io_time", 0)))
                    total_times.append(float(row.get("total_time", 0)))
                    prefetch_savings.append(
                        float(row.get("prefetch_overlap_savings", 0))
                    )

                    # KV transfer metrics (PDD)
                    try:
                        kv_time = float(row.get("inter_gpu_kv_transfer_time_ms", 0))
                        if kv_time > 0:
                            kv_transfer_times.append(kv_time)
                    except (ValueError, KeyError):
                        pass

                metrics["num_batches"] = len(batch_ids)
                metrics["num_stages"] = len(stage_ids)

                if compute_times:
                    metrics["avg_compute_time_ms"] = sum(compute_times) / len(
                        compute_times
                    )
                if io_times:
                    metrics["avg_io_time_ms"] = sum(io_times) / len(io_times)
                if total_times:
                    metrics["avg_total_time_ms"] = sum(total_times) / len(total_times)
                if prefetch_savings:
                    metrics["avg_prefetch_savings_ms"] = sum(prefetch_savings) / len(
                        prefetch_savings
                    )
                if kv_transfer_times:
                    metrics["avg_kv_transfer_time_ms"] = sum(kv_transfer_times) / len(
                        kv_transfer_times
                    )
                    metrics["max_kv_transfer_time_ms"] = max(kv_transfer_times)
                    metrics["min_kv_transfer_time_ms"] = min(kv_transfer_times)
                    metrics["total_kv_transfer_time_ms"] = sum(kv_transfer_times)

            except Exception as e:
                print(f"    Warning: Error parsing layer_timings.csv: {e}")

    # Request metrics
    request_csv = os.path.join(output_dir, "request_metrics.csv")
    if os.path.exists(request_csv):
        rows = []
        with open(request_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if rows:
            try:
                num_requests = len(rows)
                metrics["num_requests"] = num_requests

                ttft_times = []
                e2e_times = []
                token_counts = []
                kv_bytes_list = []
                kv_times_per_req = []

                for row in rows:
                    try:
                        arrived = float(row.get("arrived_at", 0))
                        prefill_done = float(row.get("prefill_completed_at", 0))
                        completed = float(row.get("completed_at", 0))

                        if prefill_done > arrived:
                            ttft_times.append((prefill_done - arrived) * 1000)
                        if completed > arrived:
                            e2e_times.append((completed - arrived) * 1000)

                        prefill_tokens = float(row.get("num_prefill_tokens", 0))
                        decode_tokens = float(row.get("num_decode_tokens", 0))
                        token_counts.append(prefill_tokens + decode_tokens)

                        # KV cache metrics
                        try:
                            kv_bytes = float(row.get("kv_cache_bytes", 0))
                            if kv_bytes > 0:
                                kv_bytes_list.append(kv_bytes)
                        except (ValueError, KeyError):
                            pass

                        try:
                            kv_time = float(row.get("kv_transfer_time_ms", 0))
                            if kv_time > 0:
                                kv_times_per_req.append(kv_time)
                        except (ValueError, KeyError):
                            pass

                    except (ValueError, KeyError):
                        pass

                if ttft_times:
                    metrics["avg_ttft_ms"] = sum(ttft_times) / len(ttft_times)
                if e2e_times:
                    metrics["avg_e2e_time_ms"] = sum(e2e_times) / len(e2e_times)
                if token_counts:
                    metrics["avg_tokens_per_request"] = sum(token_counts) / len(
                        token_counts
                    )
                if kv_bytes_list:
                    metrics["avg_kv_cache_bytes"] = sum(kv_bytes_list) / len(
                        kv_bytes_list
                    )
                    metrics["total_kv_cache_bytes"] = sum(kv_bytes_list)
                if kv_times_per_req:
                    metrics["avg_kv_transfer_time_per_request_ms"] = (
                        sum(kv_times_per_req) / len(kv_times_per_req)
                    )

            except Exception as e:
                print(f"    Warning: Error parsing request_metrics.csv: {e}")

    return metrics


def run_pdd_test_suite():
    """Run comprehensive PDD test scenarios."""
    print("\n" + "="*70)
    print("  Prefill-Decode Disaggregation (PDD) Test Suite")
    print("="*70)

    results = []

    # Test 1: PDD with different PCIe bandwidths
    print("\n\n=== TEST 1: PDD with Different PCIe Bandwidths ===")
    pcie_bandwidths = [16.0, 32.0, 50.0, 80.0]  # Gen3, Gen4, custom, Gen5
    for bw in pcie_bandwidths:
        label = f"PDD (PCIe {bw} GB/s)"
        config_path = create_pdd_config(
            scheduler_type="pdd",
            pcie_bandwidth_gbps=bw,
            num_requests=50,
            batch_size=32,
        )
        success, output_dir = run_simulation(config_path, label)

        if success:
            metrics = extract_metrics(output_dir)
            metrics["test"] = "pdd_bandwidth"
            metrics["scheduler"] = "pdd"
            metrics["pcie_bandwidth_gbps"] = bw
            results.append(metrics)
            print(f"    KV Transfer Time: {metrics.get('avg_kv_transfer_time_ms', 0):.2f} ms")
            print(f"    E2E Latency: {metrics.get('avg_e2e_time_ms', 0):.2f} ms")

    # Test 2: PDD vs Baseline (vLLM) comparison
    print("\n\n=== TEST 2: PDD vs Baseline (vLLM) ===")
    schedulers = [("pdd", "PDD (50 GB/s)"), ("vllm", "vLLM (Baseline)")]
    for sched_type, sched_label in schedulers:
        config_path = create_pdd_config(
            scheduler_type=sched_type,
            pcie_bandwidth_gbps=50.0,
            num_requests=50,
            batch_size=32,
        )
        success, output_dir = run_simulation(config_path, sched_label)

        if success:
            metrics = extract_metrics(output_dir)
            metrics["test"] = "scheduler_comparison"
            metrics["scheduler"] = sched_type
            results.append(metrics)
            print(f"    E2E Latency: {metrics.get('avg_e2e_time_ms', 0):.2f} ms")
            if sched_type == "pdd":
                print(f"    KV Transfer: {metrics.get('avg_kv_transfer_time_ms', 0):.2f} ms")

    # Test 3: Impact of KV prefetch overlap
    print("\n\n=== TEST 3: KV Prefetch Overlap Impact ===")
    prefetch_configs = [(True, "With Prefetch"), (False, "Without Prefetch")]
    for enable_prefetch, prefetch_label in prefetch_configs:
        config_path = create_pdd_config(
            scheduler_type="pdd",
            pcie_bandwidth_gbps=50.0,
            enable_kv_prefetch=enable_prefetch,
            num_requests=50,
            batch_size=32,
        )
        success, output_dir = run_simulation(config_path, f"PDD {prefetch_label}")

        if success:
            metrics = extract_metrics(output_dir)
            metrics["test"] = "kv_prefetch"
            metrics["scheduler"] = "pdd"
            metrics["enable_kv_prefetch"] = enable_prefetch
            results.append(metrics)
            print(f"    Prefetch Savings: {metrics.get('avg_prefetch_savings_ms', 0):.2f} ms")
            print(f"    E2E Latency: {metrics.get('avg_e2e_time_ms', 0):.2f} ms")

    # Save results
    if results:
        results_path = os.path.join(RESULTS_DIR, "pdd_test_results.csv")

        # Write CSV
        fieldnames = set()
        for r in results:
            fieldnames.update(r.keys())
        fieldnames = sorted(list(fieldnames))

        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n\n{'='*70}")
        print(f"  Results saved to: {results_path}")
        print(f"{'='*70}")

        # Print table
        print("\nResults Summary:")
        print("-" * 70)
        for result in results:
            print(f"\n  {result.get('scheduler', 'unknown')} - {result.get('test', 'unknown')}")
            if "pcie_bandwidth_gbps" in result:
                print(f"    PCIe BW: {result['pcie_bandwidth_gbps']:.1f} GB/s")
            if "enable_kv_prefetch" in result:
                print(f"    KV Prefetch: {result['enable_kv_prefetch']}")
            print(f"    E2E Latency: {result.get('avg_e2e_time_ms', 0):.2f} ms")
            if result.get("avg_kv_transfer_time_ms", 0) > 0:
                print(f"    KV Transfer: {result.get('avg_kv_transfer_time_ms', 0):.2f} ms")
            if result.get("avg_prefetch_savings_ms", 0) > 0:
                print(f"    Prefetch Savings: {result.get('avg_prefetch_savings_ms', 0):.2f} ms")

        # Summary statistics
        print(f"\n\n{'='*70}")
        print("  Summary Statistics")
        print(f"{'='*70}")

        pdd_results = [r for r in results if r.get("scheduler") == "pdd"]
        if pdd_results:
            kv_transfers = [
                r.get("avg_kv_transfer_time_ms", 0)
                for r in pdd_results
                if r.get("avg_kv_transfer_time_ms", 0) > 0
            ]
            e2e_times = [
                r.get("avg_e2e_time_ms", 0)
                for r in pdd_results
                if r.get("avg_e2e_time_ms", 0) > 0
            ]
            prefetch_savings = [
                r.get("avg_prefetch_savings_ms", 0)
                for r in pdd_results
                if r.get("avg_prefetch_savings_ms", 0) > 0
            ]

            print(f"\nPDD Metrics:")
            if kv_transfers:
                print(f"  Avg KV Transfer Time: {sum(kv_transfers)/len(kv_transfers):.2f} ms")
            if e2e_times:
                print(f"  Avg E2E Latency: {sum(e2e_times)/len(e2e_times):.2f} ms")
            if prefetch_savings:
                print(
                    f"  Avg Prefetch Savings: {sum(prefetch_savings)/len(prefetch_savings):.2f} ms"
                )

        vllm_results = [r for r in results if r.get("scheduler") == "vllm"]
        if vllm_results:
            e2e_times = [
                r.get("avg_e2e_time_ms", 0)
                for r in vllm_results
                if r.get("avg_e2e_time_ms", 0) > 0
            ]
            print(f"\nvLLM (Baseline) Metrics:")
            if e2e_times:
                print(f"  Avg E2E Latency: {sum(e2e_times)/len(e2e_times):.2f} ms")

    return results


if __name__ == "__main__":
    try:
        results = run_pdd_test_suite()
        print("\n✓ PDD test suite completed successfully")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n✗ Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
