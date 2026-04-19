#!/usr/bin/env python3
"""Simple test to validate PDD implementation works."""

import sys
import os

# Add to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidur.config import (
    SimulationConfig,
    ClusterConfig,
    ReplicaConfig,
    PddSchedulerConfig,
    PoissonRequestIntervalGeneratorConfig,
    ZipfRequestLengthGeneratorConfig,
    SyntheticRequestGeneratorConfig,
    MetricsConfig,
)
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def test_pdd():
    """Run a simple PDD simulation."""
    print("\n" + "="*70)
    print("  Testing Prefill-Decode Disaggregation (PDD)")
    print("="*70)

    # Create configuration
    config = SimulationConfig(
        seed=42,
        cluster_config=ClusterConfig(
            num_replicas=1,
            replica_config=ReplicaConfig(
                model_name="meta-llama/Llama-2-7b-hf",
                num_pipeline_stages=2,
                tensor_parallel_size=1,
            ),
            replica_scheduler_config=PddSchedulerConfig(
                batch_size_cap=8,
                block_size=16,
                watermark_blocks_fraction=0.01,
                pcie_bandwidth_gbps=50.0,
                kv_cache_bytes_per_token=0.256,
                enable_kv_prefetch=True,
                max_tokens_in_batch=512,
            ),
        ),
        request_generator_config=SyntheticRequestGeneratorConfig(
            interval_generator_config=PoissonRequestIntervalGeneratorConfig(
                qps=2.0,
                seed=42,
            ),
            length_generator_config=ZipfRequestLengthGeneratorConfig(
                min_tokens=64,
                max_tokens=512,
                theta=1.5,
                seed=42,
            ),
            num_requests=10,  # Small test
        ),
        metrics_config=MetricsConfig(
            write_metrics=True,
            write_json_trace=False,
            store_plots=False,
            store_utilization_metrics=False,
        ),
    )

    set_seeds(config.seed)

    sched_cfg = config.cluster_config.replica_scheduler_config
    print(f"\nSimulation Config:")
    print(f"  Scheduler: PDD")
    if hasattr(sched_cfg, 'pcie_bandwidth_gbps'):
        print(f"  PCIe BW: {sched_cfg.pcie_bandwidth_gbps} GB/s")
        print(f"  KV Cache Per Token: {sched_cfg.kv_cache_bytes_per_token} bytes")
        print(f"  KV Prefetch Enabled: {sched_cfg.enable_kv_prefetch}")
    print(f"  Batch Size Cap: {sched_cfg.batch_size_cap}")
    print(f"  Num Requests: {config.request_generator_config.num_requests}")
    print(f"  Pipeline Stages: {config.cluster_config.replica_config.num_pipeline_stages}")

    try:
        print(f"\nRunning simulator...")
        simulator = Simulator(config)
        simulator.run()
        print(f"✓ Simulation completed successfully!")

        # Check if output files were created
        output_dirs = [d for d in os.listdir("simulator_output") if d.startswith("20")]
        if output_dirs:
            latest_output = os.path.join("simulator_output", sorted(output_dirs)[-1])
            print(f"\nOutput directory: {latest_output}")

            # Check for metrics files
            if os.path.exists(os.path.join(latest_output, "request_metrics.csv")):
                print(f"  ✓ Request metrics file created")
            if os.path.exists(os.path.join(latest_output, "layer_timings.csv")):
                print(f"  ✓ Layer timings file created")

            # Try to read and display some metrics
            import csv
            req_csv = os.path.join(latest_output, "request_metrics.csv")
            if os.path.exists(req_csv):
                with open(req_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        print(f"\n  Sample metrics from {len(rows)} requests:")
                        # Calculate averages
                        e2e_times = []
                        for row in rows:
                            try:
                                arrived = float(row.get("arrived_at", 0))
                                completed = float(row.get("completed_at", 0))
                                if completed > arrived:
                                    e2e_times.append((completed - arrived) * 1000)
                            except:
                                pass
                        if e2e_times:
                            avg_e2e = sum(e2e_times) / len(e2e_times)
                            print(f"    Avg E2E Latency: {avg_e2e:.2f} ms")

            # Check layer timings
            layer_csv = os.path.join(latest_output, "layer_timings.csv")
            if os.path.exists(layer_csv):
                with open(layer_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        print(f"\n  Sample metrics from {len(rows)} layer executions:")
                        # Look for KV transfer metrics
                        kv_transfers = []
                        for row in rows:
                            try:
                                kv_time = float(row.get("inter_gpu_kv_transfer_time_ms", 0))
                                if kv_time > 0:
                                    kv_transfers.append(kv_time)
                            except:
                                pass
                        if kv_transfers:
                            avg_kv = sum(kv_transfers) / len(kv_transfers)
                            print(f"    Avg KV Transfer Time: {avg_kv:.2f} ms")
                            print(f"    KV Transfers Found: {len(kv_transfers)}")
                        else:
                            print(f"    Note: No KV transfer times recorded in layer metrics")
                            print(f"    (This is expected if KV transfers happen at batch level)")

        return 0

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_pdd())
