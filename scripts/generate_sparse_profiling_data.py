#!/usr/bin/env python3
"""Generate synthetic sparse profiling CSVs for DeepSeek-V3.

Produces the same CSV format as `python -m vidur.profiling.sparse.main`
would generate on real hardware, but uses analytical timing models
calibrated to A100 GPU specifications.

Output files (matching the sparse profiler format):
  - sparse_mlp.csv: MoE MLP operation timings by num_tokens
  - mla_attention.csv: MLA projection timings by num_tokens
  - io_bandwidth.csv: HBM and PCIe bandwidth measurements

Usage:
    python scripts/generate_sparse_profiling_data.py [--output_dir DIR]
"""

import argparse
import os

import numpy as np
import pandas as pd

# ── DeepSeek-V3 architecture ─────────────────────────────────────
HIDDEN_SIZE = 7168
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
EXPERT_INTERMEDIATE_SIZE = 2048
NUM_SHARED_EXPERTS = 1

# MLA parameters
NUM_HEADS = 128
KV_LORA_RANK = 512
Q_LORA_RANK = 1536
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128

# ── A100 GPU specs ───────────────────────────────────────────────
GPU_FP16_TFLOPS = 312
HBM_BW_GBS = 2039
PCIE_BW_GBS = 31.5
BW_EFF = 0.8
HBM_EFF = HBM_BW_GBS * BW_EFF
PCIE_EFF = PCIE_BW_GBS * BW_EFF


def gemm_time_ms(m, k, n, mfu=0.15):
    """GEMM latency in ms from FLOPs."""
    flops = 2.0 * m * k * n
    gflops = flops / 1e9
    return gflops / (GPU_FP16_TFLOPS * 1024 * mfu) * 1e3


def add_noise(val, std_frac=0.05):
    """Add realistic measurement noise."""
    return max(0.001, val * (1 + np.random.normal(0, std_frac)))


def make_stats(val):
    """Create time_stats dict with min/max/mean/median/std."""
    samples = [add_noise(val) for _ in range(10)]
    return {
        "min": min(samples),
        "max": max(samples),
        "mean": np.mean(samples),
        "median": np.median(samples),
        "std": np.std(samples),
    }


def generate_sparse_mlp(num_tokens_list):
    """Generate sparse_mlp.csv matching the profiler output format."""
    rows = []
    for nt in num_tokens_list:
        # Router gate: hidden -> num_experts (small GEMM, low MFU)
        router_gate_ms = gemm_time_ms(nt, HIDDEN_SIZE, NUM_ROUTED_EXPERTS, mfu=0.05)
        # Router topk + softmax: small ops
        router_topk_ms = 0.005 * (nt / 1024)
        router_softmax_ms = 0.003 * (nt / 1024)

        # Expert dispatch: memory ops
        dispatch_ms = 0.01 * (nt / 1024) * NUM_EXPERTS_PER_TOK

        # Expert GEMM: grouped GEMM for all active experts
        # Each token goes through top-k experts, each with 3 GEMMs
        expert_gemm_ms = gemm_time_ms(
            nt, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, mfu=0.15
        ) * 3 * NUM_EXPERTS_PER_TOK

        # Expert combine: scatter-add
        combine_ms = 0.008 * (nt / 1024) * NUM_EXPERTS_PER_TOK

        # Shared expert: dense GEMM (higher MFU)
        shared_ms = gemm_time_ms(
            nt, HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, mfu=0.3
        ) * 3 * NUM_SHARED_EXPERTS

        # Block-level ops
        norm_ms = 0.01 * (nt / 1024)
        residual_ms = 0.005 * (nt / 1024)

        row = {}
        for name, val in [
            ("moe_router_gate", router_gate_ms),
            ("moe_router_topk", router_topk_ms),
            ("moe_router_softmax", router_softmax_ms),
            ("moe_expert_dispatch", dispatch_ms),
            ("moe_expert_gemm", expert_gemm_ms),
            ("moe_expert_combine", combine_ms),
            ("moe_shared_expert", shared_ms),
            ("moe_block_norm", norm_ms),
            ("moe_block_residual", residual_ms),
        ]:
            stats = make_stats(val)
            for k, v in stats.items():
                row[f"time_stats.{name}.{k}"] = v

        row["num_tokens"] = nt
        row["hidden_size"] = HIDDEN_SIZE
        row["num_routed_experts"] = NUM_ROUTED_EXPERTS
        row["num_experts_per_tok"] = NUM_EXPERTS_PER_TOK
        row["expert_intermediate_size"] = EXPERT_INTERMEDIATE_SIZE
        row["num_shared_experts"] = NUM_SHARED_EXPERTS
        rows.append(row)

    return pd.DataFrame(rows)


def generate_mla_attention(num_tokens_list):
    """Generate mla_attention.csv matching the profiler output format."""
    rows = []
    for nt in num_tokens_list:
        # Q compression: hidden -> q_lora_rank
        q_down_ms = gemm_time_ms(nt, HIDDEN_SIZE, Q_LORA_RANK, mfu=0.25)
        # Q decompression: q_lora_rank -> H * (nope + rope)
        q_up_ms = gemm_time_ms(
            nt, Q_LORA_RANK, NUM_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM), mfu=0.25
        )
        # KV compression: hidden -> kv_lora_rank + rope
        kv_down_ms = gemm_time_ms(nt, HIDDEN_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM, mfu=0.25)
        # KV decompression: kv_lora_rank -> H * (nope + v)
        kv_up_ms = gemm_time_ms(
            nt, KV_LORA_RANK, NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), mfu=0.25
        )
        # RoPE: element-wise ops
        rope_ms = 0.005 * (nt / 1024)
        # Attention score: simulated
        attn_ms = 0.01 * (nt / 1024)
        # Output projection: H * v_head_dim -> hidden
        o_proj_ms = gemm_time_ms(nt, NUM_HEADS * V_HEAD_DIM, HIDDEN_SIZE, mfu=0.25)
        # Block-level
        norm_ms = 0.01 * (nt / 1024)
        residual_ms = 0.005 * (nt / 1024)

        row = {}
        for name, val in [
            ("mla_q_down_proj", q_down_ms),
            ("mla_q_up_proj", q_up_ms),
            ("mla_kv_down_proj", kv_down_ms),
            ("mla_kv_up_proj", kv_up_ms),
            ("mla_rope", rope_ms),
            ("mla_attn_score", attn_ms),
            ("mla_o_proj", o_proj_ms),
            ("mla_block_norm", norm_ms),
            ("mla_block_residual", residual_ms),
        ]:
            stats = make_stats(val)
            for k, v in stats.items():
                row[f"time_stats.{name}.{k}"] = v

        row["num_tokens"] = nt
        row["hidden_size"] = HIDDEN_SIZE
        row["num_heads"] = NUM_HEADS
        row["kv_lora_rank"] = KV_LORA_RANK
        row["q_lora_rank"] = Q_LORA_RANK
        row["qk_nope_head_dim"] = QK_NOPE_HEAD_DIM
        row["qk_rope_head_dim"] = QK_ROPE_HEAD_DIM
        row["v_head_dim"] = V_HEAD_DIM
        rows.append(row)

    return pd.DataFrame(rows)


def generate_io_bandwidth():
    """Generate io_bandwidth.csv matching the profiler output format."""
    rows = []
    model = "deepseek-ai/DeepSeek-V3"

    # 1. Bandwidth sweep
    sizes_mb = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for size_mb in sizes_mb:
        size_bytes = int(size_mb * 1024 * 1024)
        for transfer_type, peak_bw in [
            ("hbm_read", HBM_EFF),
            ("pcie_h2d", PCIE_EFF),
            ("pcie_d2h", PCIE_EFF * 0.95),
        ]:
            # Smaller transfers have lower effective bandwidth
            size_factor = min(1.0, 0.5 + 0.5 * (size_mb / 16))
            eff_bw = peak_bw * size_factor
            latency_ms = (size_bytes / (eff_bw * 1024**3)) * 1e3
            rows.append({
                "transfer_type": transfer_type,
                "size_bytes": size_bytes,
                "size_mb": size_mb,
                "latency_ms": add_noise(latency_ms, 0.03),
                "bandwidth_gb_per_s": add_noise(eff_bw, 0.03),
                "model": model,
                "category": "bandwidth_sweep",
            })

    # 2. Expert weight loading
    expert_bytes = 3 * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2  # FP16
    for num_local in [1, 2, 4, 8, 16]:
        total_bytes = expert_bytes * num_local
        for source in ["hbm", "pcie"]:
            bw = HBM_EFF if source == "hbm" else PCIE_EFF
            latency_ms = (total_bytes / (bw * 1024**3)) * 1e3
            rows.append({
                "transfer_type": f"expert_weight_{source}",
                "size_bytes": total_bytes,
                "size_mb": total_bytes / (1024**2),
                "latency_ms": add_noise(latency_ms, 0.03),
                "bandwidth_gb_per_s": add_noise(bw, 0.03),
                "model": model,
                "category": "expert_weight",
                "num_local_experts": num_local,
            })

    # 3. KV cache transfers
    kv_bytes_per_token = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2  # MLA
    for num_tokens in [128, 256, 512, 1024, 2048, 4096]:
        for batch_size in [1, 4, 16, 64]:
            total_bytes = kv_bytes_per_token * num_tokens * batch_size
            for source in ["pcie", "hbm"]:
                bw = PCIE_EFF if source == "pcie" else HBM_EFF
                latency_ms = (total_bytes / (bw * 1024**3)) * 1e3
                rows.append({
                    "transfer_type": f"kv_cache_{source}",
                    "size_bytes": total_bytes,
                    "size_mb": total_bytes / (1024**2),
                    "latency_ms": add_noise(latency_ms, 0.03),
                    "bandwidth_gb_per_s": add_noise(bw, 0.03),
                    "model": model,
                    "category": "kv_cache",
                    "num_tokens": num_tokens,
                    "batch_size": batch_size,
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="data/profiling/compute/h100/deepseek_DeepSeek-V3",
        help="Output directory for profiling CSVs",
    )
    args = parser.parse_args()

    # Use repo root-relative paths
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    # Token counts matching the profiler's get_num_tokens_to_profile()
    num_tokens_list = sorted(set(
        list(range(1, 33))
        + list(range(32, 129, 8))
        + list(range(128, 513, 32))
        + list(range(512, 2049, 64))
        + list(range(2048, 4097, 128))
    ))

    print(f"Generating sparse profiling data for DeepSeek-V3...")
    print(f"Output: {output_dir}")
    print(f"Token counts: {len(num_tokens_list)} values from {num_tokens_list[0]} to {num_tokens_list[-1]}")

    mlp_df = generate_sparse_mlp(num_tokens_list)
    mlp_df.to_csv(os.path.join(output_dir, "mlp.csv"), index=False)
    print(f"  mlp.csv: {len(mlp_df)} rows")

    mla_df = generate_mla_attention(num_tokens_list)
    mla_df.to_csv(os.path.join(output_dir, "attention.csv"), index=False)
    print(f"  attention.csv: {len(mla_df)} rows")

    io_df = generate_io_bandwidth()
    io_df.to_csv(os.path.join(output_dir, "io.csv"), index=False)
    print(f"  io.csv: {len(io_df)} rows")

    print("Done.")


if __name__ == "__main__":
    main()
