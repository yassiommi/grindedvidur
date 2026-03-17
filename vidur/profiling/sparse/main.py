"""Sparse model profiler: MoE MLP + MLA attention + DSA sparse attention + IO bandwidth.

Profiles sparse model components (DeepSeek-V3, Mixtral, etc.) to collect
empirical timing data that replaces analytical FLOPs-based estimation.

Usage:
    python -m vidur.profiling.sparse.main \
        --model deepseek-ai/DeepSeek-V3 \
        --max_tokens 4096 \
        --output_dir profiling_outputs/sparse

Components profiled:
    1. MoE MLP: routing, grouped expert GEMM, shared experts, dispatch/combine
    2. MLA attention: Q/KV compression/decompression projections (DeepSeek models)
    3. DSA sparse attention: indexer loading, top-k selection, KV fetch, Q projections,
       attention calculation — the full DeepSeek Sparse Attention decode pipeline
    4. IO bandwidth: HBM read, PCIe H2D/D2H for expert weights and KV cache
"""

import argparse
import datetime
import itertools
import os
from typing import Any, List

import pandas as pd
import ray
import yaml
from tqdm import tqdm

from vidur.config.model_config import BaseModelConfig
from vidur.profiling.sparse.sparse_mlp_wrapper import SparseMlpWrapper
from vidur.profiling.sparse.mla_attention_wrapper import MLAAttentionWrapper
from vidur.profiling.sparse.dsa_attention_wrapper import DSAAttentionWrapper
from vidur.profiling.sparse.io_profiler import IOProfiler
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile


# ---- Sparse model presets ----

SPARSE_MODEL_CONFIGS = {
    "deepseek-ai/DeepSeek-V3": {
        "hidden_size": 7168,
        "num_routed_experts": 256,
        "num_experts_per_tok": 8,
        "expert_intermediate_size": 2048,
        "num_shared_experts": 1,
        "shared_expert_intermediate_size": 2048,
        # MLA parameters
        "num_heads": 128,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "has_mla": True,
        # DSA (Dynamic Sparse Attention) parameters
        "has_dsa": True,
        "dsa_selected_tokens": 2048,
        "dsa_sliding_window": 512,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "hidden_size": 4096,
        "num_routed_experts": 8,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 14336,
        "num_shared_experts": 0,
        "shared_expert_intermediate_size": 14336,
        # Standard GQA (no MLA)
        "has_mla": False,
    },
    "Qwen/Qwen3-30B-A3B": {
        "hidden_size": 4096,
        "num_routed_experts": 128,
        "num_experts_per_tok": 8,
        "expert_intermediate_size": 1408,
        "num_shared_experts": 1,
        "shared_expert_intermediate_size": 1408,
        "has_mla": False,
    },
    "deepseek-ai/Engram-27B": {
        "hidden_size": 2560,
        "num_routed_experts": 55,
        "num_experts_per_tok": 6,
        "expert_intermediate_size": 2560,
        "num_shared_experts": 2,
        "shared_expert_intermediate_size": 2560,
        "has_mla": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sparse Model Profiler (MoE + MLA + IO)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["deepseek-ai/DeepSeek-V3"],
        help="Models to profile (must be in SPARSE_MODEL_CONFIGS or registered in BaseModelConfig)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--profile_method",
        default="cuda_event",
        choices=[e.value for e in ProfileMethod],
        help="Method for measuring operation time (default: cuda_event)",
    )
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Run without Ray (single-GPU mode)",
    )
    parser.add_argument(
        "--skip_moe_mlp",
        action="store_true",
        help="Skip MoE MLP profiling",
    )
    parser.add_argument(
        "--skip_mla",
        action="store_true",
        help="Skip MLA attention profiling",
    )
    parser.add_argument(
        "--skip_io",
        action="store_true",
        help="Skip IO bandwidth profiling",
    )
    parser.add_argument(
        "--skip_dsa",
        action="store_true",
        help="Skip DSA (sparse attention) profiling",
    )
    parser.add_argument(
        "--dsa_seq_lens",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths for DSA profiling (default: logarithmic sweep up to max_tokens)",
    )
    parser.add_argument(
        "--dsa_selected_tokens",
        type=int,
        default=None,
        help="Override DSA top-k selected tokens (default: from model config)",
    )
    parser.add_argument(
        "--dsa_sliding_window",
        type=int,
        default=None,
        help="Override DSA sliding window size (default: from model config)",
    )
    parser.add_argument(
        "--io_sizes_mb",
        type=float,
        nargs="+",
        default=[0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        help="IO transfer sizes in MB for bandwidth sweep",
    )
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.output_dir = f"{args.output_dir}/sparse/{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_model_config(model_name: str) -> dict:
    """Get sparse model config from presets or BaseModelConfig registry."""
    if model_name in SPARSE_MODEL_CONFIGS:
        return SPARSE_MODEL_CONFIGS[model_name]

    # Try to load from BaseModelConfig registry
    try:
        mc = BaseModelConfig.create_from_name(model_name)
        config = {
            "hidden_size": mc.embedding_dim,
            "num_routed_experts": mc.num_routed_experts,
            "num_experts_per_tok": mc.num_experts_per_tok,
            "expert_intermediate_size": mc.expert_intermediate_size,
            "num_shared_experts": mc.num_shared_experts,
            "shared_expert_intermediate_size": mc.expert_intermediate_size,
            "has_mla": mc.attention_type == "MLA",
        }
        if config["has_mla"]:
            config.update({
                "num_heads": mc.num_q_heads,
                "kv_lora_rank": mc.kv_lora_rank,
                "q_lora_rank": mc.q_lora_rank,
                "qk_nope_head_dim": mc.qk_nope_head_dim,
                "qk_rope_head_dim": mc.qk_rope_head_dim,
                "v_head_dim": mc.v_head_dim,
            })
        return config
    except Exception as e:
        raise ValueError(
            f"Model {model_name} not found in SPARSE_MODEL_CONFIGS or BaseModelConfig registry: {e}"
        )


def profile_moe_mlp(
    args: argparse.Namespace,
    model_name: str,
    config: dict,
    num_tokens_list: List[int],
    pbar: Any,
) -> pd.DataFrame:
    """Profile MoE MLP block across token counts."""
    all_results = []

    if args.disable_ray:
        wrapper = SparseMlpWrapper(
            hidden_size=config["hidden_size"],
            num_routed_experts=config["num_routed_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            expert_intermediate_size=config["expert_intermediate_size"],
            num_shared_experts=config["num_shared_experts"],
            shared_expert_intermediate_size=config["shared_expert_intermediate_size"],
            profile_method=args.profile_method,
            output_dir=args.output_dir,
        )
        for num_tokens in num_tokens_list:
            result = wrapper.profile(num_tokens)
            all_results.append(result)
            pbar.update(1)
    else:
        wrapper_actor = ray.remote(num_cpus=1, num_gpus=1)(SparseMlpWrapper)
        wrappers = [
            wrapper_actor.remote(
                hidden_size=config["hidden_size"],
                num_routed_experts=config["num_routed_experts"],
                num_experts_per_tok=config["num_experts_per_tok"],
                expert_intermediate_size=config["expert_intermediate_size"],
                num_shared_experts=config["num_shared_experts"],
                shared_expert_intermediate_size=config["shared_expert_intermediate_size"],
                profile_method=args.profile_method,
                output_dir=args.output_dir,
            )
            for _ in range(args.num_gpus)
        ]

        promises = []
        for i, num_tokens in enumerate(num_tokens_list):
            worker_id = i % args.num_gpus
            promise = wrappers[worker_id].profile.remote(num_tokens)
            promises.append(promise)

            if len(promises) >= args.num_gpus:
                results = ray.get(promises)
                all_results.extend(results)
                promises = []
                pbar.update(len(results))

        if promises:
            results = ray.get(promises)
            all_results.extend(results)
            pbar.update(len(results))

    df = pd.DataFrame(all_results)
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    return df


def profile_mla_attention(
    args: argparse.Namespace,
    model_name: str,
    config: dict,
    num_tokens_list: List[int],
    pbar: Any,
) -> pd.DataFrame:
    """Profile MLA attention projections across token counts."""
    all_results = []

    if args.disable_ray:
        wrapper = MLAAttentionWrapper(
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            kv_lora_rank=config["kv_lora_rank"],
            q_lora_rank=config["q_lora_rank"],
            qk_nope_head_dim=config["qk_nope_head_dim"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
            v_head_dim=config["v_head_dim"],
            profile_method=args.profile_method,
            output_dir=args.output_dir,
        )
        for num_tokens in num_tokens_list:
            result = wrapper.profile(num_tokens)
            all_results.append(result)
            pbar.update(1)
    else:
        wrapper_actor = ray.remote(num_cpus=1, num_gpus=1)(MLAAttentionWrapper)
        wrappers = [
            wrapper_actor.remote(
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                kv_lora_rank=config["kv_lora_rank"],
                q_lora_rank=config["q_lora_rank"],
                qk_nope_head_dim=config["qk_nope_head_dim"],
                qk_rope_head_dim=config["qk_rope_head_dim"],
                v_head_dim=config["v_head_dim"],
                profile_method=args.profile_method,
                output_dir=args.output_dir,
            )
            for _ in range(args.num_gpus)
        ]

        promises = []
        for i, num_tokens in enumerate(num_tokens_list):
            worker_id = i % args.num_gpus
            promise = wrappers[worker_id].profile.remote(num_tokens)
            promises.append(promise)

            if len(promises) >= args.num_gpus:
                results = ray.get(promises)
                all_results.extend(results)
                promises = []
                pbar.update(len(results))

        if promises:
            results = ray.get(promises)
            all_results.extend(results)
            pbar.update(len(results))

    df = pd.DataFrame(all_results)
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    return df


def get_dsa_seq_lens(max_seq_len: int) -> List[int]:
    """Generate a logarithmic sweep of sequence lengths for DSA profiling.

    DSA cost scales with seq_len (indexer K cache size, top-k cost), so we
    sweep across a representative range from small to large contexts.
    """
    import math
    seq_lens = []
    # Powers of 2 from 1024 to max_seq_len
    power = 10  # 1024
    while 2**power <= max_seq_len:
        seq_lens.append(2**power)
        power += 1
    # Also add intermediate points for finer resolution
    intermediates = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    for s in intermediates:
        if s <= max_seq_len and s not in seq_lens:
            seq_lens.append(s)
    seq_lens.sort()
    return seq_lens


def profile_dsa_attention(
    args: argparse.Namespace,
    model_name: str,
    config: dict,
    num_tokens_list: List[int],
    seq_lens: List[int],
    pbar: Any,
) -> pd.DataFrame:
    """Profile DSA sparse attention across (num_tokens, seq_len) pairs.

    For DSA decode, we typically profile with num_tokens=1 (single decode
    token) across varying seq_lens, since seq_len controls the indexer
    cache size and top-k cost — the dominant variable.  We also sweep
    num_tokens for batch decode scenarios.
    """
    all_results = []

    dsa_selected = args.dsa_selected_tokens or config.get("dsa_selected_tokens", 2048)
    dsa_window = args.dsa_sliding_window or config.get("dsa_sliding_window", 512)

    if args.disable_ray:
        wrapper = DSAAttentionWrapper(
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            kv_lora_rank=config["kv_lora_rank"],
            q_lora_rank=config["q_lora_rank"],
            qk_nope_head_dim=config["qk_nope_head_dim"],
            qk_rope_head_dim=config["qk_rope_head_dim"],
            v_head_dim=config["v_head_dim"],
            dsa_selected_tokens=dsa_selected,
            dsa_sliding_window=dsa_window,
            max_seq_len=max(seq_lens),
            profile_method=args.profile_method,
            output_dir=args.output_dir,
        )
        for seq_len in seq_lens:
            for num_tokens in num_tokens_list:
                result = wrapper.profile(num_tokens, seq_len)
                all_results.append(result)
                pbar.update(1)
    else:
        wrapper_actor = ray.remote(num_cpus=1, num_gpus=1)(DSAAttentionWrapper)
        wrappers = [
            wrapper_actor.remote(
                hidden_size=config["hidden_size"],
                num_heads=config["num_heads"],
                kv_lora_rank=config["kv_lora_rank"],
                q_lora_rank=config["q_lora_rank"],
                qk_nope_head_dim=config["qk_nope_head_dim"],
                qk_rope_head_dim=config["qk_rope_head_dim"],
                v_head_dim=config["v_head_dim"],
                dsa_selected_tokens=dsa_selected,
                dsa_sliding_window=dsa_window,
                max_seq_len=max(seq_lens),
                profile_method=args.profile_method,
                output_dir=args.output_dir,
            )
            for _ in range(args.num_gpus)
        ]

        promises = []
        idx = 0
        for seq_len in seq_lens:
            for num_tokens in num_tokens_list:
                worker_id = idx % args.num_gpus
                promise = wrappers[worker_id].profile.remote(num_tokens, seq_len)
                promises.append(promise)
                idx += 1

                if len(promises) >= args.num_gpus:
                    results = ray.get(promises)
                    all_results.extend(results)
                    promises = []
                    pbar.update(len(results))

        if promises:
            results = ray.get(promises)
            all_results.extend(results)
            pbar.update(len(results))

    df = pd.DataFrame(all_results)
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )
    return df


def profile_io_bandwidth(
    args: argparse.Namespace,
    model_name: str,
    config: dict,
) -> pd.DataFrame:
    """Profile IO bandwidth sweep and model-specific transfer scenarios."""
    profiler = IOProfiler()

    # 1. General bandwidth sweep
    sizes_bytes = [int(s * 1024 * 1024) for s in args.io_sizes_mb]
    sweep_results = profiler.run_bandwidth_sweep(sizes_bytes)

    rows = []
    for transfer_type, results in sweep_results.items():
        for r in results:
            rows.append({
                "transfer_type": r.transfer_type,
                "size_bytes": r.size_bytes,
                "size_mb": r.size_bytes / (1024 * 1024),
                "latency_ms": r.latency_ms,
                "bandwidth_gb_per_s": r.bandwidth_gb_per_s,
                "model": model_name,
                "category": "bandwidth_sweep",
            })

    # 2. Model-specific expert weight loading
    hidden_size = config["hidden_size"]
    expert_intermediate = config["expert_intermediate_size"]
    num_experts = config["num_routed_experts"]

    for num_local_experts in [1, 2, 4, 8, min(16, num_experts)]:
        for source in ["hbm", "pcie"]:
            result = profiler.profile_expert_weight_load(
                hidden_size=hidden_size,
                expert_intermediate_size=expert_intermediate,
                num_local_experts=num_local_experts,
                bytes_per_param=2,
                source=source,
            )
            rows.append({
                "transfer_type": f"expert_weight_{source}",
                "size_bytes": result.size_bytes,
                "size_mb": result.size_bytes / (1024 * 1024),
                "latency_ms": result.latency_ms,
                "bandwidth_gb_per_s": result.bandwidth_gb_per_s,
                "model": model_name,
                "category": "expert_weight",
                "num_local_experts": num_local_experts,
            })

    # 3. KV cache transfer profiling
    # Compute KV bytes per token
    if config.get("has_mla"):
        kv_bytes_per_token = (config["kv_lora_rank"] + config["qk_rope_head_dim"]) * 2
    else:
        head_dim = hidden_size // config.get("num_heads", 128)
        num_kv_heads = config.get("num_kv_heads", config.get("num_heads", 128))
        kv_bytes_per_token = 2 * num_kv_heads * head_dim * 2

    for num_tokens in [128, 256, 512, 1024, 2048, 4096]:
        for batch_size in [1, 4, 16, 64]:
            for source in ["pcie", "hbm"]:
                result = profiler.profile_kv_cache_transfer(
                    kv_bytes_per_token=kv_bytes_per_token,
                    num_tokens=num_tokens,
                    batch_size=batch_size,
                    source=source,
                )
                rows.append({
                    "transfer_type": f"kv_cache_{source}",
                    "size_bytes": result.size_bytes,
                    "size_mb": result.size_bytes / (1024 * 1024),
                    "latency_ms": result.latency_ms,
                    "bandwidth_gb_per_s": result.bandwidth_gb_per_s,
                    "model": model_name,
                    "category": "kv_cache",
                    "num_tokens": num_tokens,
                    "batch_size": batch_size,
                })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))

    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Profiling: {model_name}")
        print(f"{'='*60}")

        config = get_model_config(model_name)
        model_dir = f"{args.output_dir}/{model_name.replace('/', '_')}"
        os.makedirs(model_dir, exist_ok=True)

        # Save model config
        yaml.dump(config, open(f"{model_dir}/model_config.yaml", "w"))

        # 1. MoE MLP profiling
        if not args.skip_moe_mlp:
            print(f"\n[1/4] Profiling MoE MLP ({len(num_tokens_to_profile)} token counts)...")
            if not args.disable_ray:
                ray.init(ignore_reinit_error=True)

            pbar = tqdm(total=len(num_tokens_to_profile), desc="MoE MLP")
            moe_df = profile_moe_mlp(args, model_name, config, num_tokens_to_profile, pbar)
            pbar.close()
            moe_df.to_csv(f"{model_dir}/sparse_mlp.csv", index=False)
            print(f"  Saved: {model_dir}/sparse_mlp.csv ({len(moe_df)} rows)")

        # 2. MLA attention profiling (only for models with MLA)
        if not args.skip_mla and config.get("has_mla"):
            print(f"\n[2/4] Profiling MLA attention ({len(num_tokens_to_profile)} token counts)...")
            if not args.disable_ray:
                ray.init(ignore_reinit_error=True)

            pbar = tqdm(total=len(num_tokens_to_profile), desc="MLA Attention")
            mla_df = profile_mla_attention(args, model_name, config, num_tokens_to_profile, pbar)
            pbar.close()
            mla_df.to_csv(f"{model_dir}/mla_attention.csv", index=False)
            print(f"  Saved: {model_dir}/mla_attention.csv ({len(mla_df)} rows)")
        elif not args.skip_mla:
            print(f"\n[2/4] Skipping MLA (model uses standard GQA attention)")

        # 3. DSA sparse attention profiling (only for models with DSA)
        if not args.skip_dsa and config.get("has_dsa"):
            dsa_seq_lens = args.dsa_seq_lens or get_dsa_seq_lens(args.max_tokens)
            # For DSA decode, profile with small batch sizes (1, 4, 16)
            dsa_num_tokens = [1, 4, 16]
            total_dsa_combos = len(dsa_seq_lens) * len(dsa_num_tokens)
            print(f"\n[3/4] Profiling DSA sparse attention "
                  f"({len(dsa_seq_lens)} seq_lens × {len(dsa_num_tokens)} batch sizes "
                  f"= {total_dsa_combos} combos)...")
            if not args.disable_ray:
                ray.init(ignore_reinit_error=True)

            pbar = tqdm(total=total_dsa_combos, desc="DSA Attention")
            dsa_df = profile_dsa_attention(
                args, model_name, config,
                dsa_num_tokens, dsa_seq_lens, pbar,
            )
            pbar.close()
            dsa_df.to_csv(f"{model_dir}/dsa_attention.csv", index=False)
            print(f"  Saved: {model_dir}/dsa_attention.csv ({len(dsa_df)} rows)")
        elif not args.skip_dsa:
            print(f"\n[3/4] Skipping DSA (model does not use sparse attention)")

        # 4. IO bandwidth profiling
        if not args.skip_io:
            print(f"\n[4/4] Profiling IO bandwidth...")
            io_df = profile_io_bandwidth(args, model_name, config)
            io_df.to_csv(f"{model_dir}/io_bandwidth.csv", index=False)
            print(f"  Saved: {model_dir}/io_bandwidth.csv ({len(io_df)} rows)")

        print(f"\nAll profiling data saved to: {model_dir}/")

    if not args.disable_ray and ray.is_initialized():
        ray.shutdown()

    print(f"\nDone. Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
