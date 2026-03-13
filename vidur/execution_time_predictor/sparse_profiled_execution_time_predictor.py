"""Sparse-profiled execution time predictor.

Uses empirically measured profiling data from the sparse profiler
(vidur.profiling.sparse) to predict MoE and MLA execution times,
replacing the analytical FLOPs-based estimates in MoEExecutionTimePredictor.

When sparse profiling CSVs are available (sparse_mlp.csv, mla_attention.csv,
io_bandwidth.csv), this predictor trains sklearn models on the measured data.
When profiling data is missing, it falls back to analytical calculation.

Profiling data is expected at:
    {sparse_profiling_dir}/sparse_mlp.csv
    {sparse_profiling_dir}/mla_attention.csv
    {sparse_profiling_dir}/io_bandwidth.csv

Generate profiling data with:
    python -m vidur.profiling.sparse.main --model deepseek-ai/DeepSeek-V3
Or use the synthetic generator:
    python scripts/generate_sparse_profiling_data.py
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from vidur.config import (
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.config.config import SparseProfiledExecutionTimePredictorConfig
from vidur.entities import Batch
from vidur.execution_time_predictor.moe_execution_time_predictor import (
    MoEExecutionTimePredictor,
)
from vidur.logger import init_logger

logger = init_logger(__name__)


class SparseProfiledExecutionTimePredictor(MoEExecutionTimePredictor):
    """Execution time predictor that uses sparse profiling data.

    Overrides MoE and MLA timing methods with empirically-measured values
    from the sparse profiler CSVs, while keeping the sklearn-based dense
    model predictions for standard attention and MLP operations.
    """

    def __init__(
        self,
        predictor_config: SparseProfiledExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        # Initialize parent (sklearn + MoE analytical)
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

        # Resolve sparse profiling directory
        sparse_dir = predictor_config.sparse_profiling_dir
        if sparse_dir:
            sparse_dir = (
                sparse_dir
                .replace("{DEVICE}", replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
            )

        self._sparse_dir = sparse_dir
        self._sparse_mlp_predictions = None
        self._sparse_mla_predictions = None
        self._profiled_hbm_bw_gbs = None
        self._profiled_pcie_bw_gbs = None

        if sparse_dir and os.path.isdir(sparse_dir):
            self._load_sparse_profiling_data(sparse_dir)
        else:
            logger.info(
                f"Sparse profiling dir not found ({sparse_dir}), "
                f"using analytical MoE/MLA timing"
            )

    def _load_sparse_profiling_data(self, sparse_dir: str):
        """Load and train models from sparse profiling CSVs."""
        mlp_path = os.path.join(sparse_dir, "sparse_mlp.csv")
        mla_path = os.path.join(sparse_dir, "mla_attention.csv")
        io_path = os.path.join(sparse_dir, "io_bandwidth.csv")

        if os.path.exists(mlp_path):
            self._load_sparse_mlp(mlp_path)
        else:
            logger.info(f"sparse_mlp.csv not found, using analytical MoE timing")

        if os.path.exists(mla_path):
            self._load_mla_attention(mla_path)
        else:
            logger.info(f"mla_attention.csv not found, using analytical MLA timing")

        if os.path.exists(io_path):
            self._load_io_bandwidth(io_path)
        else:
            logger.info(f"io_bandwidth.csv not found, using analytical IO bandwidth")

    def _load_sparse_mlp(self, path: str):
        """Load sparse MLP profiling data and build prediction lookup."""
        df = pd.read_csv(path)
        logger.info(f"Loaded sparse MLP profiling: {len(df)} rows from {path}")

        # Build per-num_tokens lookup for MoE operation timings (in ms)
        # Sum the relevant operations into routing and expert compute
        predictions = {}

        for _, row in df.iterrows():
            nt = int(row["num_tokens"])

            # Routing time: gate + topk + softmax
            routing_ms = (
                row.get("time_stats.moe_router_gate.median", 0)
                + row.get("time_stats.moe_router_topk.median", 0)
                + row.get("time_stats.moe_router_softmax.median", 0)
            )

            # Expert compute time: dispatch + expert GEMM + combine + shared
            expert_compute_ms = (
                row.get("time_stats.moe_expert_dispatch.median", 0)
                + row.get("time_stats.moe_expert_gemm.median", 0)
                + row.get("time_stats.moe_expert_combine.median", 0)
                + row.get("time_stats.moe_shared_expert.median", 0)
            )

            predictions[nt] = {
                "routing_ms": routing_ms,
                "expert_compute_ms": expert_compute_ms,
            }

        self._sparse_mlp_predictions = predictions
        self._sparse_mlp_max_tokens = max(predictions.keys()) if predictions else 0

        logger.info(
            f"  MoE profiled predictions: {len(predictions)} token counts, "
            f"max={self._sparse_mlp_max_tokens}"
        )

    def _load_mla_attention(self, path: str):
        """Load MLA attention profiling data and build prediction lookup."""
        df = pd.read_csv(path)
        logger.info(f"Loaded MLA profiling: {len(df)} rows from {path}")

        predictions = {}

        for _, row in df.iterrows():
            nt = int(row["num_tokens"])

            # Pre-projection: Q compress + Q decompress + KV compress + KV decompress
            pre_proj_ms = (
                row.get("time_stats.mla_q_down_proj.median", 0)
                + row.get("time_stats.mla_q_up_proj.median", 0)
                + row.get("time_stats.mla_kv_down_proj.median", 0)
                + row.get("time_stats.mla_kv_up_proj.median", 0)
            )

            # Post-projection: output projection
            post_proj_ms = row.get("time_stats.mla_o_proj.median", 0)

            # RoPE
            rope_ms = row.get("time_stats.mla_rope.median", 0)

            predictions[nt] = {
                "pre_proj_ms": pre_proj_ms,
                "post_proj_ms": post_proj_ms,
                "rope_ms": rope_ms,
            }

        self._sparse_mla_predictions = predictions
        self._sparse_mla_max_tokens = max(predictions.keys()) if predictions else 0

        logger.info(
            f"  MLA profiled predictions: {len(predictions)} token counts, "
            f"max={self._sparse_mla_max_tokens}"
        )

    def _load_io_bandwidth(self, path: str):
        """Load IO bandwidth profiling data to get empirical bandwidth."""
        df = pd.read_csv(path)
        logger.info(f"Loaded IO bandwidth profiling: {len(df)} rows from {path}")

        # Extract median bandwidth for large transfers (most representative)
        sweep = df[df["category"] == "bandwidth_sweep"]

        if len(sweep) > 0:
            # Use only large transfers (>= 16 MB) for stable bandwidth estimate
            large = sweep[sweep["size_mb"] >= 16]
            if len(large) == 0:
                large = sweep

            hbm_rows = large[large["transfer_type"] == "hbm_read"]
            pcie_rows = large[large["transfer_type"] == "pcie_h2d"]

            if len(hbm_rows) > 0:
                self._profiled_hbm_bw_gbs = hbm_rows["bandwidth_gb_per_s"].median()
                logger.info(f"  Profiled HBM bandwidth: {self._profiled_hbm_bw_gbs:.1f} GB/s")

            if len(pcie_rows) > 0:
                self._profiled_pcie_bw_gbs = pcie_rows["bandwidth_gb_per_s"].median()
                logger.info(f"  Profiled PCIe bandwidth: {self._profiled_pcie_bw_gbs:.1f} GB/s")

        # Update internal bandwidth values if profiled data is available
        if self._profiled_hbm_bw_gbs is not None:
            self._mem_bw_bytes_per_s = self._profiled_hbm_bw_gbs * (1024 ** 3)
            if self._is_moe:
                # Recompute weight load times with profiled bandwidth
                total_expert_bytes = self._expert_params_bytes * self._local_experts
                self._weight_load_hbm_ms = (
                    total_expert_bytes / self._mem_bw_bytes_per_s
                ) * 1e3

        if self._profiled_pcie_bw_gbs is not None:
            self._pcie_bw_bytes_per_s = self._profiled_pcie_bw_gbs * (1024 ** 3)
            if self._is_moe:
                total_expert_bytes = self._expert_params_bytes * self._local_experts
                self._weight_load_pcie_ms = (
                    total_expert_bytes / self._pcie_bw_bytes_per_s
                ) * 1e3

    def _interpolate_sparse_prediction(
        self,
        predictions: Dict[int, dict],
        max_tokens: int,
        num_tokens: int,
        key: str,
    ) -> float:
        """Interpolate profiled prediction for a given token count."""
        if num_tokens in predictions:
            return predictions[num_tokens][key]

        # Find surrounding token counts
        token_counts = sorted(predictions.keys())

        if num_tokens <= token_counts[0]:
            # Extrapolate below: scale linearly from smallest
            ratio = num_tokens / token_counts[0]
            return predictions[token_counts[0]][key] * ratio

        if num_tokens >= token_counts[-1]:
            # Extrapolate above: scale linearly from largest
            ratio = num_tokens / token_counts[-1]
            return predictions[token_counts[-1]][key] * ratio

        # Linear interpolation between surrounding points
        lo = max(t for t in token_counts if t <= num_tokens)
        hi = min(t for t in token_counts if t >= num_tokens)
        if lo == hi:
            return predictions[lo][key]

        frac = (num_tokens - lo) / (hi - lo)
        return predictions[lo][key] * (1 - frac) + predictions[hi][key] * frac

    # ── MoE method overrides ────────────────────────────────────────

    def _get_moe_routing_time(self, batch: Batch) -> float:
        if not self._is_moe:
            return 0.0

        if self._sparse_mlp_predictions is None:
            return super()._get_moe_routing_time(batch)

        num_tokens = sum(batch.num_tokens)
        return self._interpolate_sparse_prediction(
            self._sparse_mlp_predictions,
            self._sparse_mlp_max_tokens,
            num_tokens,
            "routing_ms",
        )

    def _get_moe_expert_compute_time(self, batch: Batch) -> float:
        if not self._is_moe:
            return 0.0

        if self._sparse_mlp_predictions is None:
            return super()._get_moe_expert_compute_time(batch)

        num_tokens = sum(batch.num_tokens)
        return self._interpolate_sparse_prediction(
            self._sparse_mlp_predictions,
            self._sparse_mlp_max_tokens,
            num_tokens,
            "expert_compute_ms",
        )

    # ── MLA projection overrides ────────────────────────────────────

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        """Use profiled MLA projections when available."""
        if self._sparse_mla_predictions is None:
            return super()._get_attention_layer_pre_proj_execution_time(batch)

        num_tokens = batch._total_num_tokens_rounded
        return self._interpolate_sparse_prediction(
            self._sparse_mla_predictions,
            self._sparse_mla_max_tokens,
            num_tokens,
            "pre_proj_ms",
        )

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        """Use profiled MLA output projection when available."""
        if self._sparse_mla_predictions is None:
            return super()._get_attention_layer_post_proj_execution_time(batch)

        num_tokens = batch._total_num_tokens_rounded
        return self._interpolate_sparse_prediction(
            self._sparse_mla_predictions,
            self._sparse_mla_max_tokens,
            num_tokens,
            "post_proj_ms",
        )

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        """Use profiled MLA RoPE when available."""
        if self._sparse_mla_predictions is None:
            return super()._get_attention_rope_execution_time(batch)

        num_tokens = batch._total_num_tokens_rounded
        return self._interpolate_sparse_prediction(
            self._sparse_mla_predictions,
            self._sparse_mla_max_tokens,
            num_tokens,
            "rope_ms",
        )

    # ── sklearn interface (required by parent) ──────────────────────

    def _get_grid_search_params(self):
        return {
            "n_estimators": self._config.num_estimators,
            "max_depth": self._config.max_depth,
            "min_samples_split": self._config.min_samples_split,
        }

    def _get_estimator(self):
        return RandomForestRegressor()
