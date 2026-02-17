"""Gantt-style visualization of per-layer execution times.

Produces horizontal bar charts showing compute, I/O, and communication
time breakdowns for each layer in a pipeline stage, with prefetch
overlap indicated.
"""

import json
import os
from typing import Dict, List, Optional

from vidur.entities.execution_time import ExecutionTime
from vidur.entities.layer_execution_time import LayerExecutionTime
from vidur.logger import init_logger

logger = init_logger(__name__)


class LayerTimingStore:
    """Stores per-layer timing data across batches for analysis and plotting."""

    def __init__(self, output_dir: str):
        self._output_dir = output_dir
        self._batch_layer_timings: List[Dict] = []

    def record_batch(
        self,
        batch_id: int,
        replica_id: int,
        stage_id: int,
        execution_time: ExecutionTime,
        batch_start_time: float,
    ) -> None:
        """Record per-layer timings for a batch execution."""
        layers = execution_time.layer_executions
        if not layers:
            return

        cumulative_time = 0.0  # ms from batch start

        for layer in layers:
            layer.start_time = batch_start_time + cumulative_time * 1e-3  # seconds
            cumulative_time += layer.total_time
            layer.end_time = batch_start_time + cumulative_time * 1e-3

        record = {
            "batch_id": batch_id,
            "replica_id": replica_id,
            "stage_id": stage_id,
            "enable_prefetch": execution_time.enable_kv_prefetch,
            "total_prefetch_savings_ms": execution_time.total_prefetch_savings_ms,
            "layers": [l.to_dict() for l in layers],
        }
        self._batch_layer_timings.append(record)

    def save_json(self) -> None:
        """Save all layer timing data to JSON."""
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, "layer_timings.json")
        with open(path, "w") as f:
            json.dump(self._batch_layer_timings, f, indent=2, default=str)
        logger.info(f"Layer timings saved to {path}")

    def save_csv(self) -> None:
        """Save per-layer timing data to CSV for analysis."""
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, "layer_timings.csv")

        rows = []
        for record in self._batch_layer_timings:
            for layer in record["layers"]:
                rows.append({
                    "batch_id": record["batch_id"],
                    "replica_id": record["replica_id"],
                    "stage_id": record["stage_id"],
                    "enable_prefetch": record["enable_prefetch"],
                    **layer,
                })

        if not rows:
            return

        import csv
        fieldnames = rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Layer timings CSV saved to {path}")

    def plot_gantt(self, batch_id: Optional[int] = None, max_batches: int = 5) -> None:
        """Generate Gantt-style plots showing per-layer time breakdown.

        Each layer is a horizontal bar split into:
        - Green: Compute time (attention + MLP)
        - Blue: I/O time (KV cache load + weight load)
        - Orange: Communication time (TP all-reduce + EP dispatch/combine)
        - Red hatching: Prefetch overlap savings
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not available, skipping Gantt plots")
            return

        os.makedirs(os.path.join(self._output_dir, "plots"), exist_ok=True)

        # Select batches to plot
        records = self._batch_layer_timings
        if batch_id is not None:
            records = [r for r in records if r["batch_id"] == batch_id]
        else:
            records = records[:max_batches]

        for record in records:
            bid = record["batch_id"]
            layers = record["layers"]
            if not layers:
                continue

            fig, ax = plt.subplots(figsize=(14, max(6, len(layers) * 0.4)))

            y_positions = list(range(len(layers)))
            bar_height = 0.6

            for i, layer in enumerate(layers):
                y = len(layers) - 1 - i  # Top to bottom
                offset = 0.0

                # Compute (green)
                compute = layer["compute_time"]
                if compute > 0:
                    ax.barh(y, compute, left=offset, height=bar_height,
                            color="#4CAF50", edgecolor="black", linewidth=0.5)
                    offset += compute

                # Effective I/O (blue) - after prefetch savings
                effective_io = layer["effective_io_time"]
                if effective_io > 0:
                    ax.barh(y, effective_io, left=offset, height=bar_height,
                            color="#2196F3", edgecolor="black", linewidth=0.5)
                    offset += effective_io

                # Communication (orange)
                comm = layer["comm_time"]
                if comm > 0:
                    ax.barh(y, comm, left=offset, height=bar_height,
                            color="#FF9800", edgecolor="black", linewidth=0.5)
                    offset += comm

                # Show prefetch savings as hatched region
                savings = layer.get("prefetch_overlap_savings", 0)
                if savings > 0:
                    ax.barh(y, savings, left=offset, height=bar_height * 0.3,
                            color="#F44336", alpha=0.5, edgecolor="red",
                            linewidth=0.5, hatch="//")

                # MoE indicator
                if layer.get("is_moe_layer", False):
                    ax.annotate("MoE", xy=(offset + 0.1, y),
                               fontsize=6, color="purple", fontweight="bold")

            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"Layer {l['layer_index']}" for l in reversed(layers)],
                              fontsize=8)
            ax.set_xlabel("Time (ms)")
            ax.set_title(
                f"Batch {bid} - Per-Layer Execution Breakdown "
                f"(R{record['replica_id']} S{record['stage_id']})"
            )

            # Legend
            legend_patches = [
                mpatches.Patch(color="#4CAF50", label="Compute"),
                mpatches.Patch(color="#2196F3", label="I/O (effective)"),
                mpatches.Patch(color="#FF9800", label="Communication"),
            ]
            if record["enable_prefetch"]:
                legend_patches.append(
                    mpatches.Patch(
                        facecolor="#F44336", alpha=0.5, edgecolor="red",
                        hatch="//", label=f"Prefetch savings ({record['total_prefetch_savings_ms']:.2f}ms)"
                    )
                )
            ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

            plt.tight_layout()
            plot_path = os.path.join(
                self._output_dir, "plots", f"layer_gantt_batch_{bid}.png"
            )
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.info(f"Layer Gantt plot saved to {plot_path}")

    def plot_summary(self) -> None:
        """Plot aggregate layer timing summary across all batches."""
        if not self._batch_layer_timings:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not available, skipping summary plot")
            return

        os.makedirs(os.path.join(self._output_dir, "plots"), exist_ok=True)

        # Aggregate across batches
        num_layers = len(self._batch_layer_timings[0]["layers"])
        compute_avgs = [0.0] * num_layers
        io_avgs = [0.0] * num_layers
        comm_avgs = [0.0] * num_layers
        savings_avgs = [0.0] * num_layers
        count = len(self._batch_layer_timings)

        for record in self._batch_layer_timings:
            for i, layer in enumerate(record["layers"]):
                if i < num_layers:
                    compute_avgs[i] += layer["compute_time"] / count
                    io_avgs[i] += layer["effective_io_time"] / count
                    comm_avgs[i] += layer["comm_time"] / count
                    savings_avgs[i] += layer.get("prefetch_overlap_savings", 0) / count

        fig, ax = plt.subplots(figsize=(12, 6))
        x = list(range(num_layers))

        ax.bar(x, compute_avgs, label="Compute", color="#4CAF50", alpha=0.8)
        ax.bar(x, io_avgs, bottom=compute_avgs, label="I/O (effective)", color="#2196F3", alpha=0.8)
        bottoms = [c + io for c, io in zip(compute_avgs, io_avgs)]
        ax.bar(x, comm_avgs, bottom=bottoms, label="Communication", color="#FF9800", alpha=0.8)

        if any(s > 0 for s in savings_avgs):
            ax.plot(x, savings_avgs, "r--o", markersize=3, label="Prefetch savings", alpha=0.7)

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Average Time (ms)")
        ax.set_title("Per-Layer Execution Time Breakdown (averaged across batches)")
        ax.legend()
        plt.tight_layout()

        plot_path = os.path.join(self._output_dir, "plots", "layer_timing_summary.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Layer timing summary saved to {plot_path}")
