"""Gantt-style visualization of per-layer execution times.

Produces timeline charts showing three hardware streams (Compute/SM,
I/O/DMA, Communication/NCCL) for each layer, with time progressing
along the x-axis and layers stacked on the y-axis.  Overlapping
streams (e.g., DMA prefetch during compute) are visually apparent.
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
            # PDD KV transfer metrics
            "inter_gpu_kv_transfer_time_ms": execution_time.inter_gpu_kv_transfer_time_ms,
            "inter_gpu_kv_transfer_bytes": execution_time.inter_gpu_kv_transfer_bytes,
            "effective_inter_gpu_kv_transfer_ms": (
                execution_time.effective_inter_gpu_kv_transfer_time_ms
            ),
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
                    "inter_gpu_kv_transfer_time_ms": record.get(
                        "inter_gpu_kv_transfer_time_ms", 0.0
                    ),
                    "inter_gpu_kv_transfer_bytes": record.get(
                        "inter_gpu_kv_transfer_bytes", 0.0
                    ),
                    "effective_inter_gpu_kv_transfer_ms": record.get(
                        "effective_inter_gpu_kv_transfer_ms", 0.0
                    ),
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
        """Generate Gantt-style timeline plots with three stream rows per layer.

        Y-axis: Each layer gets three sub-rows (Compute/SM, I/O/DMA, Comm/NCCL).
        X-axis: Wall-clock time (ms) from batch start.

        This shows the actual temporal overlap between streams — e.g., DMA
        prefetching the next layer's KV cache while the current layer computes.
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

            self._plot_timeline_gantt(record, layers)

    def _plot_timeline_gantt(self, record: Dict, layers: List[Dict]) -> None:
        """Draw a timeline Gantt chart: one row per layer, overlapping stream bars.

        Each layer gets a single row.  Within the row, three streams are drawn
        as overlapping horizontal bars at different vertical offsets within
        the row, so temporal overlap is visually clear:

          - Compute (SM): full-height green bar (attention dark, MLP light)
          - I/O (DMA):    mid-height blue bar, drawn below compute
          - Comm (NCCL):  mid-height orange bar, drawn above compute

        X-axis is wall-clock time (ms) from batch start.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        bid = record["batch_id"]
        num_layers = len(layers)

        # One row per layer
        row_spacing = 1.0
        row_height_full = 0.7   # Compute bar height
        row_height_half = 0.3   # I/O and Comm bar height

        fig_height = max(6, num_layers * 0.4 + 2)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        # Compute absolute start time for each layer
        layer_abs_starts = []
        abs_time = 0.0
        for layer in layers:
            layer_abs_starts.append(abs_time)
            abs_time += layer["total_time"]

        y_ticks = []
        y_labels = []

        for i, layer in enumerate(layers):
            layer_abs = layer_abs_starts[i]
            y = (num_layers - 1 - i) * row_spacing  # layer 0 at top

            # --- Compute stream (SM) — full-height, centered ---
            compute_start = layer_abs + layer["compute_start"]
            compute_dur = layer["compute_end"] - layer["compute_start"]
            if compute_dur > 0:
                attn_dur = layer["attention_compute_time"]
                mlp_dur = layer["mlp_compute_time"] + layer.get("routing_time", 0)
                if attn_dur > 0:
                    ax.barh(y, attn_dur, left=compute_start,
                            height=row_height_full, color="#388E3C",
                            edgecolor="black", linewidth=0.3, zorder=2)
                if mlp_dur > 0:
                    ax.barh(y, mlp_dur, left=compute_start + attn_dur,
                            height=row_height_full, color="#81C784",
                            edgecolor="black", linewidth=0.3, zorder=2)

            # --- I/O stream (DMA) — half-height, below center ---
            io_start = layer_abs + layer["io_start"]
            io_dur = layer["io_end"] - layer["io_start"]
            if io_dur > 0:
                y_io = y - (row_height_full - row_height_half) / 2
                ax.barh(y_io, io_dur, left=io_start,
                        height=row_height_half, color="#2196F3",
                        edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3)

            # --- Communication stream (NCCL) — half-height, above center ---
            comm_start = layer_abs + layer["comm_start"]
            comm_dur = layer["comm_end"] - layer["comm_start"]
            if comm_dur > 0:
                y_comm = y + (row_height_full - row_height_half) / 2
                ax.barh(y_comm, comm_dur, left=comm_start,
                        height=row_height_half, color="#FF9800",
                        edgecolor="black", linewidth=0.3, alpha=0.85, zorder=3)

            y_ticks.append(y)
            label = f"L{layer['layer_index']}"
            if layer.get("is_moe_layer", False):
                label += " (MoE)"
            y_labels.append(label)

        # Light grid lines between layers
        for i in range(1, num_layers):
            sep_y = (num_layers - i) * row_spacing - row_spacing / 2
            ax.axhline(y=sep_y, color="gray", linewidth=0.2, alpha=0.4)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=max(5, min(8, 300 // num_layers)))
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.set_title(
            f"Batch {bid} - Per-Layer Timeline "
            f"(R{record['replica_id']} S{record['stage_id']}, "
            f"prefetch={'ON' if record['enable_prefetch'] else 'OFF'}, "
            f"savings={record['total_prefetch_savings_ms']:.2f}ms)",
            fontsize=11,
        )

        # Legend
        legend_patches = [
            mpatches.Patch(color="#388E3C", label="Attention (SM)"),
            mpatches.Patch(color="#81C784", label="MLP/MoE (SM)"),
            mpatches.Patch(color="#2196F3", label="I/O - KV DMA (PCIe)", alpha=0.85),
            mpatches.Patch(color="#FF9800", label="Comm - All-reduce (NVLink)", alpha=0.85),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
                  framealpha=0.9)

        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(
            self._output_dir, "plots", f"layer_gantt_batch_{bid}.png"
        )
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
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

        fig, ax = plt.subplots(figsize=(14, 6))
        x = list(range(num_layers))

        ax.bar(x, compute_avgs, label="Compute (SM)", color="#4CAF50", alpha=0.8)
        ax.bar(x, io_avgs, bottom=compute_avgs, label="I/O (DMA, effective)", color="#2196F3", alpha=0.8)
        bottoms = [c + io for c, io in zip(compute_avgs, io_avgs)]
        ax.bar(x, comm_avgs, bottom=bottoms, label="Communication (NCCL)", color="#FF9800", alpha=0.8)

        if any(s > 0 for s in savings_avgs):
            ax.plot(x, savings_avgs, "r--o", markersize=3, label="Prefetch overlap savings", alpha=0.7)

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Average Time (ms)")
        ax.set_title("Per-Layer Execution Time Breakdown (averaged across batches)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self._output_dir, "plots", "layer_timing_summary.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Layer timing summary saved to {plot_path}")
