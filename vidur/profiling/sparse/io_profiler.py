"""IO profiler: measures actual PCIe and HBM bandwidth for sparse model data transfers.

Profiles real transfer latencies for:
  1. Expert weight loading from HBM (GPU memory -> SM, simulating weight fetch)
  2. Expert weight loading from CPU (host -> GPU via PCIe, simulating CPU-offloaded experts)
  3. KV cache loading from CPU (host -> GPU via PCIe, simulating KV prefetch)
  4. KV cache loading from GPU (HBM, simulating on-device KV access)

Each test measures actual bandwidth by transferring tensors of known size
and timing with CUDA events.  This gives empirically-grounded IO costs
rather than relying on peak theoretical bandwidth numbers.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import numpy as np


WARMUP_STEPS = 5
ACTIVE_STEPS = 20


@dataclass
class IOProfileResult:
    """Result of a single IO profile measurement."""
    transfer_type: str       # "hbm_read", "pcie_h2d", "pcie_d2h"
    size_bytes: int
    latency_ms: float        # median across active steps
    bandwidth_gb_per_s: float  # effective bandwidth


class IOProfiler:
    """Profiles actual IO bandwidth for expert weight and KV cache transfers."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)

    def _time_transfer(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        num_warmup: int = WARMUP_STEPS,
        num_active: int = ACTIVE_STEPS,
    ) -> List[float]:
        """Time a tensor copy operation using CUDA events.

        Returns list of elapsed times in milliseconds.
        """
        # Warmup
        for _ in range(num_warmup):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()

        # Active measurement
        times = []
        for _ in range(num_active):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dst.copy_(src, non_blocking=True)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        return times

    def profile_hbm_read(self, size_bytes: int) -> IOProfileResult:
        """Profile GPU HBM read bandwidth (GPU -> GPU copy, same device).

        Simulates expert weight loading from HBM by copying a tensor
        within GPU memory.  This measures the effective HBM bandwidth
        for large sequential reads.
        """
        num_elements = size_bytes // 2  # FP16
        src = torch.randn(num_elements, device=self.device, dtype=torch.float16)
        dst = torch.empty_like(src)

        times = self._time_transfer(src, dst)
        median_ms = float(np.median(times))
        bw = (size_bytes / (median_ms * 1e-3)) / (1024**3) if median_ms > 0 else 0

        return IOProfileResult(
            transfer_type="hbm_read",
            size_bytes=size_bytes,
            latency_ms=median_ms,
            bandwidth_gb_per_s=bw,
        )

    def profile_pcie_h2d(self, size_bytes: int) -> IOProfileResult:
        """Profile PCIe Host-to-Device bandwidth.

        Simulates CPU-offloaded expert weight or KV cache transfer
        from pinned host memory to GPU.
        """
        num_elements = size_bytes // 2
        # Use pinned memory for realistic PCIe throughput
        src_cpu = torch.randn(num_elements, dtype=torch.float16).pin_memory()
        dst_gpu = torch.empty(num_elements, device=self.device, dtype=torch.float16)

        times = self._time_transfer(src_cpu, dst_gpu)
        median_ms = float(np.median(times))
        bw = (size_bytes / (median_ms * 1e-3)) / (1024**3) if median_ms > 0 else 0

        return IOProfileResult(
            transfer_type="pcie_h2d",
            size_bytes=size_bytes,
            latency_ms=median_ms,
            bandwidth_gb_per_s=bw,
        )

    def profile_pcie_d2h(self, size_bytes: int) -> IOProfileResult:
        """Profile PCIe Device-to-Host bandwidth.

        Simulates KV cache offloading from GPU to host memory.
        """
        num_elements = size_bytes // 2
        src_gpu = torch.randn(num_elements, device=self.device, dtype=torch.float16)
        dst_cpu = torch.empty(num_elements, dtype=torch.float16).pin_memory()

        times = self._time_transfer(src_gpu, dst_cpu)
        median_ms = float(np.median(times))
        bw = (size_bytes / (median_ms * 1e-3)) / (1024**3) if median_ms > 0 else 0

        return IOProfileResult(
            transfer_type="pcie_d2h",
            size_bytes=size_bytes,
            latency_ms=median_ms,
            bandwidth_gb_per_s=bw,
        )

    def profile_expert_weight_load(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        num_local_experts: int,
        bytes_per_param: int = 2,
        source: str = "hbm",
    ) -> IOProfileResult:
        """Profile loading expert weights for a single MoE layer.

        Args:
            hidden_size: Model hidden dimension
            expert_intermediate_size: Expert FFN intermediate dimension
            num_local_experts: Number of experts on this GPU
            bytes_per_param: Bytes per parameter (2 for FP16, 1 for INT8)
            source: "hbm" for GPU-resident, "pcie" for CPU-offloaded

        Returns:
            IOProfileResult with measured latency and bandwidth
        """
        # 3 matrices per expert: gate_proj, up_proj, down_proj
        expert_bytes = 3 * hidden_size * expert_intermediate_size * bytes_per_param
        total_bytes = expert_bytes * num_local_experts

        if source == "hbm":
            return self.profile_hbm_read(total_bytes)
        else:
            return self.profile_pcie_h2d(total_bytes)

    def profile_kv_cache_transfer(
        self,
        kv_bytes_per_token: int,
        num_tokens: int,
        batch_size: int,
        source: str = "pcie",
    ) -> IOProfileResult:
        """Profile KV cache transfer for a batch.

        Args:
            kv_bytes_per_token: KV bytes per token per layer
            num_tokens: Average sequence length (KV cache tokens)
            batch_size: Decode batch size
            source: "pcie" for host->device, "hbm" for on-device

        Returns:
            IOProfileResult with measured latency and bandwidth
        """
        total_bytes = kv_bytes_per_token * num_tokens * batch_size

        if source == "pcie":
            return self.profile_pcie_h2d(total_bytes)
        else:
            return self.profile_hbm_read(total_bytes)

    def run_bandwidth_sweep(
        self,
        sizes_bytes: List[int],
    ) -> Dict[str, List[IOProfileResult]]:
        """Run a full bandwidth sweep across transfer types and sizes.

        Returns dict mapping transfer_type -> list of IOProfileResult
        sorted by size_bytes.
        """
        results = {
            "hbm_read": [],
            "pcie_h2d": [],
            "pcie_d2h": [],
        }

        for size_bytes in sizes_bytes:
            results["hbm_read"].append(self.profile_hbm_read(size_bytes))
            results["pcie_h2d"].append(self.profile_pcie_h2d(size_bytes))
            results["pcie_d2h"].append(self.profile_pcie_d2h(size_bytes))

        return results
