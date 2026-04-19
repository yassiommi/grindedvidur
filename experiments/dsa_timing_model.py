"""DeepSeek-V3 DSA Decode timing model (BS=1).

Hybrid model:
  - Compute ops: profiled H100 data from
        data/profiling/compute/h100/deepseek_DeepSeek-V3/{attention,mlp}.csv
    indexed by num_tokens. For decode BS=1 ops use num_tokens=1; for
    kv_up_proj on 2560 fetched tokens use num_tokens=2560. Fallback to
    analytical (memory-bound) estimate if a key is missing or the
    profiled value is implausible (inflated Python loops, etc).
  - IO ops: purely analytical. HBM peak ~1384 GB/s (profiled) with a
    ~0.015 ms latency floor below 16 MB. PCIe Gen4 x16 ~51.5 GB/s with
    a ~0.02 ms floor. NVLink ~5 us per message.
  - MoE expert_gemm: analytical HBM read (84 MB / HBM_bw), overriding
    the profile's 2.38 ms value that is inflated by a Python per-expert
    loop in the profiler.

Two modes:
  - Mode 1 "offload": full KV cache and indexer K cache live on CPU.
    Per layer: read seq_len*512 bytes of indexer K via PCIe, then
    gather 2560 * 1152 bytes of MLA KV via PCIe.
  - Mode 2 "hbm": both live in GPU HBM. The same reads happen but at
    HBM bandwidth.

Pipeline per layer:

    Block A (Q compute) ─┐
                         ├─ parallel ─→ Block C (attention + output)
    Block B (KV index) ─┘
                         then → MoE block (sequential)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# DeepSeek-V3 config
# ─────────────────────────────────────────────────────────────────────
NUM_LAYERS = 61
HIDDEN_SIZE = 7168
NUM_HEADS = 128
KV_LORA_RANK = 512
Q_LORA_RANK = 1536
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128

Q_TOTAL_DIM = NUM_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)  # 24576
KV_UP_OUT_DIM = NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)       # 32768
O_PROJ_IN_DIM = NUM_HEADS * V_HEAD_DIM                             # 16384

# KV cache bytes per token per layer (FP16 MLA: kv_lora + rope)
MLA_KV_BYTES_PER_TOKEN = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 2  # 1152 bytes

# DSA selection
DSA_SELECTED_TOKENS = 2048
DSA_SLIDING_WINDOW = 512
DSA_ATTENDED = DSA_SELECTED_TOKENS + DSA_SLIDING_WINDOW  # 2560

# Indexer K: FP8 on kv_lora_rank dims per token
INDEXER_K_BYTES_PER_TOKEN = KV_LORA_RANK * 1  # 512 bytes (FP8)

# MoE config
NUM_ROUTED_EXPERTS = 256
NUM_EXPERTS_PER_TOK = 8
NUM_SHARED_EXPERTS = 1
EXPERT_INTERMEDIATE_SIZE = 2048
EXPERT_WEIGHT_BYTES = 3 * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE * 2  # ~84 MB
SHARED_EXPERT_WEIGHT_BYTES = EXPERT_WEIGHT_BYTES

EP = 8  # expert parallelism: 256 / 8 = 32 experts resident per GPU
TP = 1

# ─────────────────────────────────────────────────────────────────────
# H100 SXM analytical IO parameters (profiled-peak-calibrated)
# ─────────────────────────────────────────────────────────────────────
# From data/profiling/compute/h100/deepseek_DeepSeek-V3/io.csv:
#   hbm_read at 512 MB → 1383.6 GB/s        → peak
#   hbm_read floor at <1 MB → ~0.009-0.015 ms
#   pcie_h2d at 256+ MB → ~51.5 GB/s        → Gen4 x16 peak
#   pcie_h2d floor at 0.5 MB → ~0.018 ms
H100_HBM_PEAK_GBS = 1384.0
H100_HBM_FLOOR_MS = 0.015
H100_PCIE_PEAK_GBS = 51.5
H100_PCIE_FLOOR_MS = 0.020
NVLINK_LATENCY_US = 5.0   # per message
H100_HBM_EFF = 0.80       # effective fraction of peak used for analytical GEMMs

# ── H100 FP16 Tensor Core peak (for compute-bound analytical estimates) ──
H100_FP16_TFLOPS = 989.5
H100_MFU = 0.50  # conservative model-FLOPs utilisation for GEMM

LOCAL_EXPERTS_PER_GPU = NUM_ROUTED_EXPERTS // EP  # 32


# Analytical MoE expert GEMM: 84 MB HBM read dominates at BS=1
def analytical_moe_expert_gemm_ms(experts_on_gpu: float = 1.0) -> float:
    bytes_read = int(experts_on_gpu * EXPERT_WEIGHT_BYTES)
    return hbm_read_ms(bytes_read)


def analytical_moe_expert_gemm_bs_ms(batch_size: int) -> float:
    """Analytical expert GEMM for a batch of tokens (grouped-GEMM model).

    With EP=8 and 8 experts per token, each GPU executes batch_size
    expert activations. The number of *unique* experts touched out of
    the 32 resident on this GPU follows:
        E[unique] = 32 * (1 - (31/32)^batch_size)
    Weight read = unique × 84 MB (each weight read once, grouped-GEMM).
    Compute is negligible at moderate BS (AI ≈ BS, transition ≈ BS 715).
    """
    unique = LOCAL_EXPERTS_PER_GPU * (1 - (1 - 1 / LOCAL_EXPERTS_PER_GPU) ** batch_size)
    bytes_read = int(unique * EXPERT_WEIGHT_BYTES)
    # Also credit compute (tiny at small BS, starts mattering at very large BS)
    flops = batch_size * 3 * 2 * HIDDEN_SIZE * EXPERT_INTERMEDIATE_SIZE
    compute_ms = (flops / 1e9) / (H100_FP16_TFLOPS * 1024 * H100_MFU) * 1e3
    mem_ms = hbm_read_ms(bytes_read)
    return max(compute_ms, mem_ms)


def analytical_gemm_ms(m: int, k: int, n: int) -> float:
    """Analytical GEMM time: max(compute, memory).

    At small m (BS=1) → memory-bound (read weight).
    At large m → compute-bound (FLOPs dominate).
    """
    flops = 2.0 * m * k * n
    weight_bytes = k * n * 2
    act_bytes = (m * k + m * n) * 2
    compute_ms = (flops / 1e9) / (H100_FP16_TFLOPS * 1024 * H100_MFU) * 1e3
    mem_ms = hbm_read_ms(weight_bytes + act_bytes)
    return max(compute_ms, mem_ms)


def hbm_read_ms(size_bytes: int) -> float:
    """Analytical HBM read time.

    t = max(bytes / peak_bw, floor).
    """
    bw_bytes_per_s = H100_HBM_PEAK_GBS * (1024 ** 3)
    bw_ms = (size_bytes / bw_bytes_per_s) * 1e3
    return max(bw_ms, H100_HBM_FLOOR_MS)


def pcie_read_ms(size_bytes: int) -> float:
    """Analytical PCIe Gen4 x16 D2H/H2D transfer time."""
    bw_bytes_per_s = H100_PCIE_PEAK_GBS * (1024 ** 3)
    bw_ms = (size_bytes / bw_bytes_per_s) * 1e3
    return max(bw_ms, H100_PCIE_FLOOR_MS)


def nvlink_ms(num_messages: int) -> float:
    """NVLink latency-bound cost at small message sizes."""
    return num_messages * NVLINK_LATENCY_US / 1000.0


def io_read_ms(size_bytes: int, mode: str) -> float:
    """IO read on the selected mode: 'hbm' or 'offload' (PCIe)."""
    if mode == "offload":
        return pcie_read_ms(size_bytes)
    return hbm_read_ms(size_bytes)


# ─────────────────────────────────────────────────────────────────────
# Profiled compute lookup
# ─────────────────────────────────────────────────────────────────────
DEFAULT_PROFILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "profiling", "compute", "h100", "deepseek_DeepSeek-V3",
)


@dataclass
class ProfileTables:
    """Profiled-compute lookup tables indexed by num_tokens."""
    attn: Dict[int, Dict[str, float]] = field(default_factory=dict)
    mlp: Dict[int, Dict[str, float]] = field(default_factory=dict)
    attn_max: int = 0
    mlp_max: int = 0

    def lookup_attn(self, num_tokens: int, key: str) -> Optional[float]:
        return _nearest_lookup(self.attn, num_tokens, key)

    def lookup_mlp(self, num_tokens: int, key: str) -> Optional[float]:
        return _nearest_lookup(self.mlp, num_tokens, key)


def _nearest_lookup(table: Dict[int, Dict[str, float]], nt: int, key: str) -> Optional[float]:
    if not table:
        return None
    if nt in table and key in table[nt]:
        return table[nt][key]
    # fallback: nearest num_tokens that has this key
    candidates = [k for k, row in table.items() if key in row]
    if not candidates:
        return None
    best = min(candidates, key=lambda k: abs(k - nt))
    return table[best][key]


@lru_cache(maxsize=1)
def load_profiles(profile_dir: str = DEFAULT_PROFILE_DIR) -> ProfileTables:
    tables = ProfileTables()
    attn_path = os.path.join(profile_dir, "attention.csv")
    mlp_path = os.path.join(profile_dir, "mlp.csv")

    if os.path.exists(attn_path):
        df = pd.read_csv(attn_path)
        for _, row in df.iterrows():
            nt = int(row["num_tokens"])
            entry = tables.attn.setdefault(nt, {})
            for col in df.columns:
                if col.startswith("time_stats.") and col.endswith(".median"):
                    short = col[len("time_stats."):-len(".median")]
                    entry[short] = float(row[col])
        tables.attn_max = max(tables.attn.keys()) if tables.attn else 0

    if os.path.exists(mlp_path):
        df = pd.read_csv(mlp_path)
        for _, row in df.iterrows():
            nt = int(row["num_tokens"])
            entry = tables.mlp.setdefault(nt, {})
            for col in df.columns:
                if col.startswith("time_stats.") and col.endswith(".median"):
                    short = col[len("time_stats."):-len(".median")]
                    entry[short] = float(row[col])
        tables.mlp_max = max(tables.mlp.keys()) if tables.mlp else 0

    return tables


def profiled_or_analytical(
    tables: ProfileTables,
    table_name: str,
    num_tokens: int,
    key: str,
    analytical_fn,
    inflation_guard_ms: Optional[float] = None,
    override_always: bool = False,
) -> tuple[float, str]:
    """Return (time_ms, source) where source is 'profiled' or 'analytical'.

    inflation_guard_ms: if the profiled value exceeds this multiplier of
        the analytical estimate (and analytical is non-zero), prefer
        analytical. Used to guard against Python-loop inflation.
    override_always: always return the analytical time (used for
        moe_expert_gemm which is known-inflated in the profile).
    """
    analytical_ms = analytical_fn()

    if override_always:
        return analytical_ms, "analytical"

    lookup = tables.lookup_attn if table_name == "attn" else tables.lookup_mlp
    prof = lookup(num_tokens, key)

    if prof is None or prof <= 0:
        return analytical_ms, "analytical"

    if inflation_guard_ms is not None and analytical_ms > 0:
        if prof > inflation_guard_ms * analytical_ms:
            return analytical_ms, "analytical"

    return prof, "profiled"
