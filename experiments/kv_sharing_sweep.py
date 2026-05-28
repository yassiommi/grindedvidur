"""KV-sharing config sweep — mirrors exec_calculator.html's model.

Compares 5 KV residence/sharing configurations to surface where SSD
hits a wall and where HBM + sharing becomes the only viable path.

Configs:
  hbm_full        — KV in HBM, no sharing (each GPU duplicates full KV)
  hbm_shard       — KV in HBM, sharded across PP stage (1/G local + NVLink fetch)
  ssd_full        — KV on SSD, no sharing, per-token IOPS-counted
  ssd_shard       — KV on SSD, sharded, per-token IOPS-counted
  ssd_shard_merge — KV on SSD, sharded, IO units merged (BW-bound)

Outputs to reports/figures/kv_sharing/.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GB = 1024 ** 3


# ---------------------- hardware + model constants ----------------------

@dataclass
class HW:
    SSD_BW: float = 28.0          # GB/s
    SSD_IOPS: float = 6_000_000.0  # IOPS cap
    HBM_BW: float = 1384.0         # GB/s
    NV_BW: float = 450.0           # GB/s per GPU
    HBM_FLOOR: float = 0.015       # ms
    SSD_FLOOR: float = 0.050       # ms
    IOPS_ON: bool = True

    # Parallelism
    PP: int = 2
    TP: int = 2
    EP: int = 8

    # Model architecture
    L: int = 61
    FP: int = 4
    IDXB: int = 128               # indexer-K bytes/token/layer (FP8)
    KVB: int = 576                # KV bytes/token/layer (FP8 MLA latent)
    A: int = 2048                 # top-k attended tokens
    U: int = 512                  # I/O unit size (B)

    # Per-layer compute (ms, FP8 + MLA absorption)
    c_sh_flat: float = 0.281      # shardable flat (divided by TP)
    c_sh_perbs: float = 0.041
    c_un_flat: float = 0.080      # unsharded flat (EP comms)
    c_un_perbs: float = 0.022     # unsharded per-BS (expert GEMM)
    t_topk: float = 0.010         # F-only indexer compute


def hbm_ms(bytes_, hw: HW):
    return max(bytes_ / (hw.HBM_BW * GB) * 1000, hw.HBM_FLOOR)


def ssd_ms(bytes_, iops_count, hw: HW):
    t_bw = bytes_ / (hw.SSD_BW * GB) * 1000
    t_iops = (iops_count / hw.SSD_IOPS) * 1000 if hw.IOPS_ON else 0
    return max(t_bw, t_iops, hw.SSD_FLOOR)


# ---------------------- per-config evaluation ----------------------

CONFIGS = {
    # (kvres, kvshard, kvmerge)
    "hbm_full":         ("hbm", False, False),
    "hbm_shard":        ("hbm", True,  False),
    "ssd_full":         ("ssd", False, False),
    "ssd_shard":        ("ssd", True,  False),
    "ssd_shard_merge":  ("ssd", True,  True),
}

CONFIG_LABELS = {
    "hbm_full":        "HBM, no sharing",
    "hbm_shard":       "HBM, sharded (NVLink fetch)",
    "ssd_full":        "SSD, no sharing",
    "ssd_shard":       "SSD, sharded (IOPS-counted)",
    "ssd_shard_merge": "SSD, sharded + merged (BW-bound)",
}

CONFIG_COLORS = {
    "hbm_full":        "#3FAA5B",  # green
    "hbm_shard":       "#2EBFD9",  # cyan
    "ssd_full":        "#C84B4B",  # red
    "ssd_shard":       "#F0C03A",  # yellow
    "ssd_shard_merge": "#A06CD5",  # purple
}

CONFIG_STYLES = {
    # (linestyle, linewidth, marker)
    "hbm_full":        ("--", 2.2, "s"),
    "hbm_shard":       ("--", 2.2, "D"),
    "ssd_full":        ("-",  2.4, "o"),
    "ssd_shard":       ("-",  2.0, "o"),
    "ssd_shard_merge": ("-",  2.0, "^"),
}


def evaluate(sl, bsc, cfg, hw: HW, scheme="ic"):
    """Returns dict matching exec_calculator's evaluate()."""
    kvres, kv_shard, kv_merge = CONFIGS[cfg]

    dp_attn = hw.EP // hw.TP
    bs = max(1, bsc // dp_attn)

    # F/S layer split
    if scheme == "dsa":
        nF, nS = hw.L, 0
    else:
        nF = sum(1 for i in range(hw.L) if i % hw.FP == 0)
        nS = hw.L - nF

    # idx and KV totals (per-layer, per-token, per the DP_attn slice)
    idx_bytes = bs * sl * hw.IDXB
    idx_io = hbm_ms(idx_bytes, hw)

    kv_bytes = bs * hw.A * hw.KVB
    kv_iops = bs * hw.A * math.ceil(hw.KVB / hw.U)

    # KV sharding split
    G = hw.EP
    local_frac = 1.0 / G if kv_shard else 1.0
    remote_frac = (G - 1) / G if kv_shard else 0.0
    merge_io = kv_shard and kv_merge

    local_bytes = kv_bytes * local_frac
    local_iops = 1 if merge_io else kv_iops * local_frac
    remote_bytes = kv_bytes * remote_frac

    if kvres == "hbm":
        kv_io_local = hbm_ms(local_bytes, hw)
    else:
        kv_io_local = ssd_ms(local_bytes, local_iops, hw)
    kv_io_nvlink = hbm_ms(remote_bytes, hw) if remote_bytes > 0 else 0
    # NVLink BW ≈ similar magnitude to HBM-style ms calc but with NV_BW
    if remote_bytes > 0:
        kv_io_nvlink = max(remote_bytes / (hw.NV_BW * GB) * 1000, hw.HBM_FLOOR)
    kv_io = max(kv_io_local, kv_io_nvlink)

    # Compute (TP-aware)
    cmp_base = (hw.c_sh_flat + hw.c_sh_perbs * bs) / hw.TP \
               + hw.c_un_flat + hw.c_un_perbs * bs
    cmpF = cmp_base + hw.t_topk
    cmpS = cmp_base

    # Per-layer arrays
    io_F, io_S = idx_io + kv_io, kv_io
    ios = []
    cmps = []
    for i in range(hw.L):
        isF = (scheme == "dsa") or (i % hw.FP == 0)
        ios.append(io_F if isF else io_S)
        cmps.append(cmpF if isF else cmpS)

    # Pipeline walk
    lps = math.ceil(hw.L / hw.PP)
    cum_io, end_comp = 0.0, 0.0
    for i in range(min(lps, hw.L)):
        cum_io += ios[i]
        end_comp = max(end_comp, cum_io) + cmps[i]
    for s in range(1, hw.PP):
        start = s * lps
        stop = min(hw.L, (s + 1) * lps)
        if start >= hw.L:
            break
        stage_io = sum(ios[start:stop])
        stage_cmp = sum(cmps[start:stop])
        end_comp = max(end_comp, stage_io) + stage_cmp

    tpot = end_comp

    # Sustained per-GPU rates
    ssd_bytes_tok = hw.L * local_bytes if kvres == "ssd" else 0
    ssd_iops_tok = hw.L * local_iops if kvres == "ssd" else 0
    nv_bytes_tok = hw.L * remote_bytes
    hbm_kv_bytes_tok = hw.L * local_bytes if kvres == "hbm" else 0
    hbm_bytes_tok = nF * idx_bytes + hbm_kv_bytes_tok

    ssd_bw_used = (ssd_bytes_tok / GB) / (tpot / 1000)
    ssd_iops_used = ssd_iops_tok / (tpot / 1000)
    nv_bw_used = (nv_bytes_tok / GB) / (tpot / 1000)
    hbm_bw_used = (hbm_bytes_tok / GB) / (tpot / 1000)

    return dict(
        bs=bs, tpot=tpot, kv_io=kv_io, idx_io=idx_io,
        kv_io_local=kv_io_local, kv_io_nvlink=kv_io_nvlink,
        cmpF=cmpF, cmpS=cmpS, F_layer=max(io_F, cmpF), S_layer=max(io_S, cmpS),
        ssd_bw_used=ssd_bw_used, ssd_iops_used=ssd_iops_used,
        nv_bw_used=nv_bw_used, hbm_bw_used=hbm_bw_used,
        kvres=kvres, kv_shard=kv_shard, kv_merge=kv_merge,
    )


# ---------------------- plots ----------------------

OUTDIR = "reports/figures/kv_sharing"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def plot_tpot_vs_sl(hw: HW, bsc=24):
    sls = [128 * 1024, 200 * 1024, 256 * 1024, 384 * 1024,
           512 * 1024, 768 * 1024, 1024 * 1024,
           2 * 1024 * 1024, 4 * 1024 * 1024]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in CONFIGS:
        ys = [evaluate(sl, bsc, cfg, hw)["tpot"] for sl in sls]
        ls, lw, mk = CONFIG_STYLES[cfg]
        ax.plot([s / 1024 for s in sls], ys,
                label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg],
                linewidth=lw, linestyle=ls, marker=mk, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length (K tokens, log)")
    ax.set_ylabel("TPOT (ms / token, log)")
    ax.set_title(f"TPOT vs sequence length — KV sharing configs\n"
                 f"BS_cluster={bsc}, PP={hw.PP} TP={hw.TP} EP={hw.EP}, IndexCache (F:S:S:S)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.axhline(30, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(sls[-1] / 1024 * 0.7, 32, "30 ms target", color="#666", fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "tpot_vs_sl.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_tpot_vs_bs(hw: HW, sl=512 * 1024):
    bscs = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for cfg in CONFIGS:
        ys = [evaluate(sl, b, cfg, hw)["tpot"] for b in bscs]
        ls, lw, mk = CONFIG_STYLES[cfg]
        ax.plot(bscs, ys,
                label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg],
                linewidth=lw, linestyle=ls, marker=mk, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BS_cluster (log)")
    ax.set_ylabel("TPOT (ms / token, log)")
    ax.set_title(f"TPOT vs cluster batch size — KV sharing configs\n"
                 f"sl={sl // 1024}K, PP={hw.PP} TP={hw.TP} EP={hw.EP}, IndexCache (F:S:S:S)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.axhline(30, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(bscs[-1] * 0.7, 32, "30 ms target", color="#666", fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "tpot_vs_bs.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_ssd_saturation(hw: HW, bsc=24):
    """Where does SSD hit its cap? Show BW and IOPS utilization vs sl
    for the three SSD configs."""
    sls = [128 * 1024, 200 * 1024, 256 * 1024, 384 * 1024,
           512 * 1024, 768 * 1024, 1024 * 1024,
           2 * 1024 * 1024, 4 * 1024 * 1024]
    ssd_cfgs = ["ssd_full", "ssd_shard", "ssd_shard_merge"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for cfg in ssd_cfgs:
        bw_pct = [evaluate(sl, bsc, cfg, hw)["ssd_bw_used"] / hw.SSD_BW * 100
                  for sl in sls]
        iops_pct = [evaluate(sl, bsc, cfg, hw)["ssd_iops_used"] / hw.SSD_IOPS * 100
                    for sl in sls]
        ls, lw, mk = CONFIG_STYLES[cfg]
        ax1.plot([s / 1024 for s in sls], bw_pct,
                 label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg],
                 linewidth=lw, linestyle=ls, marker=mk, markersize=5)
        ax2.plot([s / 1024 for s in sls], iops_pct,
                 label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg],
                 linewidth=lw, linestyle=ls, marker=mk, markersize=5)

    for ax, ylabel, title in [
        (ax1, "SSD BW used (% of cap)", f"SSD BW utilization vs sl  (cap = {hw.SSD_BW:.0f} GB/s)"),
        (ax2, "SSD IOPS used (% of cap)", f"SSD IOPS utilization vs sl  (cap = {hw.SSD_IOPS / 1e6:.1f} M IOPS)"),
    ]:
        ax.set_xscale("log")
        ax.set_xlabel("Sequence length (K tokens, log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(100, color="#C84B4B", linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(sls[-1] / 1024 * 0.5, 102, "100% cap", color="#C84B4B", fontsize=9)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(f"Where SSD runs out — BW and IOPS utilization\n"
                 f"BS_cluster={bsc}, PP={hw.PP} TP={hw.TP} EP={hw.EP}",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "ssd_saturation.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_max_bs_at_tpot(hw: HW, tpot_target=30.0):
    """At a given TPOT target, what's the max BS for each config across sl?"""
    sls = [128 * 1024, 200 * 1024, 256 * 1024, 384 * 1024,
           512 * 1024, 768 * 1024, 1024 * 1024, 2 * 1024 * 1024]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    def max_bs(sl, cfg):
        # binary search over BS_cluster (1..4096) for largest bsc with tpot <= target
        lo, hi = 1, 4096
        if evaluate(sl, 1, cfg, hw)["tpot"] > tpot_target:
            return 0
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if evaluate(sl, mid, cfg, hw)["tpot"] <= tpot_target:
                lo = mid
            else:
                hi = mid
        return lo

    for cfg in CONFIGS:
        ys = [max_bs(sl, cfg) for sl in sls]
        ls, lw, mk = CONFIG_STYLES[cfg]
        ax.plot([s / 1024 for s in sls], ys,
                label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg],
                linewidth=lw, linestyle=ls, marker=mk, markersize=6)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Sequence length (K tokens, log)")
    ax.set_ylabel(f"Max BS_cluster within {tpot_target:.0f} ms TPOT (log)")
    ax.set_title(f"Max cluster batch size achievable at {tpot_target:.0f} ms TPOT\n"
                 f"PP={hw.PP} TP={hw.TP} EP={hw.EP}, IndexCache. 0 = even BS=1 misses target.")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "max_bs_at_tpot.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_must_merge_frontier(hw: HW, tpot_target=30.0):
    """At what (sl, BS) point are you forced to merge the I/Os to stay
    within the TPOT target? Plots, for each sl, the BS_cluster at which
    ssd_shard (unmerged) crosses the TPOT target vs. ssd_shard_merge."""
    sls = [128 * 1024, 200 * 1024, 256 * 1024, 384 * 1024,
           512 * 1024, 768 * 1024, 1024 * 1024,
           2 * 1024 * 1024, 4 * 1024 * 1024]

    def crossover_bs(sl, cfg):
        # largest BS with TPOT <= target; 0 if even BS=1 fails
        lo, hi = 1, 4096
        if evaluate(sl, 1, cfg, hw)["tpot"] > tpot_target:
            return 0
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if evaluate(sl, mid, cfg, hw)["tpot"] <= tpot_target:
                lo = mid
            else:
                hi = mid
        return lo

    unmerged = [crossover_bs(sl, "ssd_shard") for sl in sls]
    merged   = [crossover_bs(sl, "ssd_shard_merge") for sl in sls]
    hbm_lim  = [crossover_bs(sl, "hbm_shard") for sl in sls]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = [s / 1024 for s in sls]

    # Shaded zones
    # Safe (no merge needed): below the unmerged ceiling
    ax.fill_between(x, 0, unmerged, color="#3FAA5B", alpha=0.18,
                    label="Safe — no merge needed")
    # Must-merge: between unmerged ceiling and merged ceiling
    ax.fill_between(x, unmerged, merged, color="#F0C03A", alpha=0.30,
                    label="MUST merge — unmerged misses target")
    # Beyond merge: above merged ceiling, SSD can't help (need HBM)
    top = max(max(merged), max(hbm_lim)) * 1.3
    ax.fill_between(x, merged, [top] * len(x), color="#C84B4B", alpha=0.18,
                    label="Even merged SSD fails — need HBM")

    ax.plot(x, unmerged, color="#F0C03A", linewidth=2.5, marker="o",
            markersize=6, label=f"ssd_shard (unmerged) — max BS at {tpot_target:.0f} ms")
    ax.plot(x, merged, color="#A06CD5", linewidth=2.5, marker="^",
            markersize=6, label=f"ssd_shard_merge — max BS at {tpot_target:.0f} ms")
    ax.plot(x, hbm_lim, color="#2EBFD9", linewidth=2.2, linestyle="--", marker="D",
            markersize=5, label=f"hbm_shard — max BS at {tpot_target:.0f} ms")

    ax.set_xscale("log")
    ax.set_xlabel("Sequence length (K tokens, log)")
    ax.set_ylabel(f"Max BS_cluster within {tpot_target:.0f} ms TPOT")
    ax.set_title(
        f"The must-merge zone — where unmerged SSD I/O can't keep up\n"
        f"PP={hw.PP} TP={hw.TP} EP={hw.EP}, IndexCache, KV-on-SSD sharded")
    ax.set_ylim(0, top)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "must_merge_frontier.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    hw = HW()
    paths = [
        plot_tpot_vs_sl(hw, bsc=24),
        plot_tpot_vs_bs(hw, sl=512 * 1024),
        plot_ssd_saturation(hw, bsc=24),
        plot_max_bs_at_tpot(hw, tpot_target=30.0),
        plot_must_merge_frontier(hw, tpot_target=30.0),
    ]
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
