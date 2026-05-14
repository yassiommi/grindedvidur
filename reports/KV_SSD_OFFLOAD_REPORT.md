# DSA vs IndexCache with KV on SSD

**Config:** PP=2, TP=2, EP=8, FP8. The MLA KV cache lives on a local NVMe
SSD per GPU; the indexer K cache stays in HBM. All compute is analytical
(H100, no profiled data). Per-PP-stage producer/consumer schedule.

Source: `experiments/dsa_vs_ic_kv_ssd.py`.

## What changes vs HBM-only

| stream             | HBM-only mode | KV-on-SSD mode |
|--------------------|--------------:|---------------:|
| indexer K (F only) | HBM (1384 GB/s) | HBM (1384 GB/s) |
| MLA KV gather      | HBM (1384 GB/s) | **SSD (7 or 28 GB/s)** |
| weights, activations | HBM | HBM |
| compute            | TP-sharded compute | TP-sharded compute |

Both DSA and IndexCache pay the same KV-SSD cost (every layer gathers
2560 attended tokens × 576 B = 1.4 MB at BS=1). IndexCache saves only on
the **indexer K HBM reads** (45 of 61 layers' worth).

## Results — KV on Gen4 NVMe (7 GB/s, single drive)

| scenario     | DSA TPOT | IC TPOT | speedup | DSA tok/s | IC tok/s |
|---           |    ---:  |   ---:  |   ---:  |    ---:   |   ---:   |
| 32K  BS=1    |  18.14 ms| 18.59 ms| 0.98×   |  55.1     | 53.8     |
| 128K BS=1    |  17.85 ms| 18.54 ms| 0.96×   |  56.0     | 53.9     |
| 200K BS=1    |  17.88 ms| 18.56 ms| 0.96×   |  55.9     | 53.9     |
| 512K BS=1    |  20.63 ms| 18.67 ms| **1.10×** | 48.5    | 53.6     |
| 1M   BS=1    |  26.23 ms| 18.85 ms| **1.39×** | 38.1    | 53.0     |
| 2M   BS=1    |  37.43 ms| 21.36 ms| **1.75×** | 26.7    | 46.8     |
| 4M   BS=1    |  59.83 ms| 27.14 ms| **2.20×** | 16.7    | 36.8     |
| 32K  BS=8    |  86.41 ms| 84.28 ms| 1.03×   |  92.6     | 94.9     |
| 200K BS=8    | 101.11 ms| 88.08 ms| **1.15×** | 79.1    | 90.8     |
| 32K  BS=32   | 322.85 ms|312.73 ms| 1.03×   |  99.1     | 102.3    |
| 200K BS=32   | 381.65 ms|327.91 ms| **1.16×** | 83.8    | 97.6     |

## Results — KV on 4× Gen4 NVMe RAID (28 GB/s)

| scenario     | DSA TPOT | IC TPOT | speedup | DSA tok/s | IC tok/s |
|---           |    ---:  |   ---:  |   ---:  |    ---:   |   ---:   |
| 32K  BS=1    |  17.99 ms| 18.44 ms| 0.98×   |  55.6     | 54.2     |
| 128K BS=1    |  17.71 ms| 18.39 ms| 0.96×   |  56.5     | 54.4     |
| 200K BS=1    |  17.73 ms| 18.42 ms| 0.96×   |  56.4     | 54.3     |
| 512K BS=1    |  17.84 ms| 18.53 ms| 0.96×   |  56.0     | 54.0     |
| 1M   BS=1    |  21.70 ms| 18.71 ms| **1.16×** | 46.1    | 53.5     |
| 2M   BS=1    |  32.90 ms| 19.07 ms| **1.73×** | 30.4    | 52.4     |
| 4M   BS=1    |  55.30 ms| 22.90 ms| **2.41×** | 18.1    | 43.7     |
| 32K  BS=8    |  69.27 ms| 69.17 ms| 1.00×   | 115.5     | 115.7    |
| 200K BS=8    |  69.74 ms| 69.64 ms| 1.00×   | 114.7     | 114.9    |
| 32K  BS=32   | 232.23 ms|228.69 ms| 1.02×   | 137.8     | 139.9    |
| 200K BS=32   | 235.69 ms|230.59 ms| 1.02×   | 135.8     | 138.8    |

## Takeaways

- **<1.0× at small BS, short sl** — same effect as pure HBM: IC zeroes
  `idx_io` on S layers, so `block_a` becomes exposed compute and TPOT
  rises ~3 %.
- **BS=1, sl ≥ 1M** — DSA's indexer-K HBM reads dominate. IC saves 45/61
  of those, so the win grows monotonically with sl: **1.39× → 2.20×** at
  4M on a single Gen4 drive; **1.16× → 2.41×** with a faster RAID.
- **BS ≥ 8 SSD-bottlenecked** — KV gather scales with BS: at BS=32 sl=200K
  SSD is 47 MB × 30 layers/stage = ~200 ms (single drive) or ~50 ms (RAID).
  Both DSA and IC pay this equally, so IC's relative win shrinks to
  ~1.02–1.16×.
- **Better SSDs help DSA more than IC at BS≥8** because KV gather is the
  shared dominant cost; reducing it doesn't help IC's headroom.
- **Better SSDs help IC more than DSA at BS=1 long sl** because the KV
  gather cost goes away and what remains is the indexer-K HBM read,
  which IC eliminates on S layers.
