# gross_budgeting real-store benchmark (FAFB flywire_cave_post)

First real-store run of `benchmark_gross_budgeting.py`, closing the slice's
real-data acceptance criterion. Read alongside
`gross_budgeting.synthetic-benchmark.md`. Headline honestly stated: **the
soma cap and small-cloud coverage wins replicate on real data, but the
synthetic compute win does not** — on this cohort and host, pass 1's
candidate-ladder scan costs more than the old path's kNN.

## Provenance and configuration

- Date: 2026-08-24
- Host: **emdata4** (FlyEM server, Intel Xeon Platinum 8168 @ 2.70 GHz,
  Scientific Linux 7.9). Deliberately noted: an old server with possibly
  non-equivalent NIC — timing here is evidence about *relative* method cost
  on shared hardware, not canonical absolute performance.
- Environment: fresh conda-forge prefix env (`~/agentic-dev/env` on emdata4):
  Python 3.12, NumPy 2.5.2, DuckDB 1.5.5, scikit-learn 1.9.0. Code:
  `~/agentic-dev/pocaduck` rsynced at branch `gross-budgeting` @ `791f6c9`
  (the report JSON records `pocaduck_sha: null` because the host's git could
  not read the repo; the sha is pinned here from the rsync source).
- Store: `/nrs/turaga/jakob/autoproof_data/flywire_cave_post/points`
  (optimized layout; index scanned 169,297,511 labels — median stored count
  is 1 point, p99 = 24, max ≈ 30M; the store is dominated by tiny fragments).
- Cohort (3 labels per class, spread across each class's range; the script
  re-buckets by actual count):

  | Class | Label | Stored points |
  |---|---|---:|
  | real_~1e3 | 720575940596075921 | 1,000 |
  | real_~1e3 | 720575940627721624 | 4,522 |
  | real_~1e3 | 720575940632305036 | 9,999 |
  | real_~1e5 | 720575940642658267 | 50,001 |
  | real_~1e5 | 720575940623260201 | 74,322 |
  | real_~1e5 | 720575940635119723 | 299,516 |
  | real_~1e6 | 720575940615366055 | 300,080 |
  | real_~1e6 | 720575940620048133 | 409,355 |
  | real_~1e6 | 720575940619778133 | 2,155,124 |

- Config: budget 4096, voxel size 16×16×40 nm, seeds 0–4, ratios 0.90/0.75,
  `soma_threshold=8.0` (relative form; no `sampling_fraction`). Full default
  modes and cache states; no cohort warnings. Wall clock ≈ 26 min.
- Raw artifact: `gross-budgeting-real-store.json` (kept in the orchestrating
  workspace at `tmp/gross-budgeting-real-store.json` and on emdata4 at
  `~/agentic-dev/`).

## Timing (median seconds [p95], new vs old 0.75 / old 0.90)

Class rows mix three very different label sizes, so the median tracks each
class's middle label and p95 tracks its largest (the 2.16M-point label
dominates every `real_~1e6` p95).

| Mode | real_~1e5 | real_~1e6 |
|---|---|---|
| compute-only | 0.381 [1.57] vs 0.204 [1.09] / 0.236 [1.26] | 2.237 [18.9] vs 1.635 [15.3] / 2.032 [20.6] |
| end-to-end, fresh Query | 0.635 [2.37] vs 0.466 [1.86] / 0.496 [2.04] | 3.224 [25.1] vs 2.607 [22.6] / 3.063 [33.2] |
| end-to-end, warm LRU | 0.373 [1.65] vs 0.207 [1.11] / 0.238 [1.31] | 2.441 [20.0] vs 1.686 [15.7] / 2.119 [21.4] |

**The synthetic compute win inverts on real data:** gross_budgeting is
~10–85% slower than the old path at the class medians (fastest old ratio,
0.75, is the stronger baseline). Interpretation, consistent across modes:
real neurons have far larger bounding-box extents than the synthetic
fixtures, so the 1.25-ratio ladder from max span (~mm) down to voxel pitch
(40 nm) is ~45+ rungs of full-cloud binning — O(T·N) with a large T — while
the old path's kNN is cheaper on real, sparse, filamentous clouds
(kd-tree-friendly) than on the dense synthetic soma balls. The
pre-implementation speed expectation was recorded as a hypothesis; this run
refutes it for this store/host. Optimization headroom exists (the scan
re-bins all N points at every rung; early termination or a coarse-to-fine
scan could cut T) but is a follow-up decision, not part of this slice.

### Hardware-decoupling addendum (2026-08-26)

To rule out the old host as the cause, the cohort's raw clouds were dumped
to npz on emdata4 and the compute-only comparison re-run on a modern
i9-13900K with the arrays in local memory (same selector functions, seeds,
ratios, randomized order; per-label medians of 5 seeds; raw numbers in the
orchestrating workspace at `tmp/cohort-local-compute-rerun.json`):

| Points | gross | old 0.75 | old 0.90 | gross / old 0.75 |
|---:|---:|---:|---:|---:|
| 9,999 | 0.028 | 0.012 | 0.014 | 2.25× |
| 74,322 | 0.173 | 0.095 | 0.111 | 1.82× |
| 409,355 | 1.033 | 0.745 | 0.946 | 1.39× |
| 2,155,124 | 10.354 | 7.935 | 10.522 | 1.30× |

**The inversion persists on modern hardware — it is data-driven, not a
host artifact.** The ratio declines with size (kNN scales worse than the
linear ladder scan), reaching parity with the 0.90 reference config at the
largest neuron but never beating 0.75. The 4,522-point label (not shown)
is a structural outlier: the old path's subsample drops it below 4,096, so
kNN is skipped entirely and the old path is just a uniform draw there.

## Stored-sample proxies

Soma shares are **detector-relative** (the store has no soma ground truth;
the mask is gross_budgeting's own flag at the solved grid, applied to both
methods' selections).

| Metric (median) | real_~1e3 | real_~1e5 | real_~1e6 |
|---|---|---|---|
| soma budget share, new | 0.000 | **0.057** | **0.030** |
| soma budget share, old 0.75 / 0.90 | 0.000 / 0.000 | 0.066 / 0.071 | 0.086 / 0.088 |
| coverage @500 nm, new | **0.992** | 0.156 | 0.076 |
| coverage @500 nm, old 0.75 / 0.90 | 0.932 / 0.978 | 0.147 / 0.147 | 0.075 / 0.075 |
| draw overlap, new | 0.951 | 0.241 | 0.088 |
| draw overlap, old 0.75 / 0.90 | 0.748 / 0.898 | 0.282 / 0.333 | 0.023 / 0.024 |

- **Soma cap replicates on real data**: strictly below both old ratios at
  both equal-N classes (2.9× lower at ~1e6 vs the 0.90 baseline). Absolute
  shares are small (≤0.09) on this cohort — no soma domination observed, so
  the recorded global-soma-share follow-up looks unnecessary on present
  evidence.
- **Coverage at a fixed 500 nm grid** is a wash at large sizes (4096 points
  cannot cover a 2M-point cloud's 500 nm cells under any allocation; the
  guarantee lives at the *solved* partition size, which is much coarser) and
  favors the new path at ~1e3 (0.992 vs 0.932/0.978), where budgeting is
  active for the two labels above 4096 points.
- **Draw overlap** stays non-degenerate everywhere; the high ~1e3 value is
  structural (a 4,522-point cloud drawn to 4096 twice must overlap ≥ ~0.89).
