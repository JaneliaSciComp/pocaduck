# gross_budgeting synthetic benchmark

Synthetic-only evidence for `gross_budgeting`. These measurements are
stored-sample preservation proxies, not neuron morphology or training-quality
measurements. A real-store run remains for the orchestrating session.

## Provenance and configuration

- Date: 2026-08-24
- Measured PoCADuck implementation: `87823e4e1a9605ae52cc552fcd77ce80a168455b`
- Review follow-ups after the measured SHA: `32de248` (declare `tqdm`),
  `595828b` (correct coverage proxy binning), and `1e645f4` (all-points
  shortcut).
- mo-gen reference: `b6c9743c3ffe7e21284a6c07f8f6c8851ba65401`
- nps reference: `53b4e14a52c765d99db62837f3b0cb303f792dfe`
- Hardware: 13th Gen Intel Core i9-13900K, 32 logical CPUs, Linux 6.8.
- Canonical environment: Python 3.12.14, NumPy 2.4.6, scikit-learn 1.9.0;
  OMP/MKL/OpenBLAS/NumExpr thread variables unset; the sklearn baseline uses
  `n_jobs=1` as mo-gen does.
- Cohort: 3 generated soma/neurite labels in each 1,500 / 100,000 / 1,000,000
  point class; seeds 0–4; budget 4096; voxel size 16×16×40 nm;
  `soma_threshold=8.0`; old-path ratios 0.90 and 0.75.
- Timing: external `perf_counter` calls, randomized method order, compute-only
  on preloaded arrays plus consumer-faithful fresh-Query and separate warm-LRU
  modes. OS page-cache state is uncontrolled.
- Source artifact:
  `/home/katzw/work/zen-point-clouds/tmp/gross-budgeting-synthetic-benchmark.json`.
  It reported no cohort warnings.

This canonical-environment run supersedes the earlier side-environment run on
Python 3.10.8, NumPy 1.21.6, and scikit-learn 1.2.2.

## Compute-only distributions

Median seconds across 3 labels × 5 seeds (p95 in parentheses):

| Stored points | gross_budgeting | old 0.75 | old 0.90 |
|---:|---:|---:|---:|
| 1,500 | 0.001395 (0.002812) | 0.000055 (0.000080) | 0.000054 (0.000065) |
| 100,000 | 0.105066 (0.111931) | 0.152297 (0.155134) | 0.186788 (0.193425) |
| 1,000,000 | 1.461999 (1.477326) | 2.408265 (2.605990) | 3.227983 (3.453668) |

The measured 1,500-point new path still scanned the candidate ladder. Commit
`1e645f4`, which postdates the JSON, now validates inputs and returns all stored
points without pass 1 when the cloud fits within the budget. Its small-cloud
timing therefore supersedes that one row; the 100k and 1M comparisons are
unaffected.

## Consumer-faithful end-to-end distributions

Fresh lifecycle: construct a new `Query`, retrieve once, and close it for every
method call. Median seconds (p95 in parentheses):

| Stored points | gross_budgeting | old 0.75 | old 0.90 |
|---:|---:|---:|---:|
| 1,500 | 0.014033 (0.017449) | 0.012663 (0.017573) | 0.012697 (0.014690) |
| 100,000 | 0.245012 (0.250307) | 0.304902 (0.308842) | 0.340708 (0.344232) |
| 1,000,000 | 2.828726 (2.871618) | 4.113932 (4.241203) | 4.994338 (5.162207) |

The previously missing 1M-point fresh-lifecycle comparison is now measured:
the median was 2.83 seconds new versus 4.11 / 4.99 seconds old at ratios 0.75 /
0.90.

Warm-LRU counterfactual: reuse one `Query` after preloading every label. Median
seconds (p95 in parentheses):

| Stored points | gross_budgeting | old 0.75 | old 0.90 |
|---:|---:|---:|---:|
| 1,500 | 0.001295 (0.001431) | 0.000047 (0.000097) | 0.000044 (0.000064) |
| 100,000 | 0.099370 (0.101581) | 0.158023 (0.158398) | 0.193265 (0.198713) |
| 1,000,000 | 1.361155 (1.381792) | 2.663812 (2.721506) | 3.582056 (3.665241) |

The warm mode is reported separately because mo-gen's consumer lifecycle
constructs and closes a fresh `Query` per load.

## Stored-sample proxies

Known-mask median soma budget share:

| Stored points | gross_budgeting | old 0.75 | old 0.90 |
|---:|---:|---:|---:|
| 1,500 | 0.6200 | 0.6196 | 0.6193 |
| 100,000 | 0.6497 | 0.7036 | 0.7100 |
| 1,000,000 | 0.5583 | 0.6921 | 0.6978 |

The configurable default `soma_threshold=8.0` is retained: on the two classes
that enter equal-size 4096-point selection it reduces known-mask soma share
relative to both old-path ratios. This remains a per-partition cap, not a global
soma-share guarantee; real-store calibration is still required.

Selected additional medians:

- At a 500 nm evaluation grid, coverage at 1M points was 0.8750 new versus
  0.8172 / 0.8094 old (0.75 / 0.90). These JSON values were not re-derived
  after `595828b`; that fix affects the proxy only when a selected subset's max
  face lies exactly on a raw-grid boundary. The corrected behavior is covered
  by a focused regression test.
- At 100 nm, spatial recall at 1M was essentially tied: 0.4723 new versus
  0.4731 / 0.4724 old. At 100k it was 0.9958 new versus 0.9903 / 0.9910 old.
- Mean-pair draw overlap remained non-degenerate. Median overlap was 0.0950 at
  100k and 0.0251 at 1M for the new path (old: about 0.050–0.052 and
  0.0044–0.0051 respectively). The 1,500-point new result is deterministically
  all points by contract.

## Remaining gate

No real PoCADuck store was used in this run. The script accepts `--store-path`
and `--labels` for a real-store benchmark; until that runs, no claim is made
about real-data morphology or workload distributions.
