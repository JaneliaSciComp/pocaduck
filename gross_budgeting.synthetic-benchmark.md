# gross_budgeting synthetic benchmark

Synthetic-only evidence for the first `gross_budgeting` implementation. These
are stored-sample preservation proxies, not neuron morphology or training-quality
measurements. A real-store run remains for the orchestrating session.

## Provenance and configuration

- Date: 2026-08-24
- PoCADuck implementation: `5e09e470ff54fa5ce26ad5b0ab59a2f79dcea10d`
- mo-gen reference: `b6c9743c3ffe7e21284a6c07f8f6c8851ba65401`
- nps reference: `53b4e14a52c765d99db62837f3b0cb303f792dfe`
- Hardware: Intel Core i9-13900K, 32 logical CPUs, Linux 6.8
- Compute environment: Python 3.10.8, NumPy 1.21.6, scikit-learn 1.2.2;
  OMP/MKL/OpenBLAS/NumExpr thread variables unset; sklearn baseline uses
  `n_jobs=1` as mo-gen does.
- Cohort: 3 generated soma/neurite labels in each 1,500 / 100,000 / 1,000,000
  point class; seeds 0–4; budget 4096; voxel size 16×16×40 nm;
  `soma_threshold=8.0`; old-path ratios 0.90 and 0.75.
- Timing: external `perf_counter`, preloaded arrays, randomized method order.
  OS page-cache state is uncontrolled but irrelevant to compute-only results.

The full command used the script's compute mode and the stated cohort plus
coverage edges 500/1000/2000/4000 nm and recall distances 100/250/500/1000 nm.
The generated JSON reported no cohort warnings.

## Compute-only distributions

Median seconds across 3 labels × 5 seeds (p95 in parentheses):

| Stored points | gross_budgeting | old 0.75 | old 0.90 |
|---:|---:|---:|---:|
| 1,500 | 0.00137 (0.00148) | 0.000055 (0.000068) | 0.000059 (0.000066) |
| 100,000 | 0.10296 (0.10564) | 0.16342 (0.16556) | 0.20065 (0.20537) |
| 1,000,000 | 1.41474 (1.44464) | 2.67532 (2.71805) | 3.64164 (3.75432) |

The 1,500-point class never reaches the 4096-point density stage: the new path
returns all stored points, while the old path performs only its uniform 0.75 or
0.90 thinning. The comparable 100k and 1M classes support the expected compute
win after benchmark-driven implementation cleanup (compact cell IDs during the
ladder scan and one grouped pass-2 index sort).

## Stored-sample proxies

Known-mask median soma budget share:

| Stored points | gross_budgeting | old 0.75 | old 0.90 |
|---:|---:|---:|---:|
| 1,500 | 0.6200 | 0.6196 | 0.6193 |
| 100,000 | 0.6497 | 0.7036 | 0.7100 |
| 1,000,000 | 0.5583 | 0.6921 | 0.6978 |

The configurable default `soma_threshold=8.0` is retained: on the two classes
that actually enter equal-size 4096-point selection it reduces known-mask soma
share relative to both old-path ratios. This remains a per-partition cap, not a
global soma-share guarantee; real-store calibration is still required.

Selected additional medians:

- At a 500 nm evaluation grid, coverage at 1M points was 0.8750 new versus
  0.8172 / 0.8094 old (0.75 / 0.90). This independent sweep is not the strategy's
  solved-grid guarantee, which is covered directly by unit tests.
- At 100 nm, spatial recall at 1M was essentially tied: 0.4723 new versus
  0.4731 / 0.4724 old. At 100k it was 0.9958 new versus 0.9903 / 0.9910 old.
- Mean-pair draw overlap remained non-degenerate. Median overlap was 0.0950 at
  100k and 0.0251 at 1M for the new path (old: about 0.050–0.052 and
  0.0044–0.0051 respectively). The 1,500-point new result is deterministically
  all points by contract.

## End-to-end and remaining gate

The fresh-Query and warm-LRU code paths were both executed on a separate small
3-label × 5-seed synthetic matrix (300/500/700 points, budget 1024). Fresh
`gross_budgeting` medians were 8.47/8.45/9.80 ms; warm-LRU counterfactual
medians were 0.503/0.647/0.763 ms. The faithful old lifecycle was also exercised,
but these small clouds stay below the density stage because the PoCADuck runtime
environment intentionally lacks scikit-learn. Therefore this run validates the
construct/query-once/close lifecycle and cache-state reporting, not a large-cloud
end-to-end speed comparison.

No real PoCADuck store was accessible in this session. The script accepts
`--store-path` and `--labels` for the required real-store run; until that runs,
the slice's real-store benchmark acceptance criterion remains open.
