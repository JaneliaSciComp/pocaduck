"""Focused tests for the standalone gross-budgeting benchmark."""

import contextlib
import io
import unittest

import numpy as np

import benchmark_gross_budgeting as benchmark


class TestGrossBudgetingBenchmark(unittest.TestCase):
    def test_defaults_define_required_synthetic_cohort(self):
        args = benchmark.parse_args([])
        self.assertEqual(args.synthetic_counts, [1_500, 100_000, 1_000_000])
        self.assertEqual(args.labels_per_size, 3)
        self.assertEqual(args.seeds, [0, 1, 2, 3, 4])
        self.assertEqual(args.subsample_ratios, [0.9, 0.75])
        self.assertEqual(args.modes, ["compute", "end-to-end"])
        self.assertEqual(args.cache_states, ["fresh", "warm"])

    def test_store_path_and_labels_are_required_together(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmark.parse_args(["--store-path", "/tmp/store"])
            with self.assertRaises(SystemExit):
                benchmark.parse_args(["--labels", "1,2,3"])

    def test_old_path_subsamples_before_budget_check(self):
        points = np.column_stack(
            [
                np.arange(4096),
                np.zeros(4096, dtype=np.int64),
                np.zeros(4096, dtype=np.int64),
            ]
        )
        selected_90, timings_90 = benchmark.old_path_indices(
            points, budget=4096, subsample_ratio=0.90, seed=7
        )
        selected_75, timings_75 = benchmark.old_path_indices(
            points, budget=4096, subsample_ratio=0.75, seed=7
        )
        self.assertEqual(len(selected_90), 3686)
        self.assertEqual(len(selected_75), 3072)
        self.assertEqual(timings_90["density_seconds"], 0)
        self.assertEqual(timings_75["density_seconds"], 0)
        self.assertEqual(len(np.unique(selected_90)), len(selected_90))
        self.assertEqual(len(np.unique(selected_75)), len(selected_75))

    def test_synthetic_fixture_has_known_soma_mask_and_extra_columns(self):
        cases = benchmark.generate_synthetic_cases(
            counts=[200], labels_per_size=3, voxel_size=(16, 16, 40)
        )
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(case.points.shape, (200, 5))
            self.assertEqual(case.soma_mask.shape, (200,))
            self.assertEqual(case.soma_truth_source, "known synthetic fixture mask")
            np.testing.assert_array_equal(
                case.points[:, 4].astype(bool), case.soma_mask
            )

    def test_coverage_proxy_uses_nm_partition_sweep(self):
        raw = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
        selected = raw[[0, 2]]
        fine = benchmark.coverage_fraction(raw, selected, (10, 10, 10), 50)
        coarse = benchmark.coverage_fraction(raw, selected, (10, 10, 10), 500)
        self.assertEqual(fine, 2 / 3)
        self.assertEqual(coarse, 1.0)

    def test_coverage_proxy_keeps_raw_extent_for_selected_boundary(self):
        raw = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]])
        selected = raw[[0, 1]]

        coverage = benchmark.coverage_fraction(
            raw, selected, (1, 1, 1), edge_nm=10
        )

        self.assertEqual(coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
