"""Synthetic contract tests for the pure gross-budgeting allocator."""

import unittest
from unittest import mock

import numpy as np

from pocaduck.budgeting import (
    DEFAULT_SOMA_THRESHOLD,
    allocate_partition_quotas,
    bin_coordinates_nm,
    candidate_edge_lengths,
    gross_budget_points,
    plan_gross_budgeting,
    sample_budget_plan,
)


VOXEL_SIZE = (4.0, 4.0, 20.0)


def soma_neurite_fixture(seed=123):
    """Return anisotropic voxel coordinates and detector-independent soma truth."""

    rng = np.random.default_rng(seed)
    voxel_size = np.asarray(VOXEL_SIZE)
    center_nm = np.array([400.0, 400.0, 400.0])

    directions = rng.normal(size=(12_000, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = 80.0 * np.cbrt(rng.random(len(directions)))
    soma_nm = center_nm + directions * radii[:, None]
    soma_voxel = np.rint(soma_nm / voxel_size).astype(np.int64)

    x_nm = np.linspace(480.0, 8_480.0, 1_200)
    neurite_nm = np.column_stack(
        [
            x_nm,
            400.0 + 20.0 * np.sin(x_nm / 300.0),
            400.0 + 20.0 * np.cos(x_nm / 300.0),
        ]
    )
    neurite_voxel = np.rint(neurite_nm / voxel_size).astype(np.int64)

    coordinates = np.vstack([soma_voxel, neurite_voxel])
    known_soma_mask = np.arange(len(coordinates)) < len(soma_voxel)
    # Unique row ids make it possible to recover the fixture truth after a draw
    # and exercise carrying arbitrary extra columns through by row index.
    row_ids = np.arange(len(coordinates), dtype=np.int64)
    points = np.column_stack(
        [coordinates, row_ids, row_ids + 100_000, -(row_ids + 1)]
    )
    return points, known_soma_mask


class TestGrossBudgeting(unittest.TestCase):
    def test_candidate_ladder_and_upper_face_clamp(self):
        candidates = candidate_edge_lengths((100.0, 20.0, 0.0), (4, 4, 20))
        self.assertEqual(candidates[0], 100.0)
        self.assertEqual(candidates[-1], 20.0)
        self.assertTrue(np.all(candidates[:-1] > candidates[1:]))
        np.testing.assert_allclose(candidates[:-2] / candidates[1:-1], 1.25)

        coordinates = np.array([[0.0, 0.0, 0.0], [100.0, 20.0, 0.0]])
        bins, extent = bin_coordinates_nm(coordinates, coordinates.min(axis=0), 100)
        np.testing.assert_array_equal(extent, [1, 1, 1])
        np.testing.assert_array_equal(bins, np.zeros((2, 3), dtype=np.int64))

    def test_smallest_feasible_candidate_is_selected(self):
        points, _ = soma_neurite_fixture()
        budget = 512
        plan = plan_gross_budgeting(points, budget, VOXEL_SIZE)
        coordinates_nm = points[:, :3].astype(np.float64) * VOXEL_SIZE

        feasible = []
        for edge in plan.candidate_edge_lengths_nm:
            bins, _ = bin_coordinates_nm(coordinates_nm, plan.bbox_min_nm, edge)
            if len(np.unique(bins, axis=0)) <= budget:
                feasible.append(edge)
        self.assertEqual(plan.edge_length_nm, min(feasible))
        self.assertLessEqual(len(plan.occupied_bins), budget)

    def test_soma_neurite_coverage_cap_exact_size_and_extra_columns(self):
        points, known_soma_mask = soma_neurite_fixture()
        budget = 512
        plan = plan_gross_budgeting(points, budget, VOXEL_SIZE)
        selected = sample_budget_plan(points, plan, seed=7)

        self.assertEqual(selected.shape, (budget, points.shape[1]))
        selected_rows = selected[:, 3]
        selected_partitions = np.unique(plan.partition_ids[selected_rows])
        self.assertEqual(len(selected_partitions), len(plan.occupied_bins))
        self.assertTrue(np.all(plan.quotas >= 1))
        self.assertTrue(np.all(plan.quotas <= plan.counts))
        self.assertGreater(np.count_nonzero(plan.soma_mask), 0)

        base_quota = budget // len(plan.occupied_bins)
        non_relaxed_soma = plan.quotas[plan.soma_mask]
        self.assertTrue(np.all(non_relaxed_soma <= base_quota))
        selected_soma_share = np.mean(known_soma_mask[selected_rows])
        self.assertLess(selected_soma_share, 0.5)

        for extra_column in range(3, points.shape[1]):
            np.testing.assert_array_equal(
                selected[:, extra_column], points[selected_rows, extra_column]
            )

    def test_seed_reproducibility_and_fresh_draws(self):
        points, _ = soma_neurite_fixture()
        first = gross_budget_points(points, 512, VOXEL_SIZE, seed=99)
        second = gross_budget_points(points, 512, VOXEL_SIZE, seed=99)
        different = gross_budget_points(points, 512, VOXEL_SIZE, seed=100)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))

    def test_exact_n_and_all_points_contract(self):
        points, _ = soma_neurite_fixture()
        selected = gross_budget_points(points, 257, VOXEL_SIZE, seed=1)
        self.assertEqual(len(selected), 257)

        all_points = gross_budget_points(points[:31], 100, VOXEL_SIZE, seed=1)
        np.testing.assert_array_equal(all_points, points[:31])

    def test_all_points_shortcut_skips_planner_after_validation(self):
        points = np.arange(20, dtype=np.int64).reshape(5, 4)

        with mock.patch("pocaduck.budgeting.plan_gross_budgeting") as planner:
            selected = gross_budget_points(points, 5, VOXEL_SIZE, seed=1)

        planner.assert_not_called()
        np.testing.assert_array_equal(selected, points)
        self.assertIsNot(selected, points)

        invalid_calls = [
            lambda: gross_budget_points(points, 0, VOXEL_SIZE),
            lambda: gross_budget_points(points, 10, (4, 4)),
            lambda: gross_budget_points(points, 10, VOXEL_SIZE, soma_threshold=1),
            lambda: gross_budget_points(np.zeros((2, 2)), 10, VOXEL_SIZE),
        ]
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call), self.assertRaises(ValueError):
                invalid_call()

    def test_all_soma_relaxes_cap_and_no_soma_stays_uncapped(self):
        row_ids = np.arange(300, dtype=np.int64)
        soma_coordinates = np.repeat(
            np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.int64),
            100,
            axis=0,
        )
        all_soma = np.column_stack(
            [soma_coordinates, row_ids]
        )
        soma_plan = plan_gross_budgeting(
            all_soma,
            50,
            VOXEL_SIZE,
            sampling_fraction=1.0,
        )
        self.assertTrue(np.all(soma_plan.soma_mask))
        self.assertEqual(int(soma_plan.quotas.sum()), 50)
        base_quota = 50 // len(soma_plan.quotas)
        self.assertGreater(np.max(soma_plan.quotas), base_quota)

        no_soma = np.column_stack(
            [np.arange(100), np.zeros(100, dtype=int), np.zeros(100, dtype=int)]
        )
        no_soma_plan = plan_gross_budgeting(no_soma, 25, (1, 1, 1))
        self.assertFalse(np.any(no_soma_plan.soma_mask))
        self.assertEqual(int(no_soma_plan.quotas.sum()), 25)

    def test_insufficient_non_soma_capacity_relaxes_soma_cap(self):
        counts = np.array([100, 2, 2], dtype=np.int64)
        quotas = allocate_partition_quotas(
            counts, np.array([True, False, False]), budget=50
        )
        np.testing.assert_array_equal(quotas, [46, 2, 2])

    def test_one_point_and_empty_clouds(self):
        one = np.array([[7, 8, 9, 10, 11]], dtype=np.int64)
        np.testing.assert_array_equal(
            gross_budget_points(one, 1, VOXEL_SIZE, seed=5), one
        )

        empty_2d = np.empty((0, 7), dtype=np.int64)
        selected_2d = gross_budget_points(empty_2d, 10, VOXEL_SIZE)
        self.assertEqual(selected_2d.shape, (0, 7))
        empty_query_shape = np.array([], dtype=np.int64)
        selected_1d = gross_budget_points(empty_query_shape, 10, VOXEL_SIZE)
        self.assertEqual(selected_1d.shape, (0,))

    def test_absolute_soma_form_uses_sampling_fraction(self):
        row_ids = np.arange(100, dtype=np.int64)
        points = np.column_stack(
            [np.zeros((100, 3), dtype=np.int64), row_ids]
        )
        relative = plan_gross_budgeting(points, 20, VOXEL_SIZE)
        absolute = plan_gross_budgeting(
            points, 20, VOXEL_SIZE, sampling_fraction=0.01
        )
        self.assertFalse(np.any(relative.soma_mask))
        self.assertTrue(np.all(absolute.soma_mask))

    def test_invalid_parameters(self):
        points = np.zeros((3, 3), dtype=np.int64)
        invalid_calls = [
            lambda: plan_gross_budgeting(points, 0, VOXEL_SIZE),
            lambda: plan_gross_budgeting(points, -1, VOXEL_SIZE),
            lambda: plan_gross_budgeting(points, 1.5, VOXEL_SIZE),
            lambda: plan_gross_budgeting(points, True, VOXEL_SIZE),
            lambda: plan_gross_budgeting(points, 1, (4, 4)),
            lambda: plan_gross_budgeting(points, 1, (4, 0, 20)),
            lambda: plan_gross_budgeting(points, 1, (4, np.nan, 20)),
            lambda: plan_gross_budgeting(points, 1, VOXEL_SIZE, soma_threshold=1),
            lambda: plan_gross_budgeting(
                points, 1, VOXEL_SIZE, soma_threshold=np.inf
            ),
            lambda: plan_gross_budgeting(
                points, 1, VOXEL_SIZE, sampling_fraction=0
            ),
            lambda: plan_gross_budgeting(
                points, 1, VOXEL_SIZE, sampling_fraction=1.1
            ),
            lambda: plan_gross_budgeting(np.zeros((2, 2)), 1, VOXEL_SIZE),
        ]
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call), self.assertRaises(ValueError):
                invalid_call()

        self.assertEqual(DEFAULT_SOMA_THRESHOLD, 8.0)


if __name__ == "__main__":
    unittest.main()
