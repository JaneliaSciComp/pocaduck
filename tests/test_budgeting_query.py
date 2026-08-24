"""Query integration tests for the gross-budgeting strategy."""

import contextlib
import io
import shutil
import tempfile
import unittest

import numpy as np

from optimize_point_cloud import (
    consolidate_optimized_indices,
    optimize_point_clouds,
)
from examples.gross_budgeting import main as gross_budgeting_example
from pocaduck import Ingestor, Query, StorageConfig
from tests.test_budgeting import VOXEL_SIZE, soma_neurite_fixture


class TestBudgetedQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="pocaduck_budget_query_")
        cls.config = StorageConfig(base_path=cls.temp_dir)
        cls.label = 71
        cls.one_point_label = 72

        points, _ = soma_neurite_fixture()
        cls.expected_points = np.unique(points, axis=0)
        ingestor = Ingestor(cls.config, worker_id="budget")
        ingestor.write(cls.label, "block_0", points[:7_000])
        ingestor.write(cls.label, "block_1", points[7_000:])
        # Exercise the raw layout's query-time deduplication explicitly.
        ingestor.write(cls.label, "block_duplicate", points[[0]])
        ingestor.write(
            cls.one_point_label,
            "block_one",
            np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
        )
        ingestor.finalize()
        cls.raw_index = Ingestor.consolidate_indexes(cls.config)

        optimize_point_clouds(
            cls.config,
            cls.raw_index,
            worker_id="budget",
            verbose=False,
        )
        consolidate_optimized_indices(cls.config, verbose=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def query_for_layout(self, layout, *, cache_size=10):
        if layout == "raw":
            query = Query(
                self.config,
                index_path=self.raw_index,
                cache_size=cache_size,
            )
            self.assertFalse(query.using_optimized_data)
            return query
        query = Query(self.config, cache_size=cache_size)
        self.assertTrue(query.using_optimized_data)
        return query

    def test_budget_none_and_positional_calls_are_unchanged(self):
        for layout in ("raw", "optimized"):
            with self.subTest(layout=layout):
                query = self.query_for_layout(layout)
                try:
                    current = query.get_points(self.label, False, False)
                    explicit_defaults = query.get_points(
                        self.label,
                        False,
                        False,
                        budget=None,
                        strategy=None,
                    )
                    np.testing.assert_array_equal(current, explicit_defaults)
                    np.testing.assert_array_equal(current, self.expected_points)

                    timed_points, timing = query.get_points(self.label, False, True)
                    np.testing.assert_array_equal(timed_points, current)
                    self.assertNotIn("pass1_time", timing)
                    self.assertNotIn("pass2_time", timing)
                finally:
                    query.close()

    def test_both_storage_layouts_return_exact_seeded_budget(self):
        for layout in ("raw", "optimized"):
            with self.subTest(layout=layout):
                query = self.query_for_layout(layout)
                try:
                    first = query.get_points(
                        self.label,
                        budget=512,
                        strategy="gross_budgeting",
                        voxel_size=VOXEL_SIZE,
                        seed=44,
                    )
                    second = query.get_points(
                        self.label,
                        budget=512,
                        strategy="gross_budgeting",
                        voxel_size=VOXEL_SIZE,
                        seed=44,
                    )
                    self.assertEqual(first.shape, (512, self.expected_points.shape[1]))
                    np.testing.assert_array_equal(first, second)
                finally:
                    query.close()

    def test_budgeted_timing_extends_existing_shape(self):
        query = self.query_for_layout("raw", cache_size=0)
        try:
            _, raw_timing = query.get_points(self.label, timing=True)
            points, budgeted_timing = query.get_points(
                self.label,
                timing=True,
                budget=257,
                strategy="gross_budgeting",
                voxel_size=VOXEL_SIZE,
                seed=5,
            )
            self.assertEqual(len(points), 257)
            self.assertEqual(
                set(budgeted_timing),
                set(raw_timing) | {"pass1_time", "pass2_time"},
            )
            self.assertGreaterEqual(budgeted_timing["pass1_time"], 0)
            self.assertGreaterEqual(budgeted_timing["pass2_time"], 0)
            self.assertEqual(budgeted_timing["points_returned"], 257)
        finally:
            query.close()

    def test_cache_contains_only_raw_cloud_and_can_be_disabled(self):
        query = self.query_for_layout("raw")
        try:
            first = query.get_points(
                self.label,
                budget=128,
                strategy="gross_budgeting",
                voxel_size=VOXEL_SIZE,
                seed=1,
            )
            self.assertIn(self.label, query._points_cache)
            self.assertEqual(len(query._points_cache[self.label]), len(self.expected_points))
            second = query.get_points(
                self.label,
                budget=128,
                strategy="gross_budgeting",
                voxel_size=VOXEL_SIZE,
                seed=2,
            )
            self.assertFalse(np.array_equal(first, second))
            self.assertEqual(len(query._points_cache), 1)
        finally:
            query.close()

        no_cache_query = self.query_for_layout("raw")
        try:
            no_cache_query.get_points(
                self.label,
                use_cache=False,
                budget=128,
                strategy="gross_budgeting",
                voxel_size=VOXEL_SIZE,
                seed=1,
            )
            self.assertEqual(no_cache_query._points_cache, {})
        finally:
            no_cache_query.close()

    def test_unknown_one_point_and_larger_than_cloud(self):
        query = self.query_for_layout("raw")
        try:
            missing = query.get_points(
                999_999,
                budget=10,
                strategy="gross_budgeting",
                voxel_size=VOXEL_SIZE,
            )
            self.assertEqual(missing.shape, (0,))

            one = query.get_points(
                self.one_point_label,
                budget=1,
                strategy="gross_budgeting",
                voxel_size=VOXEL_SIZE,
            )
            np.testing.assert_array_equal(one, [[1, 2, 3, 4, 5]])

            raw = query.get_points(self.one_point_label)
            all_points = query.get_points(
                self.one_point_label,
                budget=10,
                strategy="gross_budgeting",
                voxel_size=VOXEL_SIZE,
            )
            np.testing.assert_array_equal(all_points, raw)
        finally:
            query.close()

    def test_invalid_strategy_parameters(self):
        query = self.query_for_layout("raw")
        try:
            calls = [
                lambda: query.get_points(self.label, strategy="gross_budgeting"),
                lambda: query.get_points(self.label, budget=10),
                lambda: query.get_points(
                    self.label, budget=10, strategy="not_a_strategy"
                ),
                lambda: query.get_points(
                    self.label, budget=10, strategy="gross_budgeting"
                ),
                lambda: query.get_points(
                    self.label,
                    budget=0,
                    strategy="gross_budgeting",
                    voxel_size=VOXEL_SIZE,
                ),
                lambda: query.get_points(
                    self.label,
                    budget=-1,
                    strategy="gross_budgeting",
                    voxel_size=VOXEL_SIZE,
                ),
                lambda: query.get_points(
                    self.label,
                    budget=1.5,
                    strategy="gross_budgeting",
                    voxel_size=VOXEL_SIZE,
                ),
                lambda: query.get_points(
                    self.label,
                    budget=10,
                    strategy="gross_budgeting",
                    voxel_size=(4, 4),
                ),
                lambda: query.get_points(
                    self.label,
                    budget=10,
                    strategy="gross_budgeting",
                    voxel_size=VOXEL_SIZE,
                    soma_threshold=1,
                ),
                lambda: query.get_points(
                    self.label,
                    budget=10,
                    strategy="gross_budgeting",
                    voxel_size=VOXEL_SIZE,
                    sampling_fraction=0,
                ),
            ]
            for call in calls:
                with self.subTest(call=call), self.assertRaises(ValueError):
                    call()
        finally:
            query.close()

    def test_worked_example(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            gross_budgeting_example()
        self.assertIn("Retrieved 64 budgeted points with shape (64, 4)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
