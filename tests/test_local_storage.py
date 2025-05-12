"""
Tests for PoCADuck with local filesystem storage.

These tests verify the basic functionality of PoCADuck using a local filesystem.
"""

import unittest
import tempfile
import shutil
import os
import numpy as np
from pathlib import Path

# Import the PoCADuck modules
from pocaduck import StorageConfig, Ingestor, Query


class TestLocalStorage(unittest.TestCase):
    """Test PoCADuck functionality with local filesystem storage."""
    
    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="pocaduck_test_")
        self.storage_config = StorageConfig(base_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_storage_config(self):
        """Test the StorageConfig class with local storage."""
        config = self.storage_config
        self.assertEqual(config.storage_type, "local")
        self.assertEqual(config.base_path, self.temp_dir)
        
        # Check that DuckDB configuration is empty for local storage
        duckdb_config = config.get_duckdb_config()
        self.assertEqual(duckdb_config, {})
    
    def test_ingestor_creation(self):
        """Test creating an Ingestor instance."""
        ingestor = Ingestor(storage_config=self.storage_config, worker_id="test_worker", verbose=True)
        
        # Check that directories were created
        worker_dir = os.path.join(self.temp_dir, "worker_test_worker")
        data_dir = os.path.join(worker_dir, "data")
        self.assertTrue(os.path.exists(worker_dir))
        self.assertTrue(os.path.exists(data_dir))
        
        # Clean up
        ingestor.finalize()
    
    def test_write_and_query_single_block(self):
        """Test writing and querying a point cloud in a single block."""
        # Create test data
        label = 42
        block_id = "test_block"
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        
        # Create an ingestor and write the points
        ingestor = Ingestor(storage_config=self.storage_config, worker_id="test_worker", verbose=True)
        ingestor.write(label=label, block_id=block_id, points=points)
        ingestor.finalize()
        
        # Consolidate indexes
        Ingestor.consolidate_indexes(self.storage_config)
        
        # Create a query and retrieve the points
        query = Query(storage_config=self.storage_config)
        
        # Check that the label exists
        labels = query.get_labels()
        self.assertIn(label, labels)
        
        # Check block information
        blocks = query.get_blocks_for_label(label)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], block_id)
        
        # Check point count
        point_count = query.get_point_count(label)
        self.assertEqual(point_count, points.shape[0])
        
        # Retrieve and check points
        retrieved_points = query.get_points(label)
        self.assertEqual(retrieved_points.shape, points.shape)
        self.assertTrue(np.allclose(retrieved_points, points))
        
        # Clean up
        query.close()
    
    def test_multiple_workers(self):
        """Test writing with multiple workers and querying the results."""
        # Create test data
        label = 100
        num_workers = 3
        points_per_worker = [
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
            np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ]
        
        # Create ingestors and write points
        for worker_id in range(num_workers):
            ingestor = Ingestor(
                storage_config=self.storage_config,
                worker_id=f"worker_{worker_id}",
                verbose=True
            )
            
            block_id = f"block_{worker_id}"
            points = points_per_worker[worker_id]
            
            ingestor.write(label=label, block_id=block_id, points=points)
            ingestor.finalize()
        
        # Consolidate indexes
        Ingestor.consolidate_indexes(self.storage_config)
        
        # Create a query and retrieve the points
        query = Query(storage_config=self.storage_config)
        
        # Check that the label exists
        labels = query.get_labels()
        self.assertIn(label, labels)
        
        # Check block information
        blocks = query.get_blocks_for_label(label)
        self.assertEqual(len(blocks), num_workers)
        
        # Check point count
        expected_total_points = sum(p.shape[0] for p in points_per_worker)
        point_count = query.get_point_count(label)
        self.assertEqual(point_count, expected_total_points)
        
        # Retrieve and check points
        retrieved_points = query.get_points(label)
        self.assertEqual(retrieved_points.shape[0], expected_total_points)
        
        # Clean up
        query.close()
    
    def test_multiple_labels_per_block(self):
        """Test writing multiple labels within a single block."""
        # Create test data
        block_id = "multi_label_block"
        labels_and_points = {
            1: np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            2: np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
            3: np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        }
        
        # Create an ingestor and write the points
        ingestor = Ingestor(storage_config=self.storage_config, worker_id="test_worker")
        
        for label, points in labels_and_points.items():
            ingestor.write(label=label, block_id=block_id, points=points)
        
        ingestor.finalize()
        
        # Consolidate indexes
        Ingestor.consolidate_indexes(self.storage_config)
        
        # Create a query and retrieve the points
        query = Query(storage_config=self.storage_config)
        
        # Check that all labels exist
        labels = query.get_labels()
        for label in labels_and_points.keys():
            self.assertIn(label, labels)
        
        # Check each label's points
        for label, points in labels_and_points.items():
            retrieved_points = query.get_points(label)
            self.assertEqual(retrieved_points.shape, points.shape)
            self.assertTrue(np.allclose(retrieved_points, points))
        
        # Clean up
        query.close()
    
    def test_points_with_supervoxel(self):
        """Test writing and querying points with x, y, z coordinates plus supervoxel ID."""
        # Create test data with supervoxel column
        label = 200
        block_id = "supervoxel_block"
        
        # Points with 4 columns: x, y, z, supervoxel_id
        # Using large integer values for coordinates and supervoxel IDs
        points = np.array([
            [1000, 2000, 3000, 9223372036854775000],  # Large int64 value for supervoxel
            [4000, 5000, 6000, 9223372036854775001],
            [7000, 8000, 9000, 9223372036854775002],
            [10000, 11000, 12000, 9223372036854775003]
        ], dtype=np.int64)
        
        # Create an ingestor and write the points
        ingestor = Ingestor(storage_config=self.storage_config, worker_id="test_worker", verbose=True)
        ingestor.write(label=label, block_id=block_id, points=points)
        ingestor.finalize()
        
        # Consolidate indexes
        Ingestor.consolidate_indexes(self.storage_config)
        
        # Create a query and retrieve the points
        query = Query(storage_config=self.storage_config)
        
        # Retrieve and check points
        retrieved_points = query.get_points(label)
        
        # Check shape and values
        self.assertEqual(retrieved_points.shape, points.shape)
        
        # Verify that the original data type is preserved (int64)
        self.assertEqual(retrieved_points.dtype, np.int64)
        
        # Check that all values match exactly, including the large supervoxel IDs
        np.testing.assert_array_equal(retrieved_points, points)
        
        # Clean up
        query.close()


if __name__ == "__main__":
    unittest.main()