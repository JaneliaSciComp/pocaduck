"""
Tests for PoCADuck example code.

These tests verify that the example code functions correctly.
"""

import unittest
import tempfile
import shutil
import os
import sys
import numpy as np
from pathlib import Path
import importlib.util

# Import the PoCADuck modules
from pocaduck import StorageConfig, Ingestor, Query


class TestExamples(unittest.TestCase):
    """Test PoCADuck example code."""
    
    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="pocaduck_example_test_")
        self.original_argv = sys.argv
    
    def tearDown(self):
        """Clean up temporary directory after tests."""
        try:
            shutil.rmtree(self.temp_dir)
        except (OSError, FileNotFoundError):
            pass  # If cleanup failed earlier, ignore
        sys.argv = self.original_argv
    
    def test_basic_usage(self):
        """Test the basic_usage.py example."""
        # Set up a storage configuration directly
        config = StorageConfig(base_path=self.temp_dir)
        
        # Create a few test workers
        for worker_id in range(2):
            ingestor = Ingestor(storage_config=config, worker_id=worker_id)
            
            # Write some test data
            for block_id in [f"block_{i}" for i in range(2)]:
                for label in [10, 20, 30]:
                    # Generate some random points
                    num_points = np.random.randint(50, 100)
                    points = np.random.rand(num_points, 3) * 100
                    
                    # Write the points
                    ingestor.write(label=label, block_id=block_id, points=points)
            
            # Finalize
            ingestor.finalize()
        
        # Consolidate indexes
        unified_index_path = Ingestor.consolidate_indexes(config)
        
        # Verify that the unified index was created
        self.assertTrue(os.path.exists(unified_index_path), 
                        f"Unified index not found at {unified_index_path}")
        
        # Create a query to verify data was written
        query = Query(storage_config=config)
        
        # Verify that we have the expected labels
        labels = query.get_labels()
        self.assertEqual(set(labels), {10, 20, 30})
        
        # Verify we can retrieve points for each label
        for label in labels:
            points = query.get_points(label)
            self.assertTrue(points.shape[0] > 0, f"No points found for label {label}")
            
            # Get blocks
            blocks = query.get_blocks_for_label(label)
            self.assertTrue(len(blocks) > 0, f"No blocks found for label {label}")
        
        # Close the query
        query.close()
    
    def test_volume_scanning_simple(self):
        """Test a simplified version of the volume scanning example."""
        # Create a simple test without multiprocessing
        config = StorageConfig(base_path=self.temp_dir)
        
        # Create a few mock blocks
        blocks = []
        for i in range(2):
            for j in range(2):
                block_id = f"block_x{i}_y{j}_z0"
                # Create a simple array for the block
                shape = (10, 10, 10)
                volume = np.zeros(shape, dtype=np.uint64)
                
                # Add a few labels
                labels = [10, 20, 30]
                for idx, label in enumerate(labels):
                    x, y, z = 3 + idx, 4 + idx, 5 + idx
                    volume[x:x+3, y:y+3, z:z+3] = label
                
                blocks.append((block_id, volume, labels))
        
        # Process blocks directly (without multiprocessing)
        for worker_id, (block_id, volume, labels) in enumerate(blocks):
            # Create an ingestor
            ingestor = Ingestor(storage_config=config, worker_id=worker_id)
            
            # Process each label
            for label in labels:
                # Find coordinates where the volume equals the label
                coords = np.where(volume == label)
                if len(coords[0]) == 0:
                    continue
                
                # Stack the coordinates into an (N, 3) array
                points = np.vstack(coords).T.astype(np.float32)
                
                # Write the points
                ingestor.write(label=label, block_id=block_id, points=points)
            
            # Finalize
            ingestor.finalize()
        
        # Consolidate indexes
        unified_index_path = Ingestor.consolidate_indexes(config)
        
        # Check that the unified index was created
        self.assertTrue(os.path.exists(unified_index_path))
        
        # Create a query to verify data was written
        query = Query(storage_config=config)
        
        # Verify that we have the expected labels
        labels = query.get_labels()
        self.assertEqual(set(labels), {10, 20, 30})
        
        # Verify we can retrieve points for each label
        for label in labels:
            points = query.get_points(label)
            self.assertTrue(points.shape[0] > 0, f"No points found for label {label}")
        
        # Clean up
        query.close()


if __name__ == "__main__":
    unittest.main()