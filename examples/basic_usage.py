"""
Basic usage example for PoCADuck.

This example demonstrates the basic workflow of setting up storage,
ingesting point clouds with multiple workers, and querying the results.
"""

import os
import numpy as np
from pathlib import Path
import tempfile
import shutil
from pocaduck import StorageConfig, Ingestor, Query

def main():
    # Set up a temporary directory for this example
    temp_dir = tempfile.mkdtemp(prefix="pocaduck_example_")
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create storage configuration
        config = StorageConfig(base_path=temp_dir)
        
        # Simulate multiple workers processing blocks
        worker_count = 3
        block_count = 4
        label_count = 10
        
        # Generate random labels
        labels = np.random.randint(1, 1000, size=label_count, dtype=np.uint64)
        
        print(f"Simulating {worker_count} workers processing {block_count} blocks with {label_count} labels")
        
        # Each worker processes some blocks
        for worker_id in range(worker_count):
            print(f"Worker {worker_id} starting...")
            
            # Create an ingestor for this worker
            ingestor = Ingestor(
                storage_config=config, 
                worker_id=worker_id,
                max_points_per_file=100000
            )
            
            # Process blocks assigned to this worker
            for block_idx in range(block_count):
                block_id = f"block_{block_idx}"
                
                # For each block, generate point clouds for some of the labels
                for label in labels:
                    # Randomly decide whether this label appears in this block
                    if np.random.random() < 0.6:  # 60% chance
                        # Generate random number of points
                        num_points = np.random.randint(100, 1000)
                        
                        # Generate random 3D points with supervoxel IDs
                        # First 3 columns are x, y, z coordinates, 4th column is supervoxel ID
                        coordinates = np.random.randint(0, 1000, size=(num_points, 3))
                        # Generate large random supervoxel IDs
                        supervoxels = np.random.randint(
                            low=1000000000, 
                            high=9000000000, 
                            size=(num_points, 1), 
                            dtype=np.int64
                        )
                        # Combine coordinates and supervoxels
                        points = np.hstack((coordinates, supervoxels))
                        
                        # Write the points
                        ingestor.write(
                            label=int(label), 
                            block_id=block_id, 
                            points=points
                        )
                        
                        print(f"  Worker {worker_id} wrote {num_points} points for label {label} in {block_id}")
            
            # Finalize this worker's ingestion
            ingestor.finalize()
            print(f"Worker {worker_id} completed.")
        
        print("\nAll workers completed. Consolidating indexes...")
        
        # Consolidate all worker indexes
        unified_index_path = Ingestor.consolidate_indexes(config)
        print(f"Unified index created at: {unified_index_path}")
        
        print("\nQuerying results...")
        
        # Create a query object
        query = Query(storage_config=config)
        
        # Get all available labels
        available_labels = query.get_labels()
        print(f"Available labels: {available_labels}")
        
        # Pick a label to query
        if len(available_labels) > 0:
            test_label = available_labels[0]
            
            # Get blocks for this label
            blocks = query.get_blocks_for_label(test_label)
            print(f"\nLabel {test_label} appears in {len(blocks)} blocks: {blocks}")
            
            # Get point count
            point_count = query.get_point_count(test_label)
            print(f"Label {test_label} has {point_count} points total")
            
            # Get all points
            points = query.get_points(test_label)
            print(f"Retrieved {points.shape[0]} points for label {test_label}")
            print(f"Point data shape: {points.shape} (each point has {points.shape[1]} dimensions)")
            print(f"First few points (x, y, z, supervoxel_id): \n{points[:5]}")
            
            # Display coordiantes and supervoxel IDs separately for clarity
            if points.shape[0] > 0 and points.shape[1] >= 4:
                print(f"\nCoordinates (x, y, z):\n{points[:5, :3]}")
                print(f"\nSupervoxel IDs:\n{points[:5, 3]}")
        
        # Close the query connection
        query.close()
        
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()