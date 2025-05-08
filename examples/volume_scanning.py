"""
Volume scanning example for PoCADuck.

This example demonstrates how PoCADuck can be used with a system that scans 3D labeled volumes
and generates point clouds for each label in a blockwise fashion.
"""

import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from pocaduck import StorageConfig, Ingestor, Query

class MockVolumeBlock:
    """Mock implementation of a 3D labeled volume block."""
    
    def __init__(self, block_id, shape=(64, 64, 64), max_labels=10):
        """Initialize a mock volume block with random labels."""
        self.block_id = block_id
        self.shape = shape
        
        # Generate a random labeled volume with integers (simulating segment IDs)
        num_labels = np.random.randint(1, max_labels + 1)
        labels = np.random.randint(1, 1000, size=num_labels, dtype=np.uint64)
        
        # Create a 3D volume with these labels
        self.volume = np.zeros(shape, dtype=np.uint64)
        
        # Fill the volume with the labels
        for label in labels:
            # Add random spots of each label
            num_spots = np.random.randint(1, 5)
            for _ in range(num_spots):
                # Random spot center
                center = [np.random.randint(0, s) for s in shape]
                # Random spot radius
                radius = np.random.randint(3, 10)
                
                # Add the spot
                for x in range(max(0, center[0] - radius), min(shape[0], center[0] + radius)):
                    for y in range(max(0, center[1] - radius), min(shape[1], center[1] + radius)):
                        for z in range(max(0, center[2] - radius), min(shape[2], center[2] + radius)):
                            if np.sum(((x - center[0])**2, (y - center[1])**2, (z - center[2])**2)) <= radius**2:
                                self.volume[x, y, z] = label
    
    def get_labels(self):
        """Get unique labels in this block."""
        return np.unique(self.volume[self.volume > 0])
    
    def get_point_cloud(self, label):
        """Get the point cloud for a specific label in this block."""
        # Find coordinates where the volume equals the label
        coords = np.where(self.volume == label)
        if len(coords[0]) == 0:
            return np.zeros((0, 4), dtype=np.int64)  # Return empty array with 4 columns
        
        # Stack the coordinates into an (N, 3) array
        coords_array = np.vstack(coords).T.astype(np.int64)
        
        # Generate supervoxel IDs for each point (simulating additional data)
        num_points = coords_array.shape[0]
        supervoxel_ids = np.random.randint(
            low=10000000000, 
            high=90000000000, 
            size=(num_points, 1), 
            dtype=np.int64
        )
        
        # Combine coordinates and supervoxel IDs
        points = np.hstack((coords_array, supervoxel_ids))
        return points


def process_block(block, worker_id, storage_path):
    """Process a single block and store point clouds for each label."""
    print(f"Worker {worker_id} processing block {block.block_id}")
    
    # Create a storage configuration
    config = StorageConfig(base_path=storage_path)
    
    # Create an ingestor for this worker
    ingestor = Ingestor(
        storage_config=config,
        worker_id=worker_id
    )
    
    # Process all labels in this block
    labels = block.get_labels()
    for label in labels:
        # Get point cloud for this label
        points = block.get_point_cloud(label)
        if points.shape[0] > 0:
            # Write the point cloud
            ingestor.write(
                label=int(label),
                block_id=block.block_id,
                points=points
            )
            print(f"  Worker {worker_id} wrote {points.shape[0]} points for label {label} in {block.block_id}")
    
    # Finalize this ingestor (closes connections)
    ingestor.finalize()


def simulate_volume_scanning(num_workers=4, grid_size=4, storage_path=None):
    """
    Simulate scanning a 3D volume broken into blocks, processing blocks in parallel.
    
    Args:
        num_workers: Number of parallel workers.
        grid_size: Size of the grid in each dimension (grid_size^3 total blocks).
        storage_path: Path to store the data. If None, uses a temporary directory.
    """
    if storage_path is None:
        import tempfile
        storage_path = tempfile.mkdtemp(prefix="pocaduck_volume_scan_")
        print(f"Using temporary storage path: {storage_path}")
    
    # Create blocks
    blocks = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                block_id = f"block_x{x}_y{y}_z{z}"
                blocks.append(MockVolumeBlock(block_id=block_id))
    
    print(f"Created {len(blocks)} blocks in a {grid_size}x{grid_size}x{grid_size} grid")
    
    # Process blocks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Distribute blocks among workers
        futures = []
        for i, block in enumerate(blocks):
            worker_id = i % num_workers
            future = executor.submit(process_block, block, worker_id, storage_path)
            futures.append(future)
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()
    
    print("All blocks processed. Consolidating indexes...")
    
    # Consolidate worker indexes
    config = StorageConfig(base_path=storage_path)
    unified_index_path = Ingestor.consolidate_indexes(config)
    
    print(f"Unified index created at: {unified_index_path}")
    return storage_path


def query_results(storage_path):
    """Query and display results from the processed volume."""
    print("\nQuerying results...")
    
    # Create a storage configuration
    config = StorageConfig(base_path=storage_path)
    
    # Create a query object
    query = Query(storage_config=config)
    
    # Get all available labels
    available_labels = query.get_labels()
    print(f"Found {len(available_labels)} unique labels in the volume")
    
    if len(available_labels) > 0:
        # Display information for a few labels
        for label in available_labels[:5]:  # First 5 labels
            point_count = query.get_point_count(label)
            blocks = query.get_blocks_for_label(label)
            print(f"\nLabel {label}:")
            print(f"  Present in {len(blocks)} blocks")
            print(f"  Total points: {point_count}")
            
            # Get the full point cloud for this label
            points = query.get_points(label)
            print(f"  Retrieved {points.shape[0]} points")
            
            # Calculate bounding box of coordinates and show supervoxel data
            if points.shape[0] > 0:
                # Get coordinate dimensions (first 3 columns)
                coords = points[:, :3]
                min_coords = np.min(coords, axis=0)
                max_coords = np.max(coords, axis=0)
                print(f"  Bounding box (x,y,z): min={min_coords}, max={max_coords}")
                
                # Display supervoxel information
                supervoxels = points[:, 3]
                unique_supervoxels = np.unique(supervoxels)
                print(f"  Contains {len(unique_supervoxels)} unique supervoxel IDs")
                print(f"  Sample supervoxel IDs: {unique_supervoxels[:5] if len(unique_supervoxels) > 5 else unique_supervoxels}")
    
    # Close the query connection
    query.close()


def main():
    """Main function to demonstrate volume scanning and querying."""
    print("PoCADuck Volume Scanning Example")
    print("================================")
    
    # Simulate volume scanning
    storage_path = simulate_volume_scanning(num_workers=2, grid_size=3)
    
    # Query and display results
    query_results(storage_path)
    
    print("\nVolume scanning example complete.")
    
    # Cleanup
    # Uncomment this to remove temporary storage
    # import shutil
    # shutil.rmtree(storage_path)


if __name__ == "__main__":
    main()