"""
Cloud storage example for PoCADuck.

This example demonstrates how to configure PoCADuck for different cloud storage backends.
Note: You'll need appropriate credentials and access to run this example.
"""

from pocaduck import StorageConfig, Ingestor, Query
import numpy as np

def s3_example():
    """Example using AWS S3 storage."""
    print("Setting up S3 configuration...")
    
    # Configure S3 storage
    s3_config = StorageConfig(
        base_path="s3://your-bucket-name/pocaduck-data",
        s3_region="us-west-2",
        s3_access_key_id="YOUR_ACCESS_KEY",  # Replace with actual credentials
        s3_secret_access_key="YOUR_SECRET_KEY"  # Replace with actual credentials
    )
    
    # From here, usage is the same as with local storage
    # Create ingestors with this config, write data, consolidate indexes, and query
    
    print("S3 configuration ready. Use this config with Ingestor and Query classes.")
    return s3_config

def gcs_example():
    """Example using Google Cloud Storage."""
    print("Setting up GCS configuration...")
    
    # Configure GCS storage with a credentials file
    gcs_config = StorageConfig(
        base_path="gs://your-bucket-name/pocaduck-data",
        gcs_project_id="your-project-id",
        gcs_credentials="/path/to/credentials.json"  # Path to credentials file
    )
    
    # Alternatively, you can use a credentials dictionary
    # gcs_config = StorageConfig(
    #     base_path="gs://your-bucket-name/pocaduck-data",
    #     gcs_project_id="your-project-id",
    #     gcs_credentials={
    #         "type": "service_account",
    #         "project_id": "your-project-id",
    #         # Add other credentials data here
    #     }
    # )
    
    print("GCS configuration ready. Use this config with Ingestor and Query classes.")
    return gcs_config

def azure_example():
    """Example using Azure Blob Storage."""
    print("Setting up Azure configuration...")
    
    # Configure Azure storage
    azure_config = StorageConfig(
        base_path="azure://your-container/pocaduck-data",
        azure_storage_connection_string="YOUR_CONNECTION_STRING"  # Replace with actual connection string
    )
    
    print("Azure configuration ready. Use this config with Ingestor and Query classes.")
    return azure_config

def demonstrate_ingestion(config, worker_id="worker1"):
    """Demonstrate ingestion with a given configuration."""
    print(f"\nSimulating ingestion with worker {worker_id}...")
    
    # Create an ingestor
    ingestor = Ingestor(
        storage_config=config,
        worker_id=worker_id
    )
    
    # Generate some sample data with supervoxel IDs
    label = 12345
    block_id = "block_1"
    
    # Generate coordinates (first 3 columns)
    coordinates = np.random.randint(0, 10000, size=(100, 3))
    
    # Generate supervoxel IDs (4th column)
    supervoxels = np.random.randint(
        low=10000000000, 
        high=90000000000, 
        size=(100, 1), 
        dtype=np.int64
    )
    
    # Combine coordinates and supervoxels
    points = np.hstack((coordinates, supervoxels))
    
    print(f"Writing {points.shape[0]} points with {points.shape[1]} dimensions for label {label} in block {block_id}")
    print(f"Sample point (x, y, z, supervoxel_id): {points[0]}")
    # Note: This would actually write to cloud storage if credentials were valid
    # ingestor.write(label=label, block_id=block_id, points=points)
    
    print("Finalizing ingestion...")
    # ingestor.finalize()
    
    print("Demonstrating index consolidation (not actually executing)...")
    # Ingestor.consolidate_indexes(config)
    
    print("Ingestion demonstration complete.")

def main():
    print("PoCADuck Cloud Storage Configuration Examples")
    print("=============================================")
    print("NOTE: These examples show configuration only and don't execute actual cloud operations.\n")
    
    # Get configurations for different cloud providers
    s3_config = s3_example()
    gcs_config = gcs_example()
    azure_config = azure_example()
    
    # Demonstrate ingestion with S3 (but don't actually execute cloud operations)
    demonstrate_ingestion(s3_config)
    
    print("\nCloud storage configuration examples complete.")

if __name__ == "__main__":
    main()