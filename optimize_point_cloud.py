#!/usr/bin/env python
"""
PoCADuck Point Cloud Optimizer

This script reorganizes point cloud data to optimize label-based queries.
It groups points for each label into contiguous regions in optimized parquet files,
significantly improving query performance.

The optimizer can be run in parallel on different subsets of labels, with a final
consolidation step to merge the results.

The optimization results are stored in the following structure:
- {base_path}/optimized/                        # Main optimization directory
  - optimized_index.db                          # Consolidated optimized index
  - optimize_{worker_id}/                       # Worker-specific directories
    - metadata.json                             # Worker metadata
    - optimized_{uuid}.parquet                  # Optimized parquet files
"""

import os
import sys
import time
import argparse
import uuid
import json
from typing import List, Optional, Set, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
from tqdm import tqdm

from pocaduck.storage_config import StorageConfig

def optimize_point_clouds(
    storage_config: StorageConfig,
    source_index_path: str,
    target_path: Optional[str] = None,
    worker_id: str = "default",
    target_file_size: int = 500 * 1024 * 1024,  # 500MB default target size
    batch_size: int = 100,  # Process this many labels at once
    labels_to_process: Optional[List[int]] = None,
    threads: Optional[int] = None,
    verbose: bool = True
) -> str:
    """
    Reorganize point cloud data to optimize label-based queries.
    
    Args:
        storage_config: Storage configuration for source and target.
        source_index_path: Path to the existing unified index.
        target_path: Path where optimized files will be stored. If None, uses a
                    subdirectory "optimized" under the storage_config base_path.
        worker_id: Unique identifier for this optimization worker.
        target_file_size: Target size for optimized parquet files in bytes.
        batch_size: Number of labels to process in a single batch to manage memory.
        labels_to_process: Optional list of specific labels to process. If None, all labels are processed.
        threads: Number of threads to use for DuckDB processing. If None, uses system default.
        verbose: Whether to print progress information.
    
    Returns:
        Path to the worker's metadata file containing information about processed labels.
    """
    # Set up paths
    if target_path is None:
        target_path = os.path.join(storage_config.base_path, "optimized")
    
    # Create optimized directory and optimizer worker subdirectory
    # Use 'optimize_' prefix to distinguish from ingestion workers
    worker_dir = os.path.join(target_path, f"optimize_{worker_id}")
    os.makedirs(worker_dir, exist_ok=True)
    
    # Path for worker metadata
    worker_metadata_path = os.path.join(worker_dir, "metadata.json")
    
    # Check if this worker has already done some work
    processed_labels = set()
    if os.path.exists(worker_metadata_path):
        try:
            with open(worker_metadata_path, 'r') as f:
                metadata = json.load(f)
                processed_info = metadata.get("processed_labels", {})
                processed_labels = set(int(label) for label in processed_info.keys())
                if verbose:
                    print(f"Found {len(processed_labels)} previously processed labels for worker {worker_id}")
        except Exception as e:
            if verbose:
                print(f"Error reading metadata file: {e}")
    
    # Connect to source index
    src_con = duckdb.connect(source_index_path, read_only=True)
    
    # Apply storage configuration
    duckdb_config = storage_config.get_duckdb_config()
    for key, value in duckdb_config.items():
        src_con.execute(f"SET {key}='{value}'")
    
    # Set threads if specified
    if threads is not None:
        src_con.execute(f"PRAGMA threads={threads}")
    
    # Get all unique labels or use provided labels
    if labels_to_process is None:
        if verbose:
            print("Fetching all unique labels from the index...")
        labels = src_con.execute("SELECT DISTINCT label FROM point_cloud_index ORDER BY label").fetchall()
        labels = [label[0] for label in labels]
        if verbose:
            print(f"Found {len(labels)} unique labels")
    else:
        labels = labels_to_process
        if verbose:
            print(f"Processing {len(labels)} specified labels")
    
    # Filter out already processed labels
    if processed_labels:
        original_count = len(labels)
        labels = [label for label in labels if label not in processed_labels]
        if verbose:
            print(f"Filtered out {original_count - len(labels)} already processed labels")
    
    # Load or initialize worker metadata
    worker_metadata = {
        "worker_id": worker_id,
        "processed_labels": {},
        "files": []
    }
    
    if os.path.exists(worker_metadata_path):
        try:
            with open(worker_metadata_path, 'r') as f:
                worker_metadata = json.load(f)
        except Exception as e:
            if verbose:
                print(f"Error loading existing metadata, starting fresh: {e}")
    
    # Process labels in batches to manage memory
    current_file_path = None
    current_file_size = 0
    processed_count = 0
    total_points = 0
    start_time = time.time()
    
    if verbose:
        print(f"Processing {len(labels)} labels in batches of {batch_size}...")
    
    for i in range(0, len(labels), batch_size):
        batch_labels = labels[i:i+batch_size]
        if verbose:
            print(f"Processing batch {i//batch_size + 1}/{(len(labels) + batch_size - 1)//batch_size}: "
                  f"{len(batch_labels)} labels")
        
        batch_progress = tqdm(batch_labels, desc="Labels", unit="label") if verbose else batch_labels
        for label in batch_progress:
            # Get file paths containing this label
            file_info = src_con.execute(
                "SELECT file_path FROM point_cloud_index WHERE label = ?",
                [label]
            ).fetchall()
            
            file_paths = [info[0] for info in file_info]
            
            if not file_paths:
                if verbose:
                    print(f"Skipping label {label}: no data found")
                continue
            
            # Use DuckDB to efficiently read the point data
            query = f"""
                SELECT data
                FROM parquet_scan([{','.join(f"'{path}'" for path in file_paths)}])
                WHERE label = {label}
            """
            
            try:
                # Get points as DataFrame
                df = src_con.execute(query).fetchdf()
                
                if len(df) == 0:
                    if verbose:
                        print(f"Skipping label {label}: no point data returned")
                    continue
                
                # Stack points and remove duplicates
                points_list = df['data'].tolist()
                points = np.vstack(points_list).astype(np.int64)
                points = np.unique(points, axis=0)
                
                # Calculate size of this label's points
                # Rough estimate: points array size + overhead
                point_size_bytes = points.nbytes + 1000  # Add overhead
                
                # If adding this would exceed target file size, create a new file
                if current_file_path is None or current_file_size + point_size_bytes > target_file_size:
                    # Create new file
                    file_id = str(uuid.uuid4())
                    current_file_path = os.path.join(worker_dir, f"optimized_{file_id}.parquet")
                    current_file_size = 0
                    
                    # Add file to metadata
                    if current_file_path not in worker_metadata["files"]:
                        worker_metadata["files"].append(current_file_path)
                
                # Prepare data for writing
                point_df = pd.DataFrame({
                    'label': label,
                    'data': [points]  # Store as a single array
                })
                
                # Write to file
                if current_file_size == 0:
                    # New file
                    point_df.to_parquet(current_file_path, index=False)
                else:
                    # Append to existing file
                    point_df.to_parquet(current_file_path, index=False, append=True)
                
                # Update size tracking
                new_size = os.path.getsize(current_file_path)
                point_size_actual = new_size - current_file_size
                current_file_size = new_size
                
                # Add label info to metadata
                worker_metadata["processed_labels"][str(label)] = {
                    "file_path": current_file_path,
                    "point_count": int(len(points))
                }
                
                # Save metadata periodically (every 10 labels)
                if processed_count % 10 == 0:
                    with open(worker_metadata_path, 'w') as f:
                        json.dump(worker_metadata, f, indent=2)
                
                processed_count += 1
                total_points += len(points)
                
            except Exception as e:
                if verbose:
                    print(f"Error processing label {label}: {str(e)}")
    
    # Save final metadata
    with open(worker_metadata_path, 'w') as f:
        json.dump(worker_metadata, f, indent=2)
    
    # Close connection
    src_con.close()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if verbose:
        print("\nWorker optimization complete!")
        print(f"- Worker ID: {worker_id}")
        print(f"- Processed: {processed_count} labels")
        print(f"- Total points: {total_points:,}")
        print(f"- Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"- Worker metadata: {worker_metadata_path}")
        if processed_count > 0:
            print(f"- Points per second: {total_points/elapsed:,.2f}")
    
    return worker_metadata_path

def consolidate_optimized_indices(
    storage_config: StorageConfig,
    target_path: Optional[str] = None,
    target_index_path: Optional[str] = None,
    threads: Optional[int] = None,
    verbose: bool = True
) -> None:
    """
    Consolidate optimized indices from multiple workers into a single optimized index.
    
    Args:
        storage_config: Storage configuration.
        target_path: Path where optimized files are stored. If None, uses a
                    subdirectory "optimized" under the storage_config base_path.
        target_index_path: Path for the consolidated index. If None, uses
                          "{target_path}/optimized_index.db".
        threads: Number of threads to use for DuckDB processing. If None, uses system default.
        verbose: Whether to print progress information.
    """
    # Set up paths
    if target_path is None:
        target_path = os.path.join(storage_config.base_path, "optimized")
    
    if target_index_path is None:
        target_index_path = os.path.join(target_path, "optimized_index.db")
    
    if verbose:
        print(f"Consolidating optimized indices from {target_path}")
        print(f"Target index: {target_index_path}")
    
    # Find all optimizer worker directories
    worker_dirs = [d for d in os.listdir(target_path)
                   if os.path.isdir(os.path.join(target_path, d)) and d.startswith("optimize_")]
    
    if not worker_dirs:
        if verbose:
            print("No worker directories found. Nothing to consolidate.")
        return
    
    if verbose:
        print(f"Found {len(worker_dirs)} worker directories")
    
    # Create/connect to target index
    if os.path.exists(target_index_path):
        if verbose:
            print(f"Removing existing index at {target_index_path}")
        os.remove(target_index_path)
    
    target_con = duckdb.connect(target_index_path)
    
    # Apply storage configuration
    duckdb_config = storage_config.get_duckdb_config()
    for key, value in duckdb_config.items():
        target_con.execute(f"SET {key}='{value}'")
    
    # Set threads if specified
    if threads is not None:
        target_con.execute(f"PRAGMA threads={threads}")
    
    # Create the new optimized schema
    target_con.execute("""
        CREATE TABLE point_cloud_index (
            label BIGINT,
            file_path VARCHAR,
            point_count BIGINT
        )
    """)
    
    # Process each worker's metadata
    total_labels = 0
    for worker_dir in sorted(worker_dirs):
        worker_path = os.path.join(target_path, worker_dir)
        metadata_path = os.path.join(worker_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            if verbose:
                print(f"Skipping {worker_dir}: No metadata file found")
            continue
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            worker_id = metadata.get("worker_id", worker_dir.replace("optimize_", ""))
            processed_labels = metadata.get("processed_labels", {})
            
            if verbose:
                print(f"Processing {worker_dir} (worker_id: {worker_id}): {len(processed_labels)} labels")
            
            # Insert worker's labels into the consolidated index
            for label_str, info in processed_labels.items():
                label = int(label_str)
                file_path = info["file_path"]
                point_count = info["point_count"]
                
                target_con.execute("""
                    INSERT INTO point_cloud_index 
                    (label, file_path, point_count) 
                    VALUES (?, ?, ?)
                """, [label, file_path, point_count])
            
            total_labels += len(processed_labels)
            
        except Exception as e:
            if verbose:
                print(f"Error processing {worker_dir}: {str(e)}")
    
    # Create indexes for performance
    if verbose:
        print("Creating index on label column...")
    target_con.execute("CREATE INDEX idx_label ON point_cloud_index(label)")
    
    # Commit and close connection
    target_con.close()
    
    if verbose:
        print(f"Consolidation complete! Total labels in optimized index: {total_labels}")

def shard_labels(
    storage_config: StorageConfig,
    source_index_path: str,
    num_shards: int,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> List[str]:
    """
    Shard labels into multiple files for parallel processing.
    
    Args:
        storage_config: Storage configuration.
        source_index_path: Path to the source index.
        num_shards: Number of shards to create.
        output_dir: Directory to write shard files. If None, uses current directory.
        verbose: Whether to print progress information.
    
    Returns:
        List of paths to the shard files.
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Connect to source index
    src_con = duckdb.connect(source_index_path, read_only=True)
    
    # Apply storage configuration
    duckdb_config = storage_config.get_duckdb_config()
    for key, value in duckdb_config.items():
        src_con.execute(f"SET {key}='{value}'")
    
    # Get all labels
    if verbose:
        print("Fetching all labels...")
    
    labels = src_con.execute("SELECT DISTINCT label FROM point_cloud_index ORDER BY label").fetchall()
    labels = [label[0] for label in labels]
    
    src_con.close()
    
    if verbose:
        print(f"Found {len(labels)} labels")
    
    # Create shards
    shard_size = (len(labels) + num_shards - 1) // num_shards  # Ceiling division
    shards = [labels[i:i+shard_size] for i in range(0, len(labels), shard_size)]
    
    if verbose:
        print(f"Created {len(shards)} shards with ~{shard_size} labels each")
    
    # Write shards to files
    shard_files = []
    for i, shard in enumerate(shards):
        shard_file = os.path.join(output_dir, f"labels_shard_{i}.txt")
        with open(shard_file, 'w') as f:
            f.write("\n".join(str(label) for label in shard))
        
        shard_files.append(shard_file)
        
        if verbose:
            print(f"Wrote {len(shard)} labels to {shard_file}")
    
    return shard_files

def main():
    """Command-line entry point for the optimization script."""
    parser = argparse.ArgumentParser(description="PoCADuck Point Cloud Optimizer")
    
    parser.add_argument("--source-index", type=str,
                        help="Path to source index (if different from base_path/unified_index.db)")
    parser.add_argument("--target-path", type=str,
                        help="Path where optimized files will be stored (default: base_path/optimized)")
    parser.add_argument("--target-index", type=str,
                        help="Path for the new optimized index (default: target_path/optimized_index.db)")
    parser.add_argument("--target-file-size", type=int, default=500 * 1024 * 1024,
                        help="Target size for optimized parquet files in bytes (default: 500MB)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of labels to process in a single batch (default: 100)")
    parser.add_argument("--threads", type=int,
                        help="Number of threads to use for DuckDB processing (default: system default)")
    parser.add_argument("--worker-id", type=str, default=str(uuid.uuid4())[:8],
                        help="Unique identifier for this optimization worker (default: random UUID)")
    
    # Command selection arguments
    parser.add_argument("--action", type=str, required=True, 
                        choices=["optimize", "consolidate", "shard"],
                        help="Action to perform: optimize, consolidate, or shard")
    
    # Shard-specific arguments
    parser.add_argument("--num-shards", type=int, 
                        help="Number of shards to create (for --action=shard)")
    parser.add_argument("--shard-output-dir", type=str,
                        help="Directory to write shard files (for --action=shard)")
    
    # Label selection arguments
    parser.add_argument("--labels", type=str,
                        help="Comma-separated list of labels to process (default: all labels)")
    parser.add_argument("--labels-file", type=str,
                        help="File containing labels to process, one per line (default: all labels)")
    
    # Add storage configuration arguments
    StorageConfig.add_storage_args(parser)
    
    # Verbosity control
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Configure storage using the new class method
    storage_config = StorageConfig.from_args(args)
    
    # Get source index path
    source_index_path = args.source_index
    if source_index_path is None:
        source_index_path = os.path.join(args.base_path, "unified_index.db")
    
    verbose = not args.quiet
    
    # Handle different actions
    if args.action == "optimize":
        # Get labels to process
        labels_to_process = None
        if args.labels:
            labels_to_process = [int(label.strip()) for label in args.labels.split(",")]
        elif args.labels_file:
            with open(args.labels_file, 'r') as f:
                labels_to_process = [int(line.strip()) for line in f if line.strip()]
        
        # Run optimization
        optimize_point_clouds(
            storage_config=storage_config,
            source_index_path=source_index_path,
            target_path=args.target_path,
            worker_id=args.worker_id,
            target_file_size=args.target_file_size,
            batch_size=args.batch_size,
            labels_to_process=labels_to_process,
            threads=args.threads,
            verbose=verbose
        )
    
    elif args.action == "consolidate":
        # Run consolidation
        consolidate_optimized_indices(
            storage_config=storage_config,
            target_path=args.target_path,
            target_index_path=args.target_index,
            threads=args.threads,
            verbose=verbose
        )
    
    elif args.action == "shard":
        if not args.num_shards:
            parser.error("--num-shards is required for shard action")
        
        # Run sharding
        shard_labels(
            storage_config=storage_config,
            source_index_path=source_index_path,
            num_shards=args.num_shards,
            output_dir=args.shard_output_dir,
            verbose=verbose
        )

if __name__ == "__main__":
    main()