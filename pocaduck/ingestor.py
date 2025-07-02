"""
Point cloud ingestion for PoCADuck.

This module provides the Ingestor class to handle writing point clouds for labels within blocks
and managing the storage and indexing of these point clouds.
"""

import os
import uuid
import glob
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from pathlib import Path
from collections import defaultdict

from .storage_config import StorageConfig

try:
    import vastdb
    VASTDB_AVAILABLE = True
except ImportError:
    VASTDB_AVAILABLE = False


class VastDBIngestorBackend:
    """
    VastDB-based backend for point cloud ingestion with batching.
    
    This backend accumulates points in memory and performs batch inserts to VastDB
    when the batch size is reached or when finalize() is called.
    """
    
    def __init__(self, storage_config: StorageConfig, batch_size: int = 1000, verbose: bool = False):
        """
        Initialize VastDB ingestion backend.
        
        Args:
            storage_config: Storage configuration with VastDB parameters
            batch_size: Number of rows to batch before inserting
            verbose: Whether to output detailed logging
        """
        if not VASTDB_AVAILABLE:
            raise ImportError("VastDB SDK is not available. Please install vastdb package.")
        
        self.storage_config = storage_config
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Set up logging
        import logging
        self.logger = logging.getLogger(__name__)
        
        # Connect to VastDB
        self.session = vastdb.connect(
            endpoint=storage_config.vastdb_endpoint,
            access=storage_config.vastdb_access_key,
            secret=storage_config.vastdb_secret_key
        )
        
        # Get references to bucket, schema, and table
        self.bucket = self.session.bucket(storage_config.vastdb_bucket)
        self.schema = self.bucket.schema(storage_config.vastdb_schema)
        self.table = self.schema.table(storage_config.vastdb_table)
        
        # Batching state - accumulate by label across all blocks
        # Each label gets aggregated across blocks into a single row
        self._batch_data = defaultdict(list)  # label -> list of (block_id, points) tuples
        self._batch_count = 0
    
    def write(self, label: int, block_id: str, points: np.ndarray) -> None:
        """
        Write point cloud data for a label within a block.
        
        Points are accumulated in memory until batch_size is reached or finalize() is called.
        
        Args:
            label: The uint64 label identifier
            block_id: Identifier for the block containing the points
            points: Numpy array of shape (N, D) containing point data
        """
        # Validate input
        if not isinstance(points, np.ndarray) or points.ndim != 2:
            raise ValueError("Points must be a numpy array of shape (N, D) containing point data")
        
        num_points = points.shape[0]
        if num_points == 0:
            return  # Skip empty point clouds
        
        # Add to batch - store points flattened for Arrow list format
        flattened_points = points.flatten().astype(np.int64)
        self._batch_data[label].append((block_id, flattened_points))
        self._batch_count += 1
        
        if self.verbose:
            self.logger.info(f"Accumulated {len(flattened_points)} points for label {label}, block {block_id} (batch count: {self._batch_count})")
        
        # Flush if batch size reached
        if self._batch_count >= self.batch_size:
            self._flush_batch()
    
    def _flush_batch(self) -> None:
        """Flush the current batch to VastDB."""
        if not self._batch_data:
            return
        
        if self.verbose:
            self.logger.info(f"Flushing batch with {len(self._batch_data)} labels")
        
        # Prepare Arrow table data
        labels = []
        block_ids = []  
        points_lists = []
        
        for label, block_point_pairs in self._batch_data.items():
            # Aggregate all points for this label across blocks
            all_points = []
            all_block_ids = []
            
            for block_id, points in block_point_pairs:
                all_points.extend(points)
                all_block_ids.append(block_id)
            
            # Create single row for this label with aggregated points
            labels.append(label)
            # For MVP: use first block_id, but could be comma-separated list
            block_ids.append(all_block_ids[0] if all_block_ids else "")  
            points_lists.append(all_points)
        
        # Create Arrow table
        arrow_table = pa.table({
            'label': pa.array(labels, type=pa.uint64()),
            'block_id': pa.array(block_ids, type=pa.string()),
            'points': pa.array(points_lists, type=pa.list_(pa.int64()))
        })
        
        # Insert batch using VastDB transaction
        with self.session.transaction() as tx:
            self.table.insert(arrow_table)
            if self.verbose:
                self.logger.info(f"Inserted batch of {len(labels)} labels to VastDB")
        
        # Clear batch
        self._batch_data.clear()
        self._batch_count = 0
    
    def finalize(self) -> None:
        """Final flush and cleanup."""
        self._flush_batch()
        if self.verbose:
            self.logger.info("VastDB ingestion finalized")


class Ingestor:
    """
    Unified interface for ingesting point clouds using different backends.
    
    The Ingestor class provides a consistent API for writing point clouds to different
    storage backends:
    - VastDB: Direct Arrow-based ingestion with batching
    - Parquet+DuckDB: File-based storage with worker coordination
    
    The backend is automatically selected based on the storage_config.storage_type.
    """
    
    def __init__(
        self,
        storage_config: StorageConfig,
        worker_id: Union[str, int] = 0,
        max_points_per_file: int = 10_000_000,
        verbose: bool = False,
        batch_size: int = 1000,
    ):
        """
        Initialize an Ingestor instance with the appropriate backend.

        Args:
            storage_config: Configuration for storage backend.
            worker_id: Unique identifier for the worker (file-based backends only).
            max_points_per_file: Maximum number of points to store in a single parquet file (file-based backends only).
            verbose: Whether to output detailed logging information during operations.
            batch_size: Number of rows to batch before inserting (VastDB backend only).
        """
        self.storage_config = storage_config
        
        # Select the appropriate backend based on storage type
        if storage_config.storage_type == "vastdb":
            self._backend = VastDBIngestorBackend(storage_config, batch_size, verbose)
        elif storage_config.storage_type in ["local", "s3", "gcs", "azure"]:
            # Create the original file-based implementation
            self._backend = self._create_file_backend(storage_config, worker_id, max_points_per_file, verbose)
        else:
            raise ValueError(f"Unsupported storage type: {storage_config.storage_type}")
    
    def _create_file_backend(self, storage_config, worker_id, max_points_per_file, verbose):
        """Create file-based backend by embedding the original implementation."""
        # For now, use the existing file-based logic inline
        # This preserves backward compatibility
        return ParquetDuckDBIngestorBackend(storage_config, worker_id, max_points_per_file, verbose)
    
    # Delegate methods to backend
    def write(self, label: int, block_id: str, points: np.ndarray) -> None:
        """Write point cloud data for a label within a block."""
        return self._backend.write(label, block_id, points)
    
    def finalize(self) -> None:
        """Finalize ingestion and close connections."""
        return self._backend.finalize()
    
    @staticmethod
    def consolidate_indexes(storage_config: StorageConfig) -> None:
        """Consolidate worker indexes (file-based backends only)."""
        if storage_config.storage_type in ["local", "s3", "gcs", "azure"]:
            ParquetDuckDBIngestorBackend.consolidate_indexes(storage_config)
        else:
            # VastDB doesn't need index consolidation
            pass


class ParquetDuckDBIngestorBackend:
    """
    File-based backend for point cloud ingestion.
    
    This is the original implementation using parquet files for storage
    and DuckDB for indexing.
    """
    
    def __init__(
        self,
        storage_config: StorageConfig,
        worker_id: Union[str, int],
        max_points_per_file: int = 10_000_000,
        verbose: bool = False,
    ):
        """
        Initialize file-based ingestion backend.

        Args:
            storage_config: Configuration for storage backend.
            worker_id: Unique identifier for the worker.
            max_points_per_file: Maximum number of points to store in a single parquet file.
            verbose: Whether to output detailed logging information during operations.
        """
        self.storage_config = storage_config
        self.worker_id = str(worker_id)
        self.max_points_per_file = max_points_per_file
        self.current_points_count = 0
        self.file_counter = 0
        self.current_file_path = None
        self.current_file_df = None
        self.verbose = verbose

        # Set up logging
        import logging
        self.logger = logging.getLogger(__name__)

        # Set up storage paths
        base_path = storage_config.base_path
        self.worker_dir = os.path.join(base_path, f"worker_{self.worker_id}")
        self.data_dir = os.path.join(self.worker_dir, "data")
        self.db_path = os.path.join(self.worker_dir, f"index_{self.worker_id}.db")

        # Create directories if necessary
        if storage_config.storage_type == "local":
            os.makedirs(self.data_dir, exist_ok=True)

        # Set up database connection
        self.db_connection = self._initialize_db_connection()
    
    def _initialize_db_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Initialize connection to DuckDB for indexing.
        
        Returns:
            DuckDB connection.
        """
        # Create a connection to DuckDB
        # Pass storage configuration to DuckDB
        con = duckdb.connect(self.db_path)
        
        # Apply storage configuration
        duckdb_config = self.storage_config.get_duckdb_config()
        for key, value in duckdb_config.items():
            con.execute(f"SET {key}='{value}'")
        
        # Create the index table
        # Note: We no longer have a PRIMARY KEY constraint on (label, block_id)
        # because we need to allow multiple entries when points for a label-block
        # are split across multiple files
        con.execute("""
            CREATE TABLE IF NOT EXISTS point_cloud_index (
                label UBIGINT,
                block_id VARCHAR,
                file_path VARCHAR,
                point_count UBIGINT
            )
        """)
        
        return con
    
    def _flush_current_file(self):
        """
        Flush the current DataFrame to a parquet file.
        """
        if self.current_file_df is not None and len(self.current_file_df) > 0:
            if self.verbose:
                self.logger.info(f"Worker {self.worker_id}: Flushing {len(self.current_file_df)} rows to {os.path.basename(self.current_file_path)}")

            # Write DataFrame to parquet
            self.current_file_df.to_parquet(self.current_file_path, index=False)

            # Reset DataFrame to clear memory
            self.current_file_df = None

    def write(
        self,
        label: int,
        block_id: str,
        points: np.ndarray
    ) -> None:
        """
        Write point cloud data for a label within a block.

        Args:
            label: The uint64 label identifier.
            block_id: Identifier for the block containing the points.
            points: Numpy array of shape (N, D) containing point data where D is the dimension
                   of the data (e.g., 3 for just x,y,z coordinates, or more for additional attributes).

        Raises:
            ValueError: If points is not a valid numpy array.
        """
        # Validate input
        if not isinstance(points, np.ndarray) or points.ndim != 2:
            raise ValueError("Points must be a numpy array of shape (N, D) containing point data")

        num_points = points.shape[0]
        if num_points == 0:
            return  # Skip empty point clouds

        # Handle points in batches respecting max_points_per_file limit
        remaining_points = points
        points_written = 0

        while points_written < num_points:
            # Calculate how many points we can add to the current file
            space_in_current_file = self.max_points_per_file - self.current_points_count
            points_to_write = min(space_in_current_file, len(remaining_points))

            if points_to_write <= 0:
                # Current file is full, flush it and start a new file
                self._flush_current_file()
                old_counter = self.file_counter
                self.file_counter += 1
                self.current_points_count = 0
                self.current_file_path = None
                self.current_file_df = None
                if self.verbose:
                    self.logger.info(f"Worker {self.worker_id}: Incrementing file counter from {old_counter} to {self.file_counter}")
                continue  # Recalculate space in the new file

            # Select the batch of points to write to this file
            batch = remaining_points[:points_to_write]

            # Get or create the file path for the current file
            if self.current_file_path is None:
                self.current_file_path = os.path.join(self.data_dir, f"{self.worker_id}-{self.file_counter}.parquet")
                if self.verbose:
                    self.logger.info(f"Worker {self.worker_id}: Using file {os.path.basename(self.current_file_path)}, "
                                    f"current point count: {self.current_points_count}, "
                                    f"adding {len(batch)} points "
                                    f"(batch {points_written+1}-{points_written+len(batch)} of {num_points})")

            # Create DataFrame from batch of points
            # Ensure label is handled as a BIGINT to avoid type inconsistencies
            batch_df = pd.DataFrame({
                'label': pd.Series([label] * len(batch), dtype='int64'),
                'block_id': block_id,
                'data': list(batch)  # Store each row of points as a list in the 'data' column
            })

            # Check if this is an existing file that needs to be loaded first
            if self.current_file_df is None:
                if os.path.exists(self.current_file_path) and self.storage_config.storage_type == "local":
                    if self.verbose:
                        self.logger.info(f"Loading existing file {os.path.basename(self.current_file_path)}")

                    # Use DuckDB to efficiently read the existing file
                    self.current_file_df = self.db_connection.execute(
                        f"SELECT * FROM read_parquet('{self.current_file_path}')"
                    ).fetchdf()
                else:
                    # New file, initialize empty DataFrame
                    self.current_file_df = pd.DataFrame(columns=['label', 'block_id', 'data'])

            # Append new data to in-memory DataFrame
            self.current_file_df = pd.concat([self.current_file_df, batch_df])

            # Update the index
            # Check if this is a new entry for this label-block combination
            existing_entry = self.db_connection.execute("""
                SELECT file_path, point_count FROM point_cloud_index
                WHERE label = ? AND block_id = ?
            """, [label, block_id]).fetchone()

            if existing_entry:
                # This label-block combo exists in the index
                # Add a new entry with this file path
                self.db_connection.execute("""
                    INSERT INTO point_cloud_index (label, block_id, file_path, point_count)
                    VALUES (?, ?, ?, ?)
                """, [label, block_id, self.current_file_path, len(batch)])
            else:
                # This is a new label-block combination
                self.db_connection.execute("""
                    INSERT INTO point_cloud_index (label, block_id, file_path, point_count)
                    VALUES (?, ?, ?, ?)
                """, [label, block_id, self.current_file_path, len(batch)])

            # Update tracking variables
            self.current_points_count += len(batch)
            points_written += len(batch)
            remaining_points = remaining_points[points_to_write:]
    
    def finalize(self) -> None:
        """
        Finalize the ingestion process for this worker.

        This method should be called when the worker has completed all writes.
        It ensures all data is properly flushed to disk, committed and closes connections.
        """
        # Flush any pending data to disk
        self._flush_current_file()

        # Commit any pending transactions
        self.db_connection.commit()

        # Close the database connection
        self.db_connection.close()
    
    @staticmethod
    def consolidate_indexes(
        storage_config: StorageConfig, 
        output_path: Optional[str] = None
    ) -> str:
        """
        Consolidate worker indexes into a unified index.
        
        This static method should be called after all workers have finalized their
        ingestion processes. It consolidates all worker-specific indexes into a
        unified index that can be used for querying.
        
        Args:
            storage_config: Configuration for storage backend.
            output_path: Path to store the consolidated index. If None, defaults to
                         {base_path}/unified_index.db.
        
        Returns:
            Path to the consolidated index.
        """
        base_path = storage_config.base_path
        
        if output_path is None:
            output_path = os.path.join(base_path, "unified_index.db")
        
        # Create a connection to the output database
        con = duckdb.connect(output_path)
        
        # Apply storage configuration
        duckdb_config = storage_config.get_duckdb_config()
        for key, value in duckdb_config.items():
            con.execute(f"SET {key}='{value}'")
        
        # Create the consolidated index table
        # Note: We no longer have a PRIMARY KEY constraint on (label, block_id)
        # because we need to allow multiple entries when points for a label-block
        # are split across multiple files
        con.execute("""
            CREATE TABLE IF NOT EXISTS point_cloud_index (
                label UBIGINT,
                block_id VARCHAR,
                file_path VARCHAR,
                point_count UBIGINT
            )
        """)
        
        # Get all worker index paths
        worker_index_pattern = os.path.join(base_path, "worker_*/index_*.db")
        worker_index_paths = glob.glob(worker_index_pattern)
        
        # Process each worker index
        for worker_index_path in worker_index_paths:
            # Create a separate connection to the worker database
            worker_con = duckdb.connect(worker_index_path)
            
            # Get all rows from the worker index
            worker_data = worker_con.execute("SELECT * FROM point_cloud_index").fetchall()
            worker_con.close()
            
            # If there are rows, insert them into the unified index
            if worker_data:
                # Insert the data into the unified index
                placeholders = ", ".join(["(?, ?, ?, ?)"] * len(worker_data))
                # Flatten the data for the execute statement
                flat_data = [val for row in worker_data for val in row]
                
                con.execute(f"""
                    INSERT INTO point_cloud_index (label, block_id, file_path, point_count)
                    VALUES {placeholders}
                """, flat_data)
        
        # Commit and close
        con.commit()
        con.close()
        
        return output_path