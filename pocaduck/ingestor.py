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

from .storage_config import StorageConfig


class Ingestor:
    """
    Handles ingestion of point clouds for labels within blocks.

    The Ingestor class provides methods to write 3D point clouds associated with labels
    within blocks and to finalize the ingestion process. Each worker should have its own
    Ingestor instance.

    Attributes:
        storage_config: Configuration for storage backend.
        worker_id: Unique identifier for the worker.
        max_points_per_file: Maximum number of points to store in a single parquet file.
        current_points_count: Current count of points written to the current parquet file.
        file_counter: Counter to track the file number for sequential naming.
        current_file_path: Path to the current parquet file being written.
        current_file_df: In-memory DataFrame cache for the current file.
        db_connection: Connection to the DuckDB database for indexing.
    """
    
    def __init__(
        self,
        storage_config: StorageConfig,
        worker_id: Union[str, int],
        max_points_per_file: int = 10_000_000,
        verbose: bool = False,
    ):
        """
        Initialize an Ingestor instance.

        Args:
            storage_config: Configuration for storage backend.
            worker_id: Unique identifier for the worker.
            max_points_per_file: Maximum number of points to store in a single parquet file.
            verbose: Whether to output detailed logging information during operations.
                    Default is False for reduced output in production environments.
        """
        self.storage_config = storage_config
        self.worker_id = str(worker_id)
        self.max_points_per_file = max_points_per_file
        self.current_points_count = 0
        self.file_counter = 0  # Counter to track the file number for sequential naming
        self.current_file_path = None  # Path to the current parquet file
        self.current_file_df = None    # DataFrame cache for the current file
        self.verbose = verbose         # Whether to output detailed logging

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