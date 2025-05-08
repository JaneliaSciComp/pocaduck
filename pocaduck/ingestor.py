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
        current_file_id: Identifier for the current parquet file.
        db_connection: Connection to the DuckDB database for indexing.
    """
    
    def __init__(
        self, 
        storage_config: StorageConfig, 
        worker_id: Union[str, int],
        max_points_per_file: int = 10_000_000,
    ):
        """
        Initialize an Ingestor instance.
        
        Args:
            storage_config: Configuration for storage backend.
            worker_id: Unique identifier for the worker.
            max_points_per_file: Maximum number of points to store in a single parquet file.
        """
        self.storage_config = storage_config
        self.worker_id = str(worker_id)
        self.max_points_per_file = max_points_per_file
        self.current_points_count = 0
        self.file_counter = 0  # Counter to track the file number for sequential naming
        
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
        con.execute("""
            CREATE TABLE IF NOT EXISTS point_cloud_index (
                label UBIGINT,
                block_id VARCHAR,
                file_path VARCHAR,
                point_count UBIGINT,
                PRIMARY KEY (label, block_id)
            )
        """)
        
        return con
    
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
        
        # Check if we need to start a new file
        if self.current_points_count + num_points > self.max_points_per_file:
            old_counter = self.file_counter
            self.file_counter += 1  # Increment the file counter for a new file
            self.current_points_count = 0
            self.logger.info(f"Worker {self.worker_id}: Incrementing file counter from {old_counter} to {self.file_counter}")
        
        # Get file path for the current file with human-readable name
        file_path = os.path.join(self.data_dir, f"{self.worker_id}-{self.file_counter}.parquet")
        self.logger.info(f"Worker {self.worker_id}: Writing to file {os.path.basename(file_path)}, current point count: {self.current_points_count}, adding {num_points} points")
        
        # Create DataFrame from points
        # Ensure label is handled as a BIGINT to avoid type inconsistencies
        df = pd.DataFrame({
            'label': pd.Series([label] * len(points), dtype='int64'),
            'block_id': block_id,
            'data': list(points)  # Store each row of points as a list in the 'data' column
        })
        
        # Write to parquet file (append if it exists)
        if os.path.exists(file_path) and self.storage_config.storage_type == "local":
            self.logger.info(f"Appending to existing file {os.path.basename(file_path)}")
            
            # We'll use DuckDB to efficiently read the existing file
            # This is more efficient for large files than pd.read_parquet
            # A future enhancement could use a native DuckDB append operation
            existing_df = self.db_connection.execute(f"SELECT * FROM read_parquet('{file_path}')").fetchdf()
            
            # Append new data
            df = pd.concat([existing_df, df])
        else:
            self.logger.info(f"Creating new file {os.path.basename(file_path)}")
            
        # Write to parquet
        df.to_parquet(file_path, index=False)
        
        # Update index in DuckDB
        self.db_connection.execute("""
            INSERT OR REPLACE INTO point_cloud_index (label, block_id, file_path, point_count)
            VALUES (?, ?, ?, ?)
        """, [label, block_id, file_path, num_points])
        
        # Update current points count
        self.current_points_count += num_points
    
    def finalize(self) -> None:
        """
        Finalize the ingestion process for this worker.
        
        This method should be called when the worker has completed all writes.
        It ensures all data is properly committed and closes connections.
        """
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
        con.execute("""
            CREATE TABLE IF NOT EXISTS point_cloud_index (
                label UBIGINT,
                block_id VARCHAR,
                file_path VARCHAR,
                point_count UBIGINT,
                PRIMARY KEY (label, block_id)
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
                    INSERT OR IGNORE INTO point_cloud_index (label, block_id, file_path, point_count)
                    VALUES {placeholders}
                """, flat_data)
        
        # Commit and close
        con.commit()
        con.close()
        
        return output_path