"""
Point cloud querying for PoCADuck.

This module provides the Query class to retrieve point clouds by label, automatically
aggregating points across all blocks.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Set
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

from .storage_config import StorageConfig


class Query:
    """
    Handles querying of point clouds across blocks.
    
    The Query class provides methods to retrieve 3D point clouds for labels,
    automatically aggregating the points across all blocks that contain the label.
    
    Attributes:
        storage_config: Configuration for storage backend.
        db_connection: Connection to the DuckDB database for indexing.
    """
    
    def __init__(self, storage_config: StorageConfig, index_path: Optional[str] = None):
        """
        Initialize a Query instance.
        
        Args:
            storage_config: Configuration for storage backend.
            index_path: Path to the unified index. If None, defaults to
                        {base_path}/unified_index.db.
        """
        self.storage_config = storage_config
        
        # Set the index path
        if index_path is None:
            self.index_path = os.path.join(storage_config.base_path, "unified_index.db")
        else:
            self.index_path = index_path
        
        # Initialize database connection
        self.db_connection = self._initialize_db_connection()
    
    def _initialize_db_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Initialize connection to DuckDB for querying.
        
        Returns:
            DuckDB connection.
        """
        # Create a connection to DuckDB
        con = duckdb.connect(self.index_path)
        
        # Apply storage configuration
        duckdb_config = self.storage_config.get_duckdb_config()
        for key, value in duckdb_config.items():
            con.execute(f"SET {key}='{value}'")
        
        return con
    
    def get_labels(self) -> np.ndarray:
        """
        Get all labels in the database.
        
        Returns:
            Numpy array of all unique label IDs.
        """
        result = self.db_connection.execute("SELECT DISTINCT label FROM point_cloud_index").fetchall()
        return np.array([r[0] for r in result], dtype=np.uint64)
    
    def get_blocks_for_label(self, label: int) -> List[str]:
        """
        Get all blocks that contain a specific label.
        
        Args:
            label: The label to query for.
            
        Returns:
            List of block IDs that contain the label.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
            
        result = self.db_connection.execute(
            "SELECT block_id FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        return [r[0] for r in result]
    
    def get_point_count(self, label: int) -> int:
        """
        Get the total number of points for a specific label.
        
        Args:
            label: The label to query for.
            
        Returns:
            Total number of points for the label across all blocks.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
            
        result = self.db_connection.execute(
            "SELECT SUM(point_count) FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchone()
        return result[0] if result[0] is not None else 0
    
    def get_points(self, label: int) -> np.ndarray:
        """
        Get all point data for a specific label.
        
        Args:
            label: The label to query for.
            
        Returns:
            Numpy array containing all point data for the label. The shape is (N, D) where
            N is the number of points and D is the dimension of the point data.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
            
        # Find all files containing this label
        file_info = self.db_connection.execute(
            "SELECT file_path FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        
        file_paths = [info[0] for info in file_info]
        
        if not file_paths:
            # No data for this label
            return np.array([], dtype=np.int64)
        
        # Use DuckDB to efficiently read and filter the parquet files
        # Simplify the query to just get the data for the requested label
        query = f"""
            SELECT data
            FROM parquet_scan([{','.join(f"'{path}'" for path in file_paths)}])
            WHERE label = {label}
        """
        
        # Execute query and get results as Pandas DataFrame
        df = self.db_connection.execute(query).fetchdf()
        
        # Convert list-based data column back to numpy arrays and stack them
        if len(df) == 0:
            return np.array([], dtype=np.int64)
        
        # Convert the list-based data back to a numpy array
        points_list = df['data'].tolist()
        
        # Stack the points and remove duplicates
        if points_list:
            # Stack all points
            points = np.vstack(points_list).astype(np.int64)
            
            # Remove duplicates to ensure we only have unique points
            points = np.unique(points, axis=0)
        else:
            return np.array([], dtype=np.int64)
        
        return points
    
    def close(self) -> None:
        """
        Close the query connection.
        
        This method should be called when the query object is no longer needed.
        """
        if hasattr(self, 'db_connection') and self.db_connection is not None:
            self.db_connection.close()
            self.db_connection = None