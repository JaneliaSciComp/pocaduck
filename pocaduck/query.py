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

# Default thread count to use when os.cpu_count() fails
DEFAULT_THREAD_COUNT = 8


class Query:
    """
    Handles querying of point clouds across blocks.

    The Query class provides methods to retrieve 3D point clouds for labels,
    automatically aggregating the points across all blocks that contain the label.

    Note: This class always opens the database in read-only mode since it only
    performs read operations. This allows access to databases where the user
    only has read permissions.

    Attributes:
        storage_config: Configuration for storage backend.
        db_connection: Connection to the DuckDB database for indexing.
    """
    
    def __init__(
        self,
        storage_config: StorageConfig,
        index_path: Optional[str] = None,
        threads: Optional[int] = None,
        cache_size: int = 10
    ):
        """
        Initialize a Query instance.

        Args:
            storage_config: Configuration for storage backend.
            index_path: Path to the unified index. If None, defaults to
                        {base_path}/unified_index.db.
            threads: Number of threads to use for parallel processing. If None,
                     uses os.cpu_count() to detect available CPU cores with a
                     fallback of 8 threads if detection fails.
            cache_size: Number of label point clouds to cache in memory (0 to disable).
        """
        self.storage_config = storage_config
        self.cache_size = cache_size

        # Set threads to system CPU count if not provided
        self.threads = threads if threads is not None else (os.cpu_count() or DEFAULT_THREAD_COUNT)

        # Set the index path
        if index_path is None:
            self.index_path = os.path.join(storage_config.base_path, "unified_index.db")
        else:
            self.index_path = index_path

        # Initialize database connection
        self.db_connection = self._initialize_db_connection()
        
        # Initialize point cloud cache for frequently queried labels
        self._points_cache = {}  # Maps label to point cloud data
    
    def _initialize_db_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Initialize connection to DuckDB for querying with optimized settings
        for parallel processing.

        Returns:
            DuckDB connection.
        """
        # Always use read-only mode since Query class only performs read operations
        # This allows access to databases where the user only has read permissions
        con = duckdb.connect(self.index_path, read_only=True)

        # Apply storage configuration
        duckdb_config = self.storage_config.get_duckdb_config()
        for key, value in duckdb_config.items():
            con.execute(f"SET {key}='{value}'")

        # Configure DuckDB for optimal performance
        con.execute(f"PRAGMA threads={self.threads}")

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
            List of unique block IDs that contain the label.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
            
        # With our new schema, we may have multiple entries for the same label-block combination
        # due to splitting large point clouds, so we need to select distinct block_ids
        result = self.db_connection.execute(
            "SELECT DISTINCT block_id FROM point_cloud_index WHERE label = ?",
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
    
    def get_points(self, label: int, use_cache: bool = True) -> np.ndarray:
        """
        Get all point data for a specific label.
        
        Args:
            label: The label to query for.
            use_cache: Whether to use the in-memory point cloud cache (if enabled).
            
        Returns:
            Numpy array containing all point data for the label. The shape is (N, D) where
            N is the number of points and D is the dimension of the point data.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
            
        # Check if the points are in the cache
        if use_cache and self.cache_size > 0 and label in self._points_cache:
            # Move this label to the end of the cache to mark it as most recently used
            points = self._points_cache.pop(label)
            self._points_cache[label] = points
            return points
            
        # Find all files containing this label
        file_info = self.db_connection.execute(
            "SELECT DISTINCT file_path FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        
        file_paths = [info[0] for info in file_info]
        
        if not file_paths:
            # No data for this label
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            return empty_result
            
        # Use the most reliable query to get the point data
        query = f"""
            SELECT data
            FROM parquet_scan([{','.join(f"'{path}'" for path in file_paths)}])
            WHERE label = {label}
        """
        
        # Execute query and get results as Pandas DataFrame
        df = self.db_connection.execute(query).fetchdf()
        
        # Handle empty result
        if len(df) == 0:
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            return empty_result
        
        # Convert list-based data column back to numpy arrays and stack them
        points_list = df['data'].tolist()
        
        if not points_list:
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            return empty_result
            
        # Stack the points
        points = np.vstack(points_list).astype(np.int64)
        
        # Remove duplicates to ensure we only have unique points
        points = np.unique(points, axis=0)
        
        # Cache the result if enabled
        if use_cache and self.cache_size > 0:
            self._update_cache(label, points)
            
        return points
        
    def _update_cache(self, label: int, points: np.ndarray) -> None:
        """
        Update the points cache with the given label and points array.
        
        Args:
            label: The label identifier.
            points: The point cloud data.
        """
        if self.cache_size <= 0:
            return
            
        # Add to cache
        self._points_cache[label] = points
        
        # If cache is too large, remove least recently used entries
        if len(self._points_cache) > self.cache_size:
            # Get a key to remove (the first one in the dict, which is the oldest)
            # Python 3.7+ preserves insertion order, so this works as a simple LRU cache
            key_to_remove = next(iter(self._points_cache))
            del self._points_cache[key_to_remove]
    
    def close(self) -> None:
        """
        Close the query connection and clear caches.
        
        This method should be called when the query object is no longer needed.
        """
        # Clear the points cache
        if hasattr(self, '_points_cache'):
            self._points_cache.clear()
            
        # Close the database connection
        if hasattr(self, 'db_connection') and self.db_connection is not None:
            self.db_connection.close()
            self.db_connection = None