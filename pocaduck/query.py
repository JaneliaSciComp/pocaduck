"""
Point cloud querying for PoCADuck.

This module provides the Query class to retrieve point clouds by label, automatically
aggregating points across all blocks.
"""

import os
import time
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

    Optimization Support:
    This class automatically detects and uses optimized data if available in the
    {base_path}/optimized directory. Optimized data can be created using the
    optimize_point_cloud.py script, which reorganizes point clouds by label for
    significantly faster retrieval.

    Attributes:
        storage_config: Configuration for storage backend.
        db_connection: Connection to the DuckDB database for indexing.
        using_optimized_data: Boolean indicating if optimized data is being used.
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
        self.using_optimized_data = False

        # Set threads to system CPU count if not provided
        self.threads = threads if threads is not None else (os.cpu_count() or DEFAULT_THREAD_COUNT)

        # Set the index path
        if index_path is None:
            # Check for optimized data
            has_optimized, optimized_index, _ = self._check_for_optimized_data()
            if has_optimized:
                self.index_path = optimized_index
                self.using_optimized_data = True
            else:
                self.index_path = os.path.join(storage_config.base_path, "unified_index.db")
        else:
            self.index_path = index_path

        # Initialize database connection
        self.db_connection = self._initialize_db_connection()
        
        # Initialize point cloud cache for frequently queried labels
        self._points_cache = {}  # Maps label to point cloud data
    
    def _check_for_optimized_data(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if optimized data is available for this storage configuration.

        Returns:
            Tuple containing:
            - Boolean indicating if optimized data is available
            - Path to the optimized index (or None if not available)
            - Path to the optimized directory (or None if not available)
        """
        # Check if optimized directory exists
        optimized_dir = os.path.join(self.storage_config.base_path, "optimized")
        if not os.path.isdir(optimized_dir):
            return False, None, None

        # Check if optimized index exists
        optimized_index = os.path.join(optimized_dir, "optimized_index.db")
        if not os.path.exists(optimized_index):
            return False, None, None

        return True, optimized_index, optimized_dir

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
    
    def get_blocks_for_label(self, label: int, timing: bool = False) -> Union[List[str], Tuple[List[str], Dict]]:
        """
        Get all blocks that contain a specific label.
        
        Note: For optimized data, this returns an empty list since 
        optimized data consolidates all blocks for each label.
        
        Args:
            label: The label to query for.
            timing: If True, return timing information as second element of tuple.
            
        Returns:
            List of unique block IDs that contain the label, or empty list for optimized data.
            If timing=True, returns (blocks, timing_info) tuple.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            # If using optimized data, block information is not available
            if self.using_optimized_data:
                return []
                
            # With our new schema, we may have multiple entries for the same label-block combination
            # due to splitting large point clouds, so we need to select distinct block_ids
            result = self.db_connection.execute(
                "SELECT DISTINCT block_id FROM point_cloud_index WHERE label = ?",
                [label]
            ).fetchall()
            return [r[0] for r in result]
        
        # Timing-enabled version
        start_time = time.time()
        timing_info = {
            'total_time': 0.0,
            'index_lookup_time': 0.0,
            'using_optimized_data': self.using_optimized_data,
            'blocks_found': 0
        }
        
        # If using optimized data, block information is not available
        if self.using_optimized_data:
            timing_info['total_time'] = time.time() - start_time
            timing_info['sql_query'] = "N/A (optimized data has no block information)"
            return [], timing_info
        
        # Execute query with timing
        timing_info['sql_query'] = "SELECT DISTINCT block_id FROM point_cloud_index WHERE label = ?"
        index_start = time.time()
        result = self.db_connection.execute(
            "SELECT DISTINCT block_id FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        timing_info['index_lookup_time'] = time.time() - index_start
        timing_info['total_time'] = time.time() - start_time
        
        blocks = [r[0] for r in result]
        timing_info['blocks_found'] = len(blocks)
        
        return blocks, timing_info
    
    def get_point_count(self, label: int, timing: bool = False) -> Union[int, Tuple[int, Dict]]:
        """
        Get the total number of points for a specific label.
        
        Args:
            label: The label to query for.
            timing: If True, return timing information as second element of tuple.
            
        Returns:
            Total number of points for the label across all blocks.
            If timing=True, returns (count, timing_info) tuple.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            result = self.db_connection.execute(
                "SELECT SUM(point_count) FROM point_cloud_index WHERE label = ?",
                [label]
            ).fetchone()
            return result[0] if result[0] is not None else 0
        
        # Timing-enabled version
        start_time = time.time()
        timing_info = {
            'total_time': 0.0,
            'index_lookup_time': 0.0,
            'files_queried': 0,
            'sql_query': "SELECT SUM(point_count) FROM point_cloud_index WHERE label = ?",
            'using_optimized_data': self.using_optimized_data
        }
        
        # Execute query with timing
        index_start = time.time()
        result = self.db_connection.execute(
            "SELECT SUM(point_count) FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchone()
        timing_info['index_lookup_time'] = time.time() - index_start
        
        # Get file count for this label
        files_result = self.db_connection.execute(
            "SELECT COUNT(DISTINCT file_path) FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchone()
        timing_info['files_queried'] = files_result[0] if files_result[0] is not None else 0
        
        timing_info['total_time'] = time.time() - start_time
        
        count = result[0] if result[0] is not None else 0
        return count, timing_info
    
    def get_points(self, label: int, use_cache: bool = True, timing: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Get all point data for a specific label.

        This method automatically uses optimized data if available, for better performance.

        Args:
            label: The label to query for.
            use_cache: Whether to use the in-memory point cloud cache (if enabled).
            timing: If True, return timing information as second element of tuple.

        Returns:
            Numpy array containing all point data for the label. The shape is (N, D) where
            N is the number of points and D is the dimension of the point data.
            If timing=True, returns (points, timing_info) tuple.
        """
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)

        if not timing:
            # Check if the points are in the cache
            if use_cache and self.cache_size > 0 and label in self._points_cache:
                # Move this label to the end of the cache to mark it as most recently used
                points = self._points_cache.pop(label)
                self._points_cache[label] = points
                return points

            # Use optimized data if available, otherwise fall back to original method
            if self.using_optimized_data:
                return self._get_points_optimized(label, use_cache)
            else:
                return self._get_points_original(label, use_cache)
        
        # Timing-enabled version
        start_time = time.time()
        timing_info = {
            'label': label,
            'total_time': 0.0,
            'cache_lookup_time': 0.0,
            'index_lookup_time': 0.0,
            'data_read_time': 0.0,
            'processing_time': 0.0,
            'files_queried': 0,
            'file_details': [],
            'query_efficiency': {},
            'sql_query': '',
            'using_optimized_data': self.using_optimized_data,
            'cache_hit': False,
            'points_returned': 0
        }

        # Check cache first
        cache_start = time.time()
        if use_cache and self.cache_size > 0 and label in self._points_cache:
            points = self._points_cache.pop(label)
            self._points_cache[label] = points
            timing_info['cache_hit'] = True
            timing_info['cache_lookup_time'] = time.time() - cache_start
            timing_info['total_time'] = time.time() - start_time
            timing_info['points_returned'] = len(points)
            return points, timing_info
        
        timing_info['cache_lookup_time'] = time.time() - cache_start

        # Use optimized data if available, otherwise fall back to original method
        if self.using_optimized_data:
            points, method_timing = self._get_points_optimized_with_timing(label, use_cache)
        else:
            points, method_timing = self._get_points_original_with_timing(label, use_cache)
        
        # Merge method timing into main timing info
        for key, value in method_timing.items():
            if key in timing_info:
                timing_info[key] = value
        
        timing_info['total_time'] = time.time() - start_time
        timing_info['points_returned'] = len(points)
        
        return points, timing_info

    def _get_points_optimized_with_timing(self, label: int, use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Get points using the optimized data structure with detailed timing information.

        Args:
            label: The label to query for.
            use_cache: Whether to use the in-memory point cloud cache (if enabled).

        Returns:
            Tuple of (points array, timing_info dict).
        """
        timing_info = {
            'index_lookup_time': 0.0,
            'data_read_time': 0.0,
            'processing_time': 0.0,
            'files_queried': 0,
            'file_details': [],
            'query_efficiency': {},
            'sql_query': ''
        }

        # Get file info from the optimized index
        index_start = time.time()
        file_info = self.db_connection.execute(
            "SELECT DISTINCT file_path FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        timing_info['index_lookup_time'] = time.time() - index_start

        if not file_info:
            # No data for this label
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            timing_info['files_queried'] = 0
            timing_info['sql_query'] = "SELECT DISTINCT file_path FROM point_cloud_index WHERE label = ?"
            return empty_result, timing_info

        # Handle multiple files for a single label
        file_paths = [info[0] for info in file_info]
        timing_info['files_queried'] = len(file_paths)
        
        # Get detailed file information
        for file_path in file_paths:
            file_detail = {'file_path': file_path}
            if os.path.exists(file_path):
                file_detail['file_size_mb'] = os.path.getsize(file_path) / (1024 * 1024)
            else:
                file_detail['file_size_mb'] = 0.0
            timing_info['file_details'].append(file_detail)
        
        # Query all files that contain this label
        query = f"""
            SELECT data
            FROM parquet_scan([{','.join(f"'{path}'" for path in file_paths)}])
            WHERE label = {label}
        """
        timing_info['sql_query'] = query

        # Execute query and get results as Pandas DataFrame
        data_start = time.time()
        df = self.db_connection.execute(query).fetchdf()
        timing_info['data_read_time'] = time.time() - data_start

        # Handle empty result
        if len(df) == 0:
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            timing_info['query_efficiency']['points_per_file'] = 0.0
            return empty_result, timing_info

        # Process the data
        processing_start = time.time()
        points_list = df['data'].tolist()
        points = np.array(points_list, dtype=np.int64)
        timing_info['processing_time'] = time.time() - processing_start

        # Get actual points per file by querying each file individually
        points_per_file_counts = []
        individual_read_start = time.time()
        
        for file_path in file_paths:
            file_query = f"SELECT COUNT(*) FROM parquet_scan('{file_path}') WHERE label = {label}"
            file_count_result = self.db_connection.execute(file_query).fetchone()
            file_points = file_count_result[0] if file_count_result[0] is not None else 0
            points_per_file_counts.append(file_points)
        
        individual_query_time = time.time() - individual_read_start

        # Calculate efficiency metrics
        total_points = len(points)
        timing_info['query_efficiency'] = {
            'total_points': total_points,
            'files_accessed': len(file_paths),
            'points_per_file_range': {
                'min': min(points_per_file_counts) if points_per_file_counts else 0,
                'max': max(points_per_file_counts) if points_per_file_counts else 0,
                'avg': sum(points_per_file_counts) / len(points_per_file_counts) if points_per_file_counts else 0.0
            },
            'points_per_file_counts': points_per_file_counts,
            'individual_file_query_time': individual_query_time
        }

        # Update file details with actual point counts
        for i, file_detail in enumerate(timing_info['file_details']):
            if i < len(points_per_file_counts):
                file_detail['actual_points_in_file'] = points_per_file_counts[i]
            file_detail['read_time'] = timing_info['data_read_time'] / len(file_paths)

        # Cache the result if enabled
        if use_cache and self.cache_size > 0:
            self._update_cache(label, points)

        return points, timing_info

    def _get_points_optimized(self, label: int, use_cache: bool = True) -> np.ndarray:
        """
        Get points using the optimized data structure.

        Args:
            label: The label to query for.
            use_cache: Whether to use the in-memory point cloud cache (if enabled).

        Returns:
            Numpy array containing all point data for the label.
        """
        # Get file info from the optimized index
        file_info = self.db_connection.execute(
            "SELECT DISTINCT file_path FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()

        if not file_info:
            # No data for this label
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            return empty_result

        # Handle multiple files for a single label
        file_paths = [info[0] for info in file_info]
        
        # Query all files that contain this label
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

        # Get the points - convert list of point rows to 2D numpy array
        points_list = df['data'].tolist()
        points = np.array(points_list, dtype=np.int64)

        # Cache the result if enabled
        if use_cache and self.cache_size > 0:
            self._update_cache(label, points)

        return points

    def _get_points_original_with_timing(self, label: int, use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Original implementation with timing information.

        Args:
            label: The label to query for.
            use_cache: Whether to use the in-memory point cloud cache (if enabled).

        Returns:
            Tuple of (points array, timing_info dict).
        """
        timing_info = {
            'index_lookup_time': 0.0,
            'data_read_time': 0.0,
            'processing_time': 0.0,
            'files_queried': 0,
            'file_details': [],
            'query_efficiency': {},
            'sql_query': ''
        }

        # Find all files containing this label
        index_start = time.time()
        file_info = self.db_connection.execute(
            "SELECT DISTINCT file_path FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        timing_info['index_lookup_time'] = time.time() - index_start

        file_paths = [info[0] for info in file_info]
        timing_info['files_queried'] = len(file_paths)

        if not file_paths:
            # No data for this label
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            timing_info['sql_query'] = "SELECT DISTINCT file_path FROM point_cloud_index WHERE label = ?"
            return empty_result, timing_info

        # Get detailed file information
        for file_path in file_paths:
            file_detail = {'file_path': file_path}
            if os.path.exists(file_path):
                file_detail['file_size_mb'] = os.path.getsize(file_path) / (1024 * 1024)
            else:
                file_detail['file_size_mb'] = 0.0
            timing_info['file_details'].append(file_detail)

        # Use the most reliable query to get the point data
        query = f"""
            SELECT data
            FROM parquet_scan([{','.join(f"'{path}'" for path in file_paths)}])
            WHERE label = {label}
        """
        timing_info['sql_query'] = query

        # Execute query and get results as Pandas DataFrame
        data_start = time.time()
        df = self.db_connection.execute(query).fetchdf()
        timing_info['data_read_time'] = time.time() - data_start

        # Handle empty result
        if len(df) == 0:
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            timing_info['query_efficiency']['points_per_file'] = 0.0
            return empty_result, timing_info

        # Process the data
        processing_start = time.time()
        points_list = df['data'].tolist()

        if not points_list:
            empty_result = np.array([], dtype=np.int64)
            if use_cache and self.cache_size > 0:
                self._update_cache(label, empty_result)
            timing_info['processing_time'] = time.time() - processing_start
            timing_info['query_efficiency']['points_per_file'] = 0.0
            return empty_result, timing_info

        # Stack the points
        points = np.vstack(points_list).astype(np.int64)

        # Remove duplicates to ensure we only have unique points
        points = np.unique(points, axis=0)
        timing_info['processing_time'] = time.time() - processing_start

        # Get actual points per file by querying each file individually
        points_per_file_counts = []
        individual_read_start = time.time()
        
        for file_path in file_paths:
            file_query = f"SELECT COUNT(*) FROM parquet_scan('{file_path}') WHERE label = {label}"
            file_count_result = self.db_connection.execute(file_query).fetchone()
            file_points = file_count_result[0] if file_count_result[0] is not None else 0
            points_per_file_counts.append(file_points)
        
        individual_query_time = time.time() - individual_read_start

        # Calculate efficiency metrics
        total_points = len(points)
        timing_info['query_efficiency'] = {
            'total_points': total_points,
            'files_accessed': len(file_paths),
            'points_per_file_range': {
                'min': min(points_per_file_counts) if points_per_file_counts else 0,
                'max': max(points_per_file_counts) if points_per_file_counts else 0,
                'avg': sum(points_per_file_counts) / len(points_per_file_counts) if points_per_file_counts else 0.0
            },
            'points_per_file_counts': points_per_file_counts,
            'individual_file_query_time': individual_query_time,
            'deduplication_applied': True
        }

        # Update file details with actual point counts
        for i, file_detail in enumerate(timing_info['file_details']):
            if i < len(points_per_file_counts):
                file_detail['actual_points_in_file'] = points_per_file_counts[i]
            file_detail['read_time'] = timing_info['data_read_time'] / len(file_paths)

        # Cache the result if enabled
        if use_cache and self.cache_size > 0:
            self._update_cache(label, points)

        return points, timing_info

    def _get_points_original(self, label: int, use_cache: bool = True) -> np.ndarray:
        """
        Original implementation of get_points, used as fallback when optimized data is not available.

        Args:
            label: The label to query for.
            use_cache: Whether to use the in-memory point cloud cache (if enabled).

        Returns:
            Numpy array containing all point data for the label.
        """
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

    @staticmethod
    def print_timing_info(timing_info: Dict) -> None:
        """
        Pretty print timing information to stdout.
        
        Args:
            timing_info: Timing information dictionary returned from query methods.
        """
        # Extract label and points count from timing info
        label = timing_info.get('label', 'Unknown')
        points_count = timing_info.get('points_returned', 0)
        
        print(f"\n🧠 QUERY TIMING ANALYSIS - Label {label}")
        print("=" * 60)
        
        # Basic timing info
        total_time = timing_info.get('total_time', 0)
        print(f"📊 Points retrieved: {points_count:,}")
        print(f"⏱️  Total query time: {total_time:.4f}s")
        
        if points_count > 0 and total_time > 0:
            points_per_sec = points_count / total_time
            print(f"🚀 Points per second: {points_per_sec:,.0f}")
        
        # Cache and optimization status
        cache_hit = timing_info.get('cache_hit', False)
        using_optimized = timing_info.get('using_optimized_data', False)
        print(f"💾 Cache hit: {'✅ Yes' if cache_hit else '❌ No'}")
        print(f"⚡ Using optimized data: {'✅ Yes' if using_optimized else '❌ No'}")
        
        # Time breakdown
        print("\n📈 TIME BREAKDOWN")
        if 'cache_lookup_time' in timing_info:
            print(f"   Cache lookup: {timing_info['cache_lookup_time']*1000:.2f}ms")
        if 'index_lookup_time' in timing_info:
            print(f"   Index lookup: {timing_info['index_lookup_time']*1000:.2f}ms")
        if 'data_read_time' in timing_info:
            print(f"   Data read: {timing_info['data_read_time']*1000:.2f}ms")
        if 'processing_time' in timing_info:
            print(f"   Processing: {timing_info['processing_time']*1000:.2f}ms")
        
        # File access patterns
        files_queried = timing_info.get('files_queried', 0)
        print(f"\n📁 FILE ACCESS")
        print(f"   Files queried: {files_queried}")
        
        # Query efficiency metrics
        if 'query_efficiency' in timing_info:
            efficiency = timing_info['query_efficiency']
            files_accessed = efficiency.get('files_accessed', files_queried)
            points_range = efficiency.get('points_per_file_range', {})
            
            print(f"   Files accessed: {files_accessed}")
            
            if points_range:
                min_points = points_range.get('min', 0)
                max_points = points_range.get('max', 0)
                avg_points = points_range.get('avg', 0)
                
                print(f"   Points per file range:")
                print(f"     Min: {min_points:,}")
                print(f"     Max: {max_points:,}")
                print(f"     Avg: {avg_points:,.1f}")
                
                # Show individual file counts if reasonable number
                file_counts = efficiency.get('points_per_file_counts', [])
                if file_counts and len(file_counts) <= 10:
                    print(f"   Individual file point counts: {file_counts}")
                elif file_counts and len(file_counts) > 10:
                    print(f"   Individual file point counts: {file_counts[:5]} ... {file_counts[-5:]} ({len(file_counts)} total)")
            
            individual_query_time = efficiency.get('individual_file_query_time', 0)
            if individual_query_time > 0:
                print(f"   Individual file query time: {individual_query_time*1000:.2f}ms")
        
        # File details summary
        file_details = timing_info.get('file_details', [])
        if file_details:
            total_size_mb = sum(f.get('file_size_mb', 0) for f in file_details)
            print(f"\n📋 FILE DETAILS")
            print(f"   Total file size: {total_size_mb:.2f} MB")
            
            if len(file_details) <= 5:
                for i, file_detail in enumerate(file_details, 1):
                    file_path = file_detail.get('file_path', 'Unknown')
                    file_name = os.path.basename(file_path)
                    size_mb = file_detail.get('file_size_mb', 0)
                    actual_points = file_detail.get('actual_points_in_file', 0)
                    print(f"   File {i}: {file_name}")
                    print(f"     Size: {size_mb:.2f} MB")
                    print(f"     Points: {actual_points:,}")
            else:
                print(f"   ({len(file_details)} files total)")
        
        print("=" * 60)