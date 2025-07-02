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

# VastDB imports (optional - only loaded if needed)
try:
    import vastdb
    import pyarrow as pa
    VASTDB_AVAILABLE = True
except ImportError:
    VASTDB_AVAILABLE = False

# Default thread count to use when os.cpu_count() fails
DEFAULT_THREAD_COUNT = 8


class VastDBQueryBackend:
    """
    VastDB-based backend for point cloud queries.
    
    This backend uses VastDB as both storage and query engine, eliminating the need
    for file-based sharding and complex optimization pipelines.
    """
    
    def __init__(self, storage_config: StorageConfig, **kwargs):
        """
        Initialize VastDB connection.
        
        Args:
            storage_config: Storage configuration with VastDB parameters
            **kwargs: Additional arguments (ignored for VastDB)
        """
        if not VASTDB_AVAILABLE:
            raise ImportError("VastDB SDK is not available. Please install vastdb package.")
        
        self.storage_config = storage_config
        
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
    
    def setup(self, **kwargs):
        """Setup VastDB backend (create schema and table if needed)."""
        from .setup import setup_vastdb
        setup_vastdb(self.storage_config, **kwargs)
    
    def get_labels(self) -> np.ndarray:
        """Get all available labels."""
        result = self.table.select(columns=['label'])
        if len(result) == 0:
            return np.array([], dtype=np.uint64)
        return result['label'].to_numpy().astype(np.uint64)
    
    def get_point_count(self, label: int, timing: bool = False) -> Union[int, Tuple[int, Dict]]:
        """Get point count for a label."""
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            result = self.table.select(
                columns=['point_count'],
                predicate=(self.table.label == label)
            )
            return result['point_count'][0] if len(result) > 0 else 0
        
        # Timing version
        start_time = time.time()
        query_start = time.time()
        result = self.table.select(
            columns=['point_count'],
            predicate=(self.table.label == label)
        )
        query_time = time.time() - query_start
        
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'query_time': query_time,
            'query_type': 'vastdb_point_count',
            'backend': 'vastdb',
            'files_accessed': 1,
            'using_optimized_data': True,  # VastDB is inherently optimized
            'sql_query': f"SELECT point_count FROM {self.storage_config.vastdb_table} WHERE label = {label}"
        }
        
        count = result['point_count'][0] if len(result) > 0 else 0
        return count, timing_info
    
    def get_blocks_for_label(self, label: int, timing: bool = False) -> Union[List[str], Tuple[List[str], Dict]]:
        """Get blocks for label (not applicable for VastDB)."""
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            return []  # No concept of blocks in VastDB
        
        timing_info = {
            'label': label,
            'total_time': 0.0,
            'query_type': 'vastdb_blocks',
            'backend': 'vastdb',
            'blocks_found': 0,
            'using_optimized_data': True,
            'note': 'Block concept not applicable for VastDB backend'
        }
        return [], timing_info
    
    def get_points(self, label: int, use_cache: bool = True, timing: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Get all points for a label."""
        # Convert numpy.uint64 to int if necessary
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            result = self.table.select(
                columns=['points'],
                predicate=(self.table.label == label)
            )
            if len(result) == 0:
                return np.array([], dtype=np.int64).reshape(0, -1)  # n-dimensional empty array
            
            points_list = result['points'][0]  # Get first (and only) row
            if points_list is None or len(points_list) == 0:
                return np.array([], dtype=np.int64).reshape(0, -1)  # n-dimensional empty array
            
            # Points are stored flattened, need to reshape back to (N, D) format
            # Assume common dimensions (3D or 4D), but handle variable dimensions
            points_array = np.array(points_list, dtype=np.int64)
            if len(points_array) == 0:
                return points_array.reshape(0, -1)
            
            # Try to determine dimension from point count
            # Common cases: 3D (x,y,z) or 4D (x,y,z,supervoxel)
            total_points = len(points_array)
            if total_points % 3 == 0:
                # Assume 3D points
                return points_array.reshape(-1, 3)
            elif total_points % 4 == 0:
                # Assume 4D points
                return points_array.reshape(-1, 4)
            else:
                # Fall back to original flattened array
                return points_array.reshape(-1, 1)
        
        # Timing version
        start_time = time.time()
        query_start = time.time()
        result = self.table.select(
            columns=['points'],
            predicate=(self.table.label == label)
        )
        query_time = time.time() - query_start
        
        processing_start = time.time()
        if len(result) == 0:
            points = np.array([], dtype=np.int64).reshape(0, -1)  # n-dimensional empty array
        else:
            points_list = result['points'][0]
            if points_list is None or len(points_list) == 0:
                points = np.array([], dtype=np.int64).reshape(0, -1)  # n-dimensional empty array
            else:
                # Points are stored flattened, need to reshape back to (N, D) format
                points_array = np.array(points_list, dtype=np.int64)
                if len(points_array) == 0:
                    points = points_array.reshape(0, -1)
                else:
                    # Try to determine dimension from point count
                    total_points = len(points_array)
                    if total_points % 3 == 0:
                        # Assume 3D points
                        points = points_array.reshape(-1, 3)
                    elif total_points % 4 == 0:
                        # Assume 4D points
                        points = points_array.reshape(-1, 4)
                    else:
                        # Fall back to original flattened array
                        points = points_array.reshape(-1, 1)
        processing_time = time.time() - processing_start
        
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'query_time': query_time,
            'processing_time': processing_time,
            'query_type': 'vastdb_points',
            'backend': 'vastdb',
            'using_optimized_data': True,
            'files_accessed': 1,
            'points_returned': len(points),
            'cache_hit': False,  # VastDB handles its own caching
            'query_efficiency': {
                'total_points': len(points),
                'files_accessed': 1,
                'points_per_file_range': {
                    'min': len(points),
                    'max': len(points),
                    'avg': float(len(points))
                },
                'points_per_file_counts': [len(points)] if len(points) > 0 else []
            },
            'file_details': [{
                'file_path': f'vastdb://{self.storage_config.vastdb_bucket}/{self.storage_config.vastdb_schema}/{self.storage_config.vastdb_table}',
                'file_size_mb': 0.0,  # Not applicable for VastDB
                'actual_points_in_file': len(points),
                'read_time': query_time
            }],
            'sql_query': f"SELECT points FROM {self.storage_config.vastdb_table} WHERE label = {label}"
        }
        
        return points, timing_info
    
    def close(self):
        """Close VastDB connection."""
        if hasattr(self, 'session') and self.session:
            # VastDB connections are typically managed automatically
            # but we can explicitly clean up if needed
            self.session = None


class ParquetDuckDBQueryBackend:
    """
    Parquet + DuckDB backend for point cloud queries.
    
    This is the file-based implementation using parquet files for storage
    and DuckDB for indexing and queries. Fully featured with optimization
    support, caching, and comprehensive timing analysis.
    """
    
    def __init__(
        self,
        storage_config: StorageConfig,
        index_path: Optional[str] = None,
        threads: Optional[int] = None,
        cache_size: int = 10
    ):
        """
        Initialize the parquet + DuckDB backend.

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
    
    def setup(self, **kwargs):
        """Setup method for file-based backend (no-op)."""
        pass
        
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
    
    def _initialize_db_connection(self):
        """Initialize DuckDB connection with storage configuration."""
        con = duckdb.connect(self.index_path, read_only=True)
        
        # Configure DuckDB for storage backend
        duckdb_config = self.storage_config.get_duckdb_config()
        for key, value in duckdb_config.items():
            con.execute(f"SET {key}='{value}'")
        
        # Set thread count for parallel processing
        con.execute(f"PRAGMA threads={self.threads}")
        
        return con
    
    def get_labels(self) -> np.ndarray:
        """Get all unique labels available in the database."""
        result = self.db_connection.execute("SELECT DISTINCT label FROM point_cloud_index").fetchall()
        return np.array([r[0] for r in result], dtype=np.uint64)
    
    def get_point_count(self, label: int, timing: bool = False) -> Union[int, Tuple[int, Dict]]:
        """Get the total number of points for a specific label."""
        # Convert numpy.uint64 to int if necessary for DuckDB compatibility
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            result = self.db_connection.execute(
                "SELECT SUM(point_count) FROM point_cloud_index WHERE label = ?",
                [label]
            ).fetchone()
            return result[0] if result[0] is not None else 0
        
        # Timing version
        start_time = time.time()
        
        # Index lookup timing
        index_start = time.time()
        result = self.db_connection.execute(
            "SELECT SUM(point_count) FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchone()
        index_lookup_time = time.time() - index_start
        
        count = result[0] if result[0] is not None else 0
        
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'index_lookup_time': index_lookup_time,
            'query_type': 'parquet_duckdb_point_count',
            'backend': 'parquet_duckdb',
            'using_optimized_data': self.using_optimized_data,
            'points_returned': count
        }
        
        return count, timing_info
    
    def get_blocks_for_label(self, label: int, timing: bool = False) -> Union[List[str], Tuple[List[str], Dict]]:
        """Get all blocks that contain a specific label."""
        # Convert numpy.uint64 to int if necessary for DuckDB compatibility
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            if self.using_optimized_data:
                # Optimized data doesn't have blocks in the traditional sense
                return []
            
            result = self.db_connection.execute(
                "SELECT DISTINCT block_id FROM point_cloud_index WHERE label = ?",
                [label]
            ).fetchall()
            return [r[0] for r in result]
        
        # Timing version
        start_time = time.time()
        
        if self.using_optimized_data:
            blocks = []
        else:
            index_start = time.time()
            result = self.db_connection.execute(
                "SELECT DISTINCT block_id FROM point_cloud_index WHERE label = ?",
                [label]
            ).fetchall()
            index_lookup_time = time.time() - index_start
            blocks = [r[0] for r in result]
        
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'index_lookup_time': index_lookup_time if not self.using_optimized_data else 0.0,
            'query_type': 'parquet_duckdb_blocks',
            'backend': 'parquet_duckdb',
            'using_optimized_data': self.using_optimized_data,
            'blocks_found': len(blocks)
        }
        
        return blocks, timing_info
    
    def get_points(self, label: int, use_cache: bool = True, timing: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Get all point data for a specific label."""
        # Convert numpy.uint64 to int if necessary for DuckDB compatibility
        if isinstance(label, np.integer):
            label = int(label)
        
        if not timing:
            if self.using_optimized_data:
                return self._get_points_optimized(label, use_cache)
            else:
                return self._get_points_original(label, use_cache)
        else:
            if self.using_optimized_data:
                return self._get_points_optimized_with_timing(label, use_cache)
            else:
                return self._get_points_original_with_timing(label, use_cache)
    
    def _get_points_optimized(self, label: int, use_cache: bool = True) -> np.ndarray:
        """Get points from optimized data structure (without timing)."""
        # Check cache first if enabled
        if use_cache and label in self._points_cache:
            return self._points_cache[label]
        
        # Query the optimized index
        result = self.db_connection.execute(
            "SELECT file_path FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        
        if not result:
            return np.array([], dtype=np.int64).reshape(0, -1)  # n-dimensional empty array
        
        points_list = []
        for row in result:
            file_path = row[0]
            
            # Read the parquet file and filter for this label
            df = pd.read_parquet(file_path)
            label_points = df[df['label'] == label][['x', 'y', 'z']].values
            if len(label_points) > 0:
                points_list.append(label_points)
        
        if not points_list:
            final_points = np.array([], dtype=np.int64).reshape(0, -1)
        else:
            final_points = np.vstack(points_list).astype(np.int64)
        
        # Update cache
        self._update_cache(label, final_points)
        
        return final_points
    
    def _get_points_original(self, label: int, use_cache: bool = True) -> np.ndarray:
        """Get points from original data structure (without timing)."""
        # Check cache first if enabled
        if use_cache and label in self._points_cache:
            return self._points_cache[label]
        
        # Get file paths and metadata for this label
        result = self.db_connection.execute("""
            SELECT file_path, block_id, point_count
            FROM point_cloud_index 
            WHERE label = ?
        """, [label]).fetchall()
        
        if not result:
            return np.array([], dtype=np.int64).reshape(0, -1)  # n-dimensional empty array
        
        points_list = []
        for row in result:
            file_path, block_id, _ = row
            
            # Read the parquet file and filter for this label in this block
            df = pd.read_parquet(file_path)
            label_block_points = df[(df['label'] == label) & (df['block_id'] == block_id)]
            
            if len(label_block_points) > 0:
                # Extract coordinate columns (x, y, z and any additional dimensions)
                coord_cols = [col for col in df.columns if col not in ['label', 'block_id']]
                points = label_block_points[coord_cols].values
                points_list.append(points)
        
        if not points_list:
            final_points = np.array([], dtype=np.int64).reshape(0, -1)
        else:
            # Stack all points and remove duplicates
            stacked_points = np.vstack(points_list).astype(np.int64)
            final_points = np.unique(stacked_points, axis=0)
        
        # Update cache
        self._update_cache(label, final_points)
        
        return final_points
    
    def _get_points_optimized_with_timing(self, label: int, use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
        """Get points from optimized data with detailed timing."""
        start_time = time.time()
        
        # Cache lookup timing
        cache_start = time.time()
        cache_hit = use_cache and label in self._points_cache
        if cache_hit:
            cached_points = self._points_cache[label]
            cache_lookup_time = time.time() - cache_start
            
            timing_info = {
                'label': label,
                'total_time': time.time() - start_time,
                'cache_lookup_time': cache_lookup_time,
                'cache_hit': True,
                'using_optimized_data': True,
                'points_returned': len(cached_points),
                'query_type': 'parquet_duckdb_optimized',
                'backend': 'parquet_duckdb'
            }
            return cached_points, timing_info
        
        cache_lookup_time = time.time() - cache_start
        
        # Index lookup timing
        index_start = time.time()
        result = self.db_connection.execute(
            "SELECT file_path FROM point_cloud_index WHERE label = ?",
            [label]
        ).fetchall()
        index_lookup_time = time.time() - index_start
        
        if not result:
            empty_points = np.array([], dtype=np.int64).reshape(0, -1)
            timing_info = {
                'label': label,
                'total_time': time.time() - start_time,
                'cache_lookup_time': cache_lookup_time,
                'index_lookup_time': index_lookup_time,
                'cache_hit': False,
                'using_optimized_data': True,
                'points_returned': 0,
                'files_accessed': 0,
                'query_type': 'parquet_duckdb_optimized',
                'backend': 'parquet_duckdb'
            }
            return empty_points, timing_info
        
        # Data reading timing
        data_read_start = time.time()
        points_list = []
        file_details = []
        
        for row in result:
            file_path = row[0]
            file_read_start = time.time()
            
            # Read the parquet file and filter for this label
            df = pd.read_parquet(file_path)
            label_points = df[df['label'] == label]
            
            if len(label_points) > 0:
                # Extract coordinate columns
                coord_cols = [col for col in df.columns if col not in ['label', 'block_id']]
                points = label_points[coord_cols].values
                points_list.append(points)
            
            file_read_time = time.time() - file_read_start
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
            
            file_details.append({
                'file_path': file_path,
                'file_size_mb': file_size_mb,
                'actual_points_in_file': len(label_points),
                'read_time': file_read_time
            })
        
        data_read_time = time.time() - data_read_start
        
        # Processing timing
        processing_start = time.time()
        if not points_list:
            final_points = np.array([], dtype=np.int64).reshape(0, -1)
        else:
            final_points = np.vstack(points_list).astype(np.int64)
        
        # Update cache
        self._update_cache(label, final_points)
        processing_time = time.time() - processing_start
        
        # Build timing info
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'cache_lookup_time': cache_lookup_time,
            'index_lookup_time': index_lookup_time,
            'data_read_time': data_read_time,
            'processing_time': processing_time,
            'cache_hit': False,
            'using_optimized_data': True,
            'points_returned': len(final_points),
            'files_accessed': len(result),
            'query_type': 'parquet_duckdb_optimized',
            'backend': 'parquet_duckdb',
            'file_details': file_details,
            'query_efficiency': {
                'total_points': len(final_points),
                'files_accessed': len(result),
                'points_per_file_range': {
                    'min': min([f['actual_points_in_file'] for f in file_details]) if file_details else 0,
                    'max': max([f['actual_points_in_file'] for f in file_details]) if file_details else 0,
                    'avg': np.mean([f['actual_points_in_file'] for f in file_details]) if file_details else 0
                },
                'points_per_file_counts': [f['actual_points_in_file'] for f in file_details]
            }
        }
        
        return final_points, timing_info
    
    def _get_points_original_with_timing(self, label: int, use_cache: bool = True) -> Tuple[np.ndarray, Dict]:
        """Get points from original data with detailed timing."""
        start_time = time.time()
        
        # Cache lookup timing
        cache_start = time.time()
        cache_hit = use_cache and label in self._points_cache
        if cache_hit:
            cached_points = self._points_cache[label]
            cache_lookup_time = time.time() - cache_start
            
            timing_info = {
                'label': label,
                'total_time': time.time() - start_time,
                'cache_lookup_time': cache_lookup_time,
                'cache_hit': True,
                'using_optimized_data': False,
                'points_returned': len(cached_points),
                'query_type': 'parquet_duckdb_original',
                'backend': 'parquet_duckdb'
            }
            return cached_points, timing_info
        
        cache_lookup_time = time.time() - cache_start
        
        # Index lookup timing
        index_start = time.time()
        result = self.db_connection.execute("""
            SELECT file_path, block_id, point_count
            FROM point_cloud_index 
            WHERE label = ?
        """, [label]).fetchall()
        index_lookup_time = time.time() - index_start
        
        if not result:
            empty_points = np.array([], dtype=np.int64).reshape(0, -1)
            timing_info = {
                'label': label,
                'total_time': time.time() - start_time,
                'cache_lookup_time': cache_lookup_time,
                'index_lookup_time': index_lookup_time,
                'cache_hit': False,
                'using_optimized_data': False,
                'points_returned': 0,
                'files_accessed': 0,
                'query_type': 'parquet_duckdb_original',
                'backend': 'parquet_duckdb'
            }
            return empty_points, timing_info
        
        # Data reading timing
        data_read_start = time.time()
        points_list = []
        file_details = []
        files_processed = set()
        
        for row in result:
            file_path, block_id, _ = row
            
            # Avoid re-reading the same file multiple times
            if file_path not in files_processed:
                file_read_start = time.time()
                
                # Read the parquet file
                df = pd.read_parquet(file_path)
                files_processed.add(file_path)
                
                # Filter for this label in any block within this file
                label_points = df[df['label'] == label]
                
                if len(label_points) > 0:
                    # Extract coordinate columns (x, y, z and any additional dimensions)
                    coord_cols = [col for col in df.columns if col not in ['label', 'block_id']]
                    points = label_points[coord_cols].values
                    points_list.append(points)
                
                file_read_time = time.time() - file_read_start
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
                
                file_details.append({
                    'file_path': file_path,
                    'file_size_mb': file_size_mb,
                    'actual_points_in_file': len(label_points),
                    'read_time': file_read_time
                })
        
        data_read_time = time.time() - data_read_start
        
        # Processing timing (includes deduplication)
        processing_start = time.time()
        if not points_list:
            final_points = np.array([], dtype=np.int64).reshape(0, -1)
            deduplication_applied = False
        else:
            # Stack all points and remove duplicates
            stacked_points = np.vstack(points_list).astype(np.int64)
            pre_dedup_count = len(stacked_points)
            final_points = np.unique(stacked_points, axis=0)
            deduplication_applied = len(final_points) < pre_dedup_count
        
        # Update cache
        self._update_cache(label, final_points)
        processing_time = time.time() - processing_start
        
        # Build timing info
        timing_info = {
            'label': label,
            'total_time': time.time() - start_time,
            'cache_lookup_time': cache_lookup_time,
            'index_lookup_time': index_lookup_time,
            'data_read_time': data_read_time,
            'processing_time': processing_time,
            'cache_hit': False,
            'using_optimized_data': False,
            'points_returned': len(final_points),
            'files_accessed': len(files_processed),
            'deduplication_applied': deduplication_applied,
            'query_type': 'parquet_duckdb_original',
            'backend': 'parquet_duckdb',
            'file_details': file_details,
            'query_efficiency': {
                'total_points': len(final_points),
                'files_accessed': len(files_processed),
                'points_per_file_range': {
                    'min': min([f['actual_points_in_file'] for f in file_details]) if file_details else 0,
                    'max': max([f['actual_points_in_file'] for f in file_details]) if file_details else 0,
                    'avg': np.mean([f['actual_points_in_file'] for f in file_details]) if file_details else 0
                },
                'points_per_file_counts': [f['actual_points_in_file'] for f in file_details]
            }
        }
        
        return final_points, timing_info
    
    def _update_cache(self, label: int, points: np.ndarray):
        """Update the point cloud cache with LRU eviction."""
        if self.cache_size <= 0:
            return
        
        # Remove oldest entries if cache is full
        while len(self._points_cache) >= self.cache_size:
            # Remove the first (oldest) entry
            oldest_label = next(iter(self._points_cache))
            del self._points_cache[oldest_label]
        
        # Add new entry (it becomes the newest)
        self._points_cache[label] = points
    
    def close(self):
        """Close the query connection and clear caches."""
        if hasattr(self, '_points_cache'):
            self._points_cache.clear()
        if hasattr(self, 'db_connection') and self.db_connection is not None:
            self.db_connection.close()
            self.db_connection = None


class Query:
    """
    Unified interface for querying point clouds using different backends.

    The Query class provides a consistent API for retrieving 3D point clouds for labels
    across different storage backends:
    - VastDB: Direct SQL-based key-value storage
    - Parquet+DuckDB: File-based storage with optimization support

    The backend is automatically selected based on the storage_config.storage_type.

    Attributes:
        storage_config: Configuration for storage backend
        _backend: The actual backend implementation (VastDB or Parquet+DuckDB)
    """
    
    def __init__(
        self,
        storage_config: StorageConfig,
        index_path: Optional[str] = None,
        threads: Optional[int] = None,
        cache_size: int = 10
    ):
        """
        Initialize a Query instance with the appropriate backend.

        Args:
            storage_config: Configuration for storage backend
            index_path: Path to the unified index (only used for Parquet+DuckDB backend)
            threads: Number of threads to use (only used for Parquet+DuckDB backend)
            cache_size: Number of label point clouds to cache (only used for Parquet+DuckDB backend)
        """
        self.storage_config = storage_config
        
        # Select the appropriate backend based on storage type
        if storage_config.storage_type == "vastdb":
            self._backend = VastDBQueryBackend(storage_config)
        elif storage_config.storage_type in ["local", "s3", "gcs", "azure"]:
            # Use the file-based Parquet+DuckDB backend
            self._backend = ParquetDuckDBQueryBackend(storage_config, index_path, threads, cache_size)
        else:
            raise ValueError(f"Unsupported storage type: {storage_config.storage_type}")
    
    # Delegate all methods to the backend
    def get_labels(self) -> np.ndarray:
        """Get all available labels."""
        return self._backend.get_labels()
    
    def get_point_count(self, label: int, timing: bool = False) -> Union[int, Tuple[int, Dict]]:
        """Get the total number of points for a specific label."""
        return self._backend.get_point_count(label, timing)
    
    def get_blocks_for_label(self, label: int, timing: bool = False) -> Union[List[str], Tuple[List[str], Dict]]:
        """Get all blocks that contain a specific label."""
        return self._backend.get_blocks_for_label(label, timing)
    
    def get_points(self, label: int, use_cache: bool = True, timing: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """Get all point data for a specific label."""
        return self._backend.get_points(label, use_cache, timing)
    
    def setup(self, **kwargs):
        """Setup the backend (create schema/tables if needed)."""
        return self._backend.setup(**kwargs)
    
    def close(self):
        """Close the query connection and clear caches."""
        if hasattr(self, '_backend') and self._backend:
            self._backend.close()
            self._backend = None

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