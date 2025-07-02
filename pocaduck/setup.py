"""
Setup utilities for PoCADuck storage backends.

This module provides utilities for initializing storage backends that require
setup, such as VastDB schema and table creation.
"""

import logging
from typing import Optional

try:
    import vastdb
    import pyarrow as pa
    VASTDB_AVAILABLE = True
except ImportError:
    VASTDB_AVAILABLE = False

from .storage_config import StorageConfig

logger = logging.getLogger(__name__)


def setup_vastdb(
    storage_config: StorageConfig,
    drop_if_exists: bool = False
) -> None:
    """
    Initialize VastDB schema and table for PoCADuck.
    
    Creates a minimal table structure optimized for label-based queries:
    - label: uint64 identifier for point cloud
    - block_id: string identifier for the block 
    - points: list of int64 coordinates (flattened, variable dimensions - 3D, 4D, etc.)
    
    Args:
        storage_config: Storage configuration with VastDB parameters
        drop_if_exists: Whether to drop existing table/schema if they exist
        
    Example:
        >>> config = StorageConfig(
        ...     storage_type="vastdb",
        ...     vastdb_endpoint="http://vast.example.com",
        ...     vastdb_access_key="access_key",
        ...     vastdb_secret_key="secret_key", 
        ...     vastdb_bucket="my-bucket",
        ...     vastdb_schema="pocaduck",
        ...     vastdb_table="point_clouds"
        ... )
        >>> setup_vastdb(config)
    """
    if not VASTDB_AVAILABLE:
        raise ImportError("VastDB SDK is not available. Please install vastdb package.")
    
    if storage_config.storage_type != "vastdb":
        raise ValueError(f"Storage type must be 'vastdb', got '{storage_config.storage_type}'")
    
    logger.info(f"Setting up VastDB backend: {storage_config.vastdb_endpoint}")
    
    # Connect to VastDB
    session = vastdb.connect(
        endpoint=storage_config.vastdb_endpoint,
        access=storage_config.vastdb_access_key,
        secret=storage_config.vastdb_secret_key
    )
    
    bucket = session.bucket(storage_config.vastdb_bucket)
    
    # Create or get schema
    try:
        schema = bucket.schema(storage_config.vastdb_schema)
        logger.info(f"Using existing schema: {storage_config.vastdb_schema}")
    except Exception:
        logger.info(f"Creating schema: {storage_config.vastdb_schema}")
        schema = bucket.create_schema(storage_config.vastdb_schema)
    
    # Handle existing table
    table_exists = False
    try:
        table = schema.table(storage_config.vastdb_table)
        table_exists = True
        logger.info(f"Table {storage_config.vastdb_table} already exists")
        
        if drop_if_exists:
            logger.info(f"Dropping existing table: {storage_config.vastdb_table}")
            schema.drop_table(storage_config.vastdb_table)
            table_exists = False
    except Exception:
        table_exists = False
    
    # Create table if it doesn't exist
    if not table_exists:
        logger.info(f"Creating table: {storage_config.vastdb_table}")
        
        # Define Arrow schema for point clouds
        # Compatible with existing parquet+DuckDB but using Arrow types
        # points: variable dimension flattened coordinates (3D, 4D with supervoxel, etc.)
        table_schema = pa.schema([
            ('label', pa.uint64()),
            ('block_id', pa.string()),
            ('points', pa.list(pa.int64())),
        ])
        
        table = schema.create_table(storage_config.vastdb_table, table_schema)
        logger.info(f"Created table with schema: {table_schema}")
        
        # Create sorting optimization for label queries (MVP: label-only queries)
        logger.info("Creating label-sorted projection for optimal query performance")
        projection = table.create_projection(
            'label_sorted',
            sorted=['label'],
            unsorted=['block_id', 'points']
        )
        logger.info("Created label_sorted projection")
    
    logger.info("VastDB setup completed successfully")


def setup_backend(storage_config: StorageConfig, **kwargs) -> None:
    """
    Setup any backend that requires initialization.
    
    This function routes to the appropriate setup function based on storage type.
    File-based backends (local, s3, gcs, azure) require no setup.
    
    Args:
        storage_config: Storage configuration
        **kwargs: Additional arguments passed to backend-specific setup functions
    """
    if storage_config.storage_type == "vastdb":
        setup_vastdb(storage_config, **kwargs)
    elif storage_config.storage_type in ["local", "s3", "gcs", "azure"]:
        # File-based backends require no setup
        logger.info(f"No setup required for {storage_config.storage_type} backend")
        pass
    else:
        raise ValueError(f"Unsupported storage type: {storage_config.storage_type}")


def validate_vastdb_setup(storage_config: StorageConfig) -> bool:
    """
    Validate that VastDB backend is properly set up.
    
    Args:
        storage_config: Storage configuration with VastDB parameters
        
    Returns:
        True if setup is valid, False otherwise
    """
    if not VASTDB_AVAILABLE:
        logger.error("VastDB SDK is not available")
        return False
    
    if storage_config.storage_type != "vastdb":
        logger.error(f"Storage type must be 'vastdb', got '{storage_config.storage_type}'")
        return False
    
    try:
        # Test connection and table access
        session = vastdb.connect(
            endpoint=storage_config.vastdb_endpoint,
            access=storage_config.vastdb_access_key,
            secret=storage_config.vastdb_secret_key
        )
        
        bucket = session.bucket(storage_config.vastdb_bucket)
        schema = bucket.schema(storage_config.vastdb_schema)
        table = schema.table(storage_config.vastdb_table)
        
        # Try a simple query to verify table structure
        result = table.select(columns=['label'], limit=1)
        logger.info("VastDB setup validation successful")
        return True
        
    except Exception as e:
        logger.error(f"VastDB setup validation failed: {e}")
        return False