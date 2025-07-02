#!/usr/bin/env python3
"""
VastDB Setup Example for PoCADuck

This script demonstrates how to set up VastDB backend for PoCADuck point cloud storage.
It creates the necessary schema, table, and projections for optimal query performance.

Usage:
    python vastdb_setup.py --endpoint http://vast.example.com --access-key YOUR_ACCESS_KEY --secret-key YOUR_SECRET_KEY
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import pocaduck
sys.path.insert(0, str(Path(__file__).parent.parent))

from pocaduck.storage_config import StorageConfig
from pocaduck.setup import setup_vastdb, validate_vastdb_setup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description='Set up VastDB backend for PoCADuck',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic setup
  python vastdb_setup.py --endpoint http://vast.example.com \\
                        --access-key YOUR_ACCESS_KEY \\
                        --secret-key YOUR_SECRET_KEY

  # Custom bucket and schema
  python vastdb_setup.py --endpoint http://vast.example.com \\
                        --access-key YOUR_ACCESS_KEY \\
                        --secret-key YOUR_SECRET_KEY \\
                        --bucket my-bucket \\
                        --schema neuroscience \\
                        --table point_clouds

  # Drop existing table and recreate
  python vastdb_setup.py --endpoint http://vast.example.com \\
                        --access-key YOUR_ACCESS_KEY \\
                        --secret-key YOUR_SECRET_KEY \\
                        --drop-if-exists
        """
    )
    
    # Required VastDB connection parameters
    parser.add_argument('--endpoint', required=True,
                       help='VastDB endpoint URL (e.g., http://vast.example.com)')
    parser.add_argument('--access-key', required=True,
                       help='VastDB access key')
    parser.add_argument('--secret-key', required=True,
                       help='VastDB secret key')
    
    # Optional VastDB parameters
    parser.add_argument('--bucket', default='pocaduck',
                       help='VastDB bucket name (default: pocaduck)')
    parser.add_argument('--schema', default='default',
                       help='VastDB schema name (default: default)')
    parser.add_argument('--table', default='point_clouds',
                       help='VastDB table name (default: point_clouds)')
    
    # Setup options
    parser.add_argument('--drop-if-exists', action='store_true',
                       help='Drop existing table and schema if they exist')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing setup, do not create')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create storage configuration
    storage_config = StorageConfig(
        storage_type="vastdb",
        vastdb_endpoint=args.endpoint,
        vastdb_access_key=args.access_key,
        vastdb_secret_key=args.secret_key,
        vastdb_bucket=args.bucket,
        vastdb_schema=args.schema,
        vastdb_table=args.table
    )
    
    logger.info("=== PoCADuck VastDB Setup ===")
    logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Schema: {args.schema}")
    logger.info(f"Table: {args.table}")
    
    try:
        if args.validate_only:
            logger.info("Validating existing VastDB setup...")
            if validate_vastdb_setup(storage_config):
                logger.info("✅ VastDB setup is valid")
                return 0
            else:
                logger.error("❌ VastDB setup validation failed")
                return 1
        
        logger.info("Setting up VastDB backend...")
        setup_vastdb(storage_config, drop_if_exists=args.drop_if_exists)
        
        logger.info("Validating setup...")
        if validate_vastdb_setup(storage_config):
            logger.info("✅ VastDB setup completed successfully!")
            logger.info("\nYou can now use PoCADuck with VastDB:")
            logger.info("```python")
            logger.info("from pocaduck import StorageConfig, Ingestor, Query")
            logger.info("")
            logger.info("# Configuration")
            logger.info("config = StorageConfig(")
            logger.info("    storage_type='vastdb',")
            logger.info(f"    vastdb_endpoint='{args.endpoint}',")
            logger.info(f"    vastdb_access_key='{args.access_key}',")
            logger.info(f"    vastdb_secret_key='***',")
            logger.info(f"    vastdb_bucket='{args.bucket}',")
            logger.info(f"    vastdb_schema='{args.schema}',")
            logger.info(f"    vastdb_table='{args.table}'")
            logger.info(")")
            logger.info("")
            logger.info("# Ingestion")
            logger.info("ingestor = Ingestor(config)")
            logger.info("ingestor.write(label=12345, block_id='block_0', points=point_array)")
            logger.info("ingestor.finalize()")
            logger.info("")
            logger.info("# Querying")
            logger.info("query = Query(config)")
            logger.info("points = query.get_points(label=12345)")
            logger.info("```")
            return 0
        else:
            logger.error("❌ Setup validation failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())