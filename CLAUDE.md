# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PoCADuck is a library for efficiently storing and retrieving vast numbers of point clouds indexed by `uint64` labels. The name stands for:
- **PoC**: Point Clouds — the core payload
- **A**: Arrow — using Arrow IPC or Parquet for storage
- **Duck**: DuckDB — for label & block indexing

The library handles ingestion of point clouds while scanning large 3D label volumes (e.g., 3D neuron segmentation volumes) and retrieval of complete point clouds for any label across all blocks.

## Architecture

### Ingestion
- Uses worker IDs and paths to data directories or cloud storage
- Workers write 3D coordinates associated with labels in each block
- Handles consolidation of coordinates into parquet files
- Creates a DuckDB index per worker that maps labels to point cloud locations
- Concatenates worker indexes into a unified DuckDB index

### Querying
- Retrieves 3D point clouds for any label
- Automatically aggregates coordinates across all blocks
- Uses the unified DuckDB index for lookups

### Storage
- Uses DuckDB for native I/O to local and cloud storage
- Handles parquet file storage
- Supports paths and credentials for cloud access

## Core Components

The library consists of three main classes:

1. **StorageConfig**: Configures storage backend (local, S3, GCS, Azure)
   - `base_path`: Base path for storage
   - Cloud-specific parameters (region, credentials, etc.)
   - Validation of required parameters for each storage type

2. **Ingestor**: Handles writing point clouds for labels within blocks
   - `write(label, block_id, points)`: Writes 3D points for a label in a block
   - `finalize()`: Closes connections and commits transactions
   - `consolidate_indexes()`: Combines worker-specific indexes (static method)
   - Manages parquet files and DuckDB indexes

3. **Query**: Retrieves point clouds across blocks
   - `get_labels()`: Gets all available labels
   - `get_blocks_for_label(label)`: Gets all blocks containing a label
   - `get_point_count(label)`: Gets total point count for a label
   - `get_points(label)`: Gets all 3D points for a label (aggregated)

## Environment

The project uses a conda environment with the following key dependencies:
- Python 3.12
- pyarrow 20.0.0
- pandas 2.2.3
- numpy 2.2.5
- python-duckdb 1.2.2
- AWS SDK components for cloud storage
- Azure storage components

## Development

To set up the development environment, use the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate pocaduck
```

## Testing

Run the test suite to verify functionality:

```bash
python run_tests.py
```

The tests cover:
- Basic storage configuration
- Ingestor functionality (single worker, multiple workers)
- Query functionality (retrieving points across blocks)
- Multiple labels per block
- Example code validation

## Usage Examples

Example code is available in the `examples/` directory:

- `basic_usage.py`: Demonstrates basic workflow with multiple workers
- `cloud_storage.py`: Shows how to configure different cloud storage backends
- `volume_scanning.py`: Illustrates processing a 3D volume in blocks

## Implementation Notes

### DuckDB Considerations

- When working with DuckDB, convert numpy.uint64 to int when passing as parameters
- Use separate connections for each worker database when consolidating indexes
- For cloud storage access, configure DuckDB with the appropriate parameters

### Parquet Storage

- The library generates UUIDs for parquet files and tracks their locations
- When writing points, it manages file size and creates new files as needed
- Point clouds are stored efficiently with label, block_id, x, y, z columns