# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PoCADuck is a library for efficiently storing and retrieving vast numbers of point clouds indexed by `uint64` labels. The name stands for:
- **PoC**: Point Clouds — the core payload
- **A**: Apache Parquet — written via `pandas.DataFrame.to_parquet` (PyArrow engine). Arrow IPC is **not** used despite the name suggesting it might be.
- **Duck**: DuckDB — both the label/block index store and the read engine that scans Parquet files at query time

The library handles ingestion of point clouds while scanning large 3D label volumes (e.g., 3D neuron segmentation volumes) and retrieval of complete point clouds for any label across all blocks. Although the typical use case is 3D coordinates, `Ingestor.write` accepts any `(N, D)` integer array — D is not constrained to 3.

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
- Local filesystem is the supported, tested target for ingestion, optimization, and querying.
- Cloud backends (S3, GCS, Azure): `StorageConfig` validates credentials and forwards them to every DuckDB connection the library opens, so DuckDB-side reads (the query path's `parquet_scan`) can in principle run against object storage. The ingest and optimize code paths, however, use local FS APIs (`os.makedirs`, `os.path.exists`, `os.path.getsize`, `pandas.to_parquet`/`read_parquet` against bare paths) and do not currently work against cloud paths. Treat cloud support as read-side / partial.

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

- Worker-stage Parquet files are named `{worker_id}-{N}.parquet` (sequential per worker, not UUID-based; UUIDs are only used for the optimized output files).
- File size is bounded by `max_points_per_file` (default 10M points). When the running point count would exceed it, the in-memory pandas DataFrame is flushed to disk and a new file is started.
- Point clouds are stored as three columns: `label` (int64), `block_id` (string), and `data` (a list-typed column where each row is one point's coordinate tuple). They are *not* stored as separate `x`, `y`, `z` columns.
- All retrieved points are returned as `np.int64`; floating-point coordinate inputs will be silently truncated.
- The unoptimized read path runs `np.unique` to dedupe; the optimized read path does not (the optimizer dedupes once at build time instead).