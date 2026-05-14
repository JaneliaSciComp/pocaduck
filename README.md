# PoCADuck

PoCADuck is a library for efficiently storing and retrieving vast numbers of point clouds indexed by `uint64` labels. The name stands for:
- **PoC**: Point Clouds — the core payload
- **A**: Apache Parquet — the on-disk format for the points themselves (written via pandas, which uses PyArrow as its engine)
- **Duck**: DuckDB — used both for the label/block index and for reading points back out of Parquet

## Features

- Ingest point clouds for labels in a blockwise fashion. Each `write()` accepts an `(N, D)` integer array — `D` is not constrained to 3, so additional per-point attributes can ride alongside coordinates.
- Parallelizable ingestion: each worker writes to its own subdirectory and its own DuckDB index file; a separate `consolidate_indexes()` step merges those indexes into one.
- Retrieval automatically aggregates a label's points across all blocks/files it lives in.
- Optional offline optimization pass that reorganizes points by label into larger Parquet files for faster queries (often 10–100×).
- DuckDB-backed indexing with an in-memory LRU cache for recently fetched labels.

## Storage backends

Local filesystems are the supported, tested target. `StorageConfig` accepts S3, GCS, and Azure parameters and forwards them to DuckDB, which lets the **read** path scan Parquet files directly out of object storage. However, the **ingest** and **optimize** pipelines currently call local-filesystem APIs (`os.makedirs`, `os.path.exists`, `pandas.DataFrame.to_parquet` against bare paths) and have not been exercised against cloud paths — treat cloud-storage ingestion as unimplemented for now. A second backend targeting Vast Data's VastDB is in development on the `dev` branch and is the next-priority storage target; cloud-storage ingestion is queued behind it. See the [Roadmap](#roadmap) for details.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pocaduck.git
cd pocaduck

# Create and activate conda environment
conda env create -f environment.yml
conda activate pocaduck

# Install the package in development mode
pip install -e .
```

## Usage

### Storage Configuration

First, configure your storage backend:

```python
from pocaduck import StorageConfig

# Local storage
config = StorageConfig(base_path="/path/to/data")

# S3 storage
s3_config = StorageConfig(
    base_path="s3://your-bucket/your-path",
    s3_region="us-west-2",
    s3_access_key_id="your-access-key",
    s3_secret_access_key="your-secret-key"
)

# GCS storage
gcs_config = StorageConfig(
    base_path="gs://your-bucket/your-path",
    gcs_project_id="your-project-id",
    gcs_credentials="/path/to/credentials.json"
)
```

### Point Cloud Ingestion

Each worker is given a non-overlapping set of blocks and creates an
Ingestor with the shared storage configuration. All workers must point
at the same `base_path` so the consolidation step can find their indexes.

```python
from pocaduck import Ingestor
import numpy as np

# Create an ingestor for a specific worker
ingestor = Ingestor(storage_config=config, worker_id="worker1")

# Write point clouds for each label in a block.
# `points` is an (N, D) integer array. Returned points come back as np.int64,
# so float coordinates will be truncated — pre-quantize voxel coordinates
# yourself before writing.
label = 12345
block_id = "block_x1_y2_z3"
points = np.random.randint(0, 1024, size=(1000, 3), dtype=np.int64)

ingestor.write(label=label, block_id=block_id, points=points)

# The worker continues writing points across blocks...

# Then finalize when finished writing for that worker.
ingestor.finalize()
```

#### What ingestion writes to disk

Per worker (under `{base_path}/worker_{worker_id}/`):

- `data/{worker_id}-{N}.parquet` — Parquet files with three columns:
  `label BIGINT`, `block_id VARCHAR`, `data` (a list column where each row
  is one point's coordinate tuple). New files roll over once
  `max_points_per_file` (default 10M points) is reached.
- `index_{worker_id}.db` — a DuckDB database with one table,
  `point_cloud_index(label UBIGINT, block_id VARCHAR, file_path VARCHAR, point_count UBIGINT)`.
  One row per `(label, block_id, file_path)` write — a single label/block
  may have multiple rows if its points were split across files.

When all workers have finished writing and called `ingestor.finalize()`,
we consolidate the workers' indices into one index.

```python
# storage config should be identical to one used by workers.
Ingestor.consolidate_indexes(config)
```

### Querying Point Clouds

Query point clouds across all blocks. The DuckDB index is opened in
read-only mode, so multiple concurrent `Query` instances are safe.

```python
from pocaduck import Query

# Create a query object. If `{base_path}/optimized/optimized_index.db`
# exists, it is used automatically; otherwise the unconsolidated
# `{base_path}/unified_index.db` is used.
query = Query(storage_config=config, cache_size=10)

# All unique labels in the index.
labels = query.get_labels()

# Total point count across every block/file for one label.
point_count = query.get_point_count(label=12345)

# Block IDs that contain a label. NOTE: the optimization step drops
# block_id, so this returns [] when the optimized index is in use.
blocks = query.get_blocks_for_label(label=12345)

# Aggregated points for a label, returned as an (N, D) np.int64 array.
points = query.get_points(label=12345)

query.close()
```

**Deduplication caveat.** When reading from the unoptimized index,
`get_points()` applies `np.unique` so duplicate coordinates collapse.
When reading from the optimized index, no deduplication runs at query
time — instead, the optimizer dedupes once when it builds the optimized
files. If you mix workflows or skip optimization on data with overlapping
writes, expect those two code paths to return different row counts.

## Architecture

PoCADuck has two required stages and one optional one:

1. **Ingestion** (`Ingestor`): Many workers run in parallel, each writing
   to its own `worker_{id}/data/*.parquet` files and its own
   `index_{id}.db`. Inside a worker, points accumulate in an in-memory
   pandas DataFrame and flush to a single Parquet file per
   `max_points_per_file`-sized chunk. Each `(label, block_id, file)` write
   becomes one row in the worker's DuckDB index.

2. **Index consolidation** (`Ingestor.consolidate_indexes`): After every
   worker has called `finalize()`, this static method scans
   `worker_*/index_*.db`, pulls all rows out of each, and inserts them
   into a single `{base_path}/unified_index.db` with the same schema. The
   underlying Parquet files are not touched — only the index is merged.

3. **Optional optimization** (`optimize_point_cloud.py`): Reorganizes
   the Parquet payload so each label's points live together in a small
   number of files, dropping the `block_id` dimension. Output goes under
   `{base_path}/optimized/`, and `Query` switches to it transparently
   when the optimized index is present.

## Performance Optimization

For large datasets where labels are scattered across many worker files, PoCADuck provides an optimization pipeline that reorganizes data by label for significantly faster retrieval.

### How It Works

The optimization process:
1. Reads point data from the original structure (in batches of `--batch-size` labels at a time).
2. Deduplicates each label's points and groups them by label into new Parquet files of approximately `--target-file-size` bytes.
3. Builds a new index `optimized_index.db` whose schema drops `block_id`: `point_cloud_index(label BIGINT, file_path VARCHAR, point_count BIGINT)`, plus a `CREATE INDEX idx_label`.

Because `block_id` is dropped, `Query.get_blocks_for_label()` returns `[]` once the optimized index is in use. If you need block provenance, query before running optimization (or keep the unconsolidated worker indexes around).

The optimized data is stored in a standard directory structure:
```
/base_path/
├── unified_index.db              # Original index
├── worker_1/                     # Original ingestion worker dirs
├── worker_2/
└── optimized/                    # Optimization directory
    ├── optimized_index.db        # Consolidated optimized index
    ├── optimize_worker1/         # Optimizer worker directories
    │   ├── metadata.json         # Worker metadata
    │   ├── optimized_*.parquet   # Optimized parquet files
    └── optimize_worker2/
        └── ...
```

After optimization, the Query class automatically detects and uses the optimized data structure with no code changes required.

### Running the Optimization Pipeline

The optimization pipeline supports parallel processing to handle large datasets efficiently:

```bash
# 1. Shard the labels for parallel processing (e.g., 8 workers).
#    This will create `labels_shard_*.txt` files that partition the labels
#    into shard files `labels_shard_1.txt` to `labels_shard_<N>.txt` 
#    where <N> is num-shards.
python optimize_point_cloud.py --action shard --base-path /path/to/data --num-shards 8

# 2. Run optimization workers in parallel (can be on separate machines)
#    Typically you'd use a bash script or a cluster feature (like job arrays)
#    to easily launch each worker as an array of jobs.

# Worker 1
python optimize_point_cloud.py --action optimize --base-path /path/to/data \
  --labels-file labels_shard_1.txt --worker-id worker1 --threads 16

# Worker 2
python optimize_point_cloud.py --action optimize --base-path /path/to/data \
  --labels-file labels_shard_1.txt --worker-id worker2 --threads 16

# ... and so on for all shards

# 3. Consolidate the results into a unified optimized index
python optimize_point_cloud.py --action consolidate --base-path /path/to/data
```

**Note**: If interrupted, workers should be restarted from the beginning with a fresh worker directory.

### Key Options

- `--target-file-size`: Target size for optimized parquet files (default: 500MB)
- `--batch-size`: Number of labels to process in a batch for memory management (default: 100)
- `--threads`: Number of threads to use for DuckDB processing
- `--quiet`: Suppress progress output

### Benefits

- Significantly faster retrieval for label-based queries (often 10-100x speedup)
- Reduced disk I/O by consolidating each label's data
- Automatic deduplication of points (applied once during optimization, rather than on every read as the unoptimized path does)
- Transparent integration — `Query` auto-detects `optimized/optimized_index.db` and uses it without code changes

## Performance Analysis and Debugging

PoCADuck provides timing functionality to help analyze and debug slow queries. This is particularly useful for understanding where time is spent during point cloud retrieval and identifying potential performance bottlenecks.

### Using Timing Features

All query methods support an optional `timing` parameter that returns detailed performance metrics:

```python
from pocaduck import Query

query = Query(storage_config=config)

# Get timing information for any query method
points, timing_info = query.get_points(label=12345, timing=True)
count, timing_info = query.get_point_count(label=12345, timing=True)  
blocks, timing_info = query.get_blocks_for_label(label=12345, timing=True)

# Pretty print timing analysis to stdout
Query.print_timing_info(timing_info)

query.close()
```

### Timing Data Structure

The timing information depends on the backend used, but for the parquet/DuckDB backend, it includes:

- **Execution breakdown**: Time spent on cache lookup, index lookup, data reading, and processing
- **File access patterns**: Number of files accessed and points per file distribution
- **Query details**: SQL queries generated and backend-specific metrics
- **Storage information**: File sizes, optimization status, and cache utilization

### Example Usage

```python
# Analyze a slow query
points, timing = query.get_points(problematic_label, timing=True)

# Print detailed timing analysis
Query.print_timing_info(timing)

# Access specific metrics programmatically
files_accessed = timing['query_efficiency']['files_accessed']
avg_points_per_file = timing['query_efficiency']['points_per_file_range']['avg']
total_time = timing['total_time']

if files_accessed > 10:
    print("Consider running optimization pipeline")
```

## Roadmap

### 1. VastDB backend (next priority)

A second backend targeting [Vast Data's VastDB](https://vastdata.com/) is in  development on the `dev` branch targeting Janelia's internal VAST storage system. The motivation is that VastDB's transactional, columnar SQL model lets us replace the entire `Parquet files + DuckDB index + offline optimization pipeline` stack with a single keyed lookup:

```
File-based backend (today):       VastDB backend (planned):
  Query → DuckDB index              Query → VastDB SQL → points
        → Parquet scan(s)
        → point cloud
```

Detailed status (what already exists on `dev`, what's left before merge) lives alongside that branch's other VastDB design notes — see `roadmap_vastdb.md` on `dev`.

### 2. Cloud object-store ingestion (later)

Today, `StorageConfig` accepts S3 / GCS / Azure parameters and the `Query` path's DuckDB connection can in principle scan Parquet directly out of object storage, but ingestion and the optimization pipeline only work against local filesystems (see "Storage backends" above). Closing that gap — wrapping the local FS calls in a small filesystem abstraction (e.g. `fsspec`) and adding round-trip tests against a MinIO or fake-GCS container — is queued **after** the VastDB backend lands. For Janelia workloads the VastDB path is the more useful next step; for external users we expect cloud ingestion to be the natural follow-up.

### 3. Smaller follow-ups

- Resumable/incremental optimizer workers (today an interrupted worker has to be restarted from scratch).
- Optional float-coordinate support, or a clearer error when float arrays are passed (currently they get silently truncated to `int64`).
- Surface `get_blocks_for_label`'s "returns `[]` on optimized data" caveat as a typed return (`None` or a sentinel) rather than an empty list, so callers can distinguish "no blocks" from "block info dropped."

## Testing

PoCADuck includes a comprehensive test suite to verify functionality. Here's how to run the tests:

```bash
# Make sure your environment is activated
conda activate pocaduck

# Run all tests
python run_tests.py

# Run a specific test file
python -m unittest tests/test_local_storage.py

# Run a specific test case
python -m unittest tests.test_local_storage.TestLocalStorage.test_write_and_query_single_block

# Run tests with more verbose output
python run_tests.py -v
```

### Test Data

The repository includes sample test data in `tests/data/` that contains real point cloud data for a variety of labels across multiple blocks. This data is used by the test suite to verify the functionality of the library with realistic data.

### Test Coverage

The test suite covers:
- Storage configuration (local and cloud storage backends)
- Point cloud ingestion with single and multiple workers
- Querying point clouds across blocks
- Handling multiple labels per block
- Working with n-dimensional data (not just 3D coordinates)
- Example code validation

## License

BSD 3-Clause License, see [LICENSE](./LICENSE)