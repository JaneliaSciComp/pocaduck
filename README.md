# PoCADuck

PoCADuck is a library for efficiently storing and retrieving vast numbers of point clouds indexed by `uint64` labels. The name stands for:
- **PoC**: Point Clouds — the core payload
- **A**: Arrow — using the Arrow ecosystem for storage (Parquet and perhaps Arrow IPC)
- **Duck**: DuckDB — for label & block indexing

## Features

- Efficiently ingest 3D point clouds for labels in a blockwise fashion
- Parallelizable ingestion with worker-specific storage
- Automatically aggregate point clouds across blocks during retrieval
- Support for local and cloud storage (S3, GCS, Azure)
- Efficient queries using DuckDB's indexing capabilities

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

Each worker is given a non-overlapping sets of blocks and create
an Ingestor with common storage configuration, e.g., if local storage
is used, the workers share a common data directory.

```python
from pocaduck import Ingestor
import numpy as np

# Create an ingestor for a specific worker
ingestor = Ingestor(storage_config=config, worker_id="worker1")

# Write point clouds for each label in a block
label = 12345
block_id = "block_x1_y2_z3"
points = np.random.rand(1000, 3) * 100  # Random 3D points

# Write the points
ingestor.write(label=label, block_id=block_id, points=points)

# The worker continues writing points across blocks...

# Then finalize when finished writing for that worker.
ingestor.finalize()
```

When all workers have finished writing and called `ingestor.finalize()`,
we consolidate the workers' indices into one index.

```python
# storage config should be identical to one used by workers.
Ingestor.consolidate_indexes(config)
```

### Querying Point Clouds

Query point clouds across all blocks:

```python
from pocaduck import Query

# Create a query object
query = Query(storage_config=config)

# Get all available labels
labels = query.get_labels()
print(f"Available labels: {labels}")

# Get point count for a label
point_count = query.get_point_count(label=12345)
print(f"Label 12345 has {point_count} points")

# Get all blocks containing a label
blocks = query.get_blocks_for_label(label=12345)
print(f"Label 12345 is in blocks: {blocks}")

# Get all points for a label (aggregated across all blocks)
points = query.get_points(label=12345)
print(f"Retrieved {points.shape[0]} points for label 12345")

# Close the query connection when done
query.close()
```

## Architecture

PoCADuck follows an architecture with two main components:

1. **Ingestion**:
   - Multiple workers process blocks independently
   - Each worker writes point clouds for labels within blocks
   - Workers maintain local indexes for fast lookup
   - Indexes are consolidated after ingestion

2. **Querying**:
   - Unified index allows fast lookup by label
   - Automatically aggregates points across all blocks
   - Efficient filtering and retrieval using DuckDB

## Performance Optimization

For large datasets where labels are scattered across many worker files, PoCADuck provides an optimization pipeline that reorganizes data by label for significantly faster retrieval.

### How It Works

The optimization process:
1. Reads point data from the original structure
2. Reorganizes points by label into optimized parquet files
3. Creates a new optimized index for efficient label-based lookups

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
- Automatic deduplication of points
- Transparent integration - no code changes needed for existing applications

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