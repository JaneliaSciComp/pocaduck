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

Create an Ingestor for each worker:

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

# Finalize when done
ingestor.finalize()

# After all workers have finalized, consolidate indexes
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

## License

BSD 3-Clause License, see [LICENSE](./LICENSE)