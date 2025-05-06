## PoCADuck Overview

PPoCADuck is a library for efficiently storing and retrieving vast numbers of point clouds indexed by `uint64` labels. It efficiently handles ingestion of the point clouds while scanning large 3d label volumes (e.g., 3d neuron segmentation volumes) by allowing writes of points clouds for different labels within each block of the much larger volume. It efficiently handles retrieval of the full point cloud of a label across all written blocks for that label.

| Element     | Meaning |
|-------------|--------|
| **PoC**     | Point Clouds — the core payload |
| **A**       | Arrow — either Arrow IPC or Parquet |
| **Duck**    | DuckDB — label & block indexing engine |


## Approach

The pocaduck library can be used with many parallel workers that scan a large segmentation volume, generating point clouds for each label in a blockwise fashion. 

### Ingestion

The pocaduck library provides an ingestion object initialized with a worker ID and a path to a data directory or cloud storage bucket. For each block, a worker uses the `ingestor.write()` to write sets of 3d coordinates associated with each label in the block. The library handles consolidation of the many sets of 3d coordinates across all the blocks into relatively few parquet files, creating new parquet files as needed when the written data exceeds a settable parquet file size.

So for each worker, we have a set of parquet files containing 3d point clouds associated with labels and their originating block. Also, each worker has a DuckDB `.db` file that stores the location of the 3d point cloud for any given label in a block. 

At the end of ingestion across all workers, pocaduck reads in the worker-specific DuckDB index and concatenates them into one unified DuckDB index.

### Querying

The pocaduck library provides a query object initialized with the same path above used in the ingestion process. The query object allows retrieval of the 3d point cloud for any label, automatically aggregating the 3d coordinates across all blocks.

Under the hood, pocaduck uses the unified DuckDB index created at the end of the ingestion process.

### Storage handling

DuckDB natively handles I/O to both local and cloud storage. It also is used to store parquet files. Paths and credentials for cloud access can be passed into both ingestion and query objects.
