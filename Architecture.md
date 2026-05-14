## PoCADuck Overview

PoCADuck is a library for efficiently storing and retrieving vast numbers of point clouds indexed by `uint64` labels. It is built for the workflow of scanning a large 3D label volume (e.g., a neuron segmentation) blockwise: many parallel workers each emit per-label point sets for the blocks they own, and at query time the library aggregates a label's points back across every block that contained it.

| Element     | Meaning |
|-------------|--------|
| **PoC**     | Point Clouds — the core payload |
| **A**       | Apache **Parquet** — the on-disk format, written via `pandas.DataFrame.to_parquet` (PyArrow engine). Arrow IPC is *not* used. |
| **Duck**    | DuckDB — label/block index *and* the read engine that scans Parquet files on retrieval |


## Approach

### Ingestion

Each worker constructs an `Ingestor(storage_config, worker_id)`, which lays out:

- `{base_path}/worker_{worker_id}/data/{worker_id}-{N}.parquet` — point payload
- `{base_path}/worker_{worker_id}/index_{worker_id}.db` — DuckDB index

`ingestor.write(label, block_id, points)` accepts an `(N, D)` numpy integer array. The library is not hard-wired to D=3; D can include extra per-point attributes. Internally:

1. The current Parquet file is held as an in-memory `pandas.DataFrame` with columns `label` (int64), `block_id` (str), and `data` (a list-typed column where each row is one point's coordinate tuple).
2. When the running point count would exceed `max_points_per_file` (default 10M), the DataFrame is flushed to a `.parquet` file and a new file is opened.
3. Every flush produces one or more rows in the worker's DuckDB table `point_cloud_index(label UBIGINT, block_id VARCHAR, file_path VARCHAR, point_count UBIGINT)`. The same `(label, block_id)` may appear multiple times if its points were split across files.

After a worker calls `ingestor.finalize()`, its Parquet files and `.db` are committed and closed.

### Index consolidation

`Ingestor.consolidate_indexes(storage_config)` is a static method that scans `{base_path}/worker_*/index_*.db`, opens each worker DB, copies all rows out, and inserts them into `{base_path}/unified_index.db` with the same schema. Only the index is merged; the Parquet payload files stay where the workers wrote them.

### Querying

`Query(storage_config)` opens the unified index (read-only) and routes `get_points(label)` through DuckDB:

```sql
SELECT data
FROM parquet_scan(['…file1.parquet', '…file2.parquet', …])
WHERE label = ?
```

The list of files is determined by an index lookup on `label`. Result rows (each a per-point list) are stacked into an `(N, D) np.int64` array and `np.unique`'d to dedupe.

A small in-memory LRU cache (default size 10) holds recently fetched label point clouds to short-circuit repeat queries.

### Optional optimization pass

`optimize_point_cloud.py` reorganizes data so a single label's points are contiguous on disk:

- Output goes under `{base_path}/optimized/optimize_{worker_id}/optimized_*.parquet`.
- The new index `{base_path}/optimized/optimized_index.db` has schema `point_cloud_index(label BIGINT, file_path VARCHAR, point_count BIGINT)` — note that **`block_id` is dropped**.
- Deduplication is applied here, once, rather than on every read.
- `Query` auto-detects this directory and prefers the optimized index over `unified_index.db`. As a consequence, `Query.get_blocks_for_label()` returns `[]` once the optimized index is in use.

The optimizer is parallelizable via the `shard` / `optimize` / `consolidate` action flow described in the README.

### Storage handling

- **Local filesystems** are the supported and tested backend for ingest, optimize, and query.
- **Cloud storage (S3/GCS/Azure)**: `StorageConfig` validates the relevant credentials and emits the corresponding `SET …='…'` statements for any DuckDB connection the library opens. This means DuckDB-side operations (the query path's `parquet_scan`) can in principle read directly from object storage. However, the ingestion and optimization code uses local-filesystem APIs (`os.makedirs`, `os.path.exists`, `os.path.getsize`, `pandas.to_parquet`, `pandas.read_parquet` against bare paths) and is not wired up for cloud writes. Cloud-storage support is best regarded as partial / read-side until that gap is closed.
