# VastDB Backend — Roadmap & Status

This file is a living planning doc for the VastDB backend work that lives on
this `dev` branch. The high-level "VastDB is the next-priority backend"
framing is in the project README on `main`; the operational detail belongs
here so it can iterate on the same branch as the code.

For the architectural rationale and target API, see the companion
[`design_vastdb_backend.md`](./design_vastdb_backend.md).

## Why this backend

The file-based Parquet + DuckDB stack works, but at scale a label's points
end up scattered across many worker Parquet files, and the offline
optimization pipeline exists almost entirely to undo that scattering.
[Vast Data's VastDB](https://vastdata.com/) gives us a transactional,
columnar SQL store that lets us replace the whole `Parquet files + DuckDB
index + offline optimization` triple with a single keyed lookup:

```
File-based backend (today):       VastDB backend (planned):
  Query → DuckDB index              Query → VastDB SQL → points
        → Parquet scan(s)
        → point cloud
```

For Janelia workloads — where a Vast cluster is already deployed — this
removes the scattering problem entirely without any optimization step.

## What already exists on `dev`

- `StorageConfig` extended with `vastdb_endpoint`, `vastdb_access_key`,
  `vastdb_secret_key`, `vastdb_bucket`, `vastdb_schema`, `vastdb_table`.
- Backend split inside `Query` and `Ingestor` so the public API is
  unchanged: callers still construct `StorageConfig(...)` and
  `Ingestor(...)` / `Query(...)`, and the right backend is selected from
  `storage_type`.
- `VastDBIngestorBackend` that batches per-label rows in memory and bulk-
  inserts via the `vastdb` Python SDK, plus a matching `VastDBQuery`.
- `pocaduck/setup.py` and `examples/vastdb_setup.py` for provisioning the
  bucket / schema / table / projections.
- A `design_vastdb_backend.md` design note describing the architecture and
  migration phases.

The branch was tagged at MVP-quality after a round of testing against
Janelia's internal Vast cluster (commit `fef93a7`).

## What's left before merge to `main`

1. **End-to-end test against a real VastDB instance.** Stand up a test
   that exercises the full ingest → query loop against either Janelia's
   cluster or a small CI-friendly target, and bring it into `run_tests.py`
   (skipped by default when no endpoint is configured).
2. **Pin down the `points` column schema.** Today the column is a flattened
   int64 list, which loses the original `(N, D)` shape. Either store `D`
   as a sibling column or use a fixed-size-list type so non-3D point
   clouds round-trip without the caller having to remember the dimension.
3. **Backend parity audit.** Walk every public method on `Query` and
   `Ingestor` and confirm the VastDB backend matches the file-based
   semantics — in particular:
   - dedup behaviour (file-based path runs `np.unique` on read; VastDB
     should match, or the divergence should be documented),
   - `get_blocks_for_label` (likely returns `[]` since the VastDB schema
     doesn't carry block IDs — same caveat as the optimized index),
   - empty-label handling and dtype of the returned array,
   - the shape of the `timing_info` dict so callers using
     `Query.print_timing_info(...)` don't break when the backend changes.
4. **Rebase / clean up unrelated drift.** The `dev` `query.py` rewrite is
   larger than just adding the VastDB backend; before merging, separate the
   pure-VastDB additions from any incidental refactors of the Parquet path
   so reviewers can read each change in isolation.
5. **Docs follow-up on `main`.** Once merged, expand the README's
   "Storage backends" section to describe the two-backend model and add
   `vastdb` to the supported list.

## Out of scope for the first merge

- Migration utility from Parquet/DuckDB → VastDB. The `design_vastdb_backend.md`
  Phase 2 mentions this; defer until at least one production deployment is
  on VastDB and actually wants to migrate.
- Deprecating the Parquet backend (Phase 3 in the design doc). The
  Parquet backend should remain the default for non-Vast deployments
  indefinitely.

## After this lands

Cloud object-store ingestion (S3 / GCS / Azure) for the Parquet backend
becomes the next-priority storage work. See the README Roadmap on `main`
for that plan.
