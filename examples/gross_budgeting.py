#!/usr/bin/env python
"""Run one budgeted retrieval against a small generated PoCADuck store.

The Query lifecycle intentionally matches mo-gen's ``PoCADuckInterface``
(``src/interfaces.py:16-29``): construct a Query, make one ``get_points`` call,
and close it. In a real consumer, pass the dataset's voxel pitches in nm.

Integration contract
--------------------
Budgeted retrieval replaces both downstream mo-gen thinning stages. The current
mo-gen path applies ``subsample_ratio`` before checking ``n_points``
(``src/dataset_dit.py:270-294``). Passing a 4096-point budgeted cloud through it
unchanged would first reduce the cloud to 3686 points at ratio 0.90 or 3072 at
ratio 0.75, then skip density-aware sampling because it is already below 4096.
Adopters must bypass both the uniform subsample and the r_k^3 density draw, or
request a correspondingly larger upstream budget.
"""

import shutil
import tempfile

import numpy as np

from pocaduck import Ingestor, Query, StorageConfig


def main():
    store_path = tempfile.mkdtemp(prefix="pocaduck_gross_budgeting_")
    try:
        config = StorageConfig(base_path=store_path)
        rng = np.random.default_rng(7)
        dense_region = rng.integers(0, 8, size=(400, 3), dtype=np.int64)
        sparse_branch = np.column_stack(
            [
                np.arange(10, 210, dtype=np.int64),
                np.full(200, 4, dtype=np.int64),
                np.full(200, 4, dtype=np.int64),
            ]
        )
        coordinates = np.vstack([dense_region, sparse_branch])
        supervoxel_ids = np.arange(len(coordinates), dtype=np.int64)[:, None]
        stored_points = np.hstack([coordinates, supervoxel_ids])

        ingestor = Ingestor(config, worker_id="example")
        ingestor.write(label=42, block_id="generated", points=stored_points)
        ingestor.finalize()
        Ingestor.consolidate_indexes(config)

        # This one-call lifecycle is the same shape as PoCADuckInterface.load_pc.
        query = Query(config)
        try:
            points = query.get_points(
                42,
                budget=64,
                strategy="gross_budgeting",
                voxel_size=(16, 16, 40),
                seed=11,
            )
        finally:
            query.close()

        print(f"Retrieved {len(points)} budgeted points with shape {points.shape}")
    finally:
        shutil.rmtree(store_path)


if __name__ == "__main__":
    main()
