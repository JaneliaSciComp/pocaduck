#!/usr/bin/env python
"""Benchmark gross_budgeting against mo-gen's current point-thinning path.

With no ``--store-path`` this command generates synthetic soma/neurite labels.
For a real PoCADuck store, pass both ``--store-path`` and ``--labels``. Results
are stored-sample preservation proxies, not neuron morphology or model quality.

scikit-learn is intentionally a benchmark-only dependency. It is imported
inside the old-path and nearest-neighbor metric functions, never by pocaduck.
"""

import argparse
import itertools
import json
import os
import platform
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from pocaduck import Ingestor, Query, StorageConfig
from pocaduck.budgeting import (
    DEFAULT_SOMA_THRESHOLD,
    bin_coordinates_nm,
    plan_gross_budgeting,
    sample_budget_plan,
    validate_gross_budgeting_parameters,
)


REFERENCE_MOGEN_SHA = "b6c9743c3ffe7e21284a6c07f8f6c8851ba65401"
REFERENCE_NPS_SHA = "53b4e14a52c765d99db62837f3b0cb303f792dfe"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_SYNTHETIC_COUNTS = (1_500, 100_000, 1_000_000)
DEFAULT_RATIOS = (0.90, 0.75)


@dataclass
class BenchmarkCase:
    label: int
    size_class: str
    points: np.ndarray
    soma_mask: np.ndarray
    soma_truth_source: str


def _nearest_neighbors_class():
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError(
            "The benchmark's faithful mo-gen baseline requires scikit-learn. "
            "Install scikit-learn in the benchmark environment; it is deliberately "
            "not a pocaduck package dependency."
        ) from exc
    return NearestNeighbors


def old_path_indices(
    points: np.ndarray,
    budget: int,
    subsample_ratio: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Return original-row indices from mo-gen's old thinning lifecycle.

    Faithfully reimplements ``mo-gen/src/dataset_dit.py:270-294`` at
    ``b6c9743``: uniform sampling happens first with ``round(ratio * n)``;
    only a remaining cloud of at least ``budget`` enters the density draw.
    The density weights match ``mo-gen/src/dataset.py:66-102``: the cube of
    each point's 8th-nearest-neighbor distance, sampled without replacement.
    """

    rng = np.random.default_rng(seed)
    indices = np.arange(len(points), dtype=np.int64)

    uniform_start = time.perf_counter()
    if subsample_ratio < 1.0 and len(indices):
        n_keep = max(1, int(round(subsample_ratio * len(indices))))
        if n_keep < len(indices):
            indices = rng.choice(indices, size=n_keep, replace=False)
    uniform_seconds = time.perf_counter() - uniform_start

    density_seconds = 0.0
    if len(indices) >= budget:
        density_start = time.perf_counter()
        xyz = np.asarray(points[indices, :3], dtype=np.float32)
        n_neighbors = min(8, len(xyz) - 1) + 1
        nearest_neighbors = _nearest_neighbors_class()
        distances, _ = nearest_neighbors(
            n_neighbors=n_neighbors, n_jobs=1
        ).fit(xyz).kneighbors(xyz)
        radii = distances[:, -1]
        weights = radii**3
        total_weight = float(np.sum(weights))
        if total_weight > 0:
            weights = weights / total_weight
        else:
            weights = np.full(len(xyz), 1.0 / len(xyz))
        selected = np.random.default_rng(seed).choice(
            len(xyz), size=budget, replace=False, p=weights
        )
        indices = indices[selected]
        density_seconds = time.perf_counter() - density_start

    return indices, {
        "uniform_seconds": uniform_seconds,
        "density_seconds": density_seconds,
    }


def gross_budgeting_indices(
    points: np.ndarray,
    budget: int,
    voxel_size: Sequence[float],
    seed: int,
    soma_threshold: float,
    sampling_fraction: Optional[float],
) -> tuple[np.ndarray, dict]:
    budget, _, _, _ = validate_gross_budgeting_parameters(
        budget,
        voxel_size,
        soma_threshold=soma_threshold,
        sampling_fraction=sampling_fraction,
    )
    if len(points) <= budget:
        pass2_start = time.perf_counter()
        selected_indices = np.arange(len(points), dtype=np.int64)
        pass2_seconds = time.perf_counter() - pass2_start
        return selected_indices, {
            "pass1_seconds": 0.0,
            "pass2_seconds": pass2_seconds,
        }

    pass1_start = time.perf_counter()
    plan = plan_gross_budgeting(
        points,
        budget,
        voxel_size,
        soma_threshold=soma_threshold,
        sampling_fraction=sampling_fraction,
    )
    pass1_seconds = time.perf_counter() - pass1_start

    pass2_start = time.perf_counter()
    index_column = np.arange(len(points), dtype=np.int64)[:, None]
    selected_indices = sample_budget_plan(index_column, plan, seed=seed)[:, 0]
    pass2_seconds = time.perf_counter() - pass2_start
    return selected_indices, {
        "pass1_seconds": pass1_seconds,
        "pass2_seconds": pass2_seconds,
    }


def _synthetic_case(
    label: int,
    point_count: int,
    voxel_size: Sequence[float],
    seed: int,
    size_class: str,
) -> BenchmarkCase:
    """Generate a dense physical-nm ball plus several thin branches."""

    rng = np.random.default_rng(seed)
    voxel_size = np.asarray(voxel_size, dtype=np.float64)
    soma_count = max(1, int(round(point_count * 0.62)))
    neurite_count = point_count - soma_count

    voxel_volume = float(np.prod(voxel_size))
    target_fill = 0.45
    soma_radius_nm = (
        soma_count * voxel_volume / (target_fill * (4.0 / 3.0) * np.pi)
    ) ** (1.0 / 3.0)
    center_nm = np.full(3, soma_radius_nm + 4.0 * np.max(voxel_size))

    directions = rng.normal(size=(soma_count, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = soma_radius_nm * np.cbrt(rng.random(soma_count))
    soma_nm = center_nm + directions * radii[:, None]

    # Nine nearby lattice points at each center make a thin, sampled tube rather
    # than a single-pixel mathematical line. Its local 8-NN scale is therefore
    # meaningful without making the neurite remotely as voluminous as the soma.
    cross_section = np.array(
        [(0, y, z) for y in (-1, 0, 1) for z in (-1, 0, 1)],
        dtype=np.float64,
    )
    center_count = max(1, int(np.ceil(neurite_count / len(cross_section))))
    branch_count = min(64, max(8, int(np.sqrt(center_count) // 2)))
    branch_ids = np.arange(center_count) % branch_count
    branch_steps = np.arange(center_count) // branch_count
    angles = 2.0 * np.pi * branch_ids / branch_count
    step_nm = float(np.min(voxel_size))
    x = center_nm[0] + soma_radius_nm + (branch_steps + 1) * step_nm
    separation_nm = 2.5 * np.max(voxel_size)
    y = center_nm[1] + separation_nm * branch_ids + 0.15 * soma_radius_nm * np.sin(
        branch_steps / 40.0 + angles
    )
    z = center_nm[2] + separation_nm * np.sin(angles) + 0.1 * soma_radius_nm * np.cos(
        branch_steps / 55.0 + angles
    )
    centers_nm = np.column_stack([x, y, z])
    neurite_nm = (
        centers_nm[:, None, :] + cross_section[None, :, :] * voxel_size
    ).reshape(-1, 3)[:neurite_count]

    coordinates_nm = np.vstack([soma_nm, neurite_nm])
    coordinates_voxel = np.rint(coordinates_nm / voxel_size).astype(np.int64)
    row_ids = np.arange(point_count, dtype=np.int64)
    soma_mask = row_ids < soma_count
    points = np.column_stack(
        [coordinates_voxel, row_ids, soma_mask.astype(np.int64)]
    )
    return BenchmarkCase(
        label=label,
        size_class=size_class,
        points=points,
        soma_mask=soma_mask,
        soma_truth_source="known synthetic fixture mask",
    )


def generate_synthetic_cases(
    counts: Sequence[int],
    labels_per_size: int,
    voxel_size: Sequence[float],
) -> list[BenchmarkCase]:
    cases = []
    next_label = 10_000
    for class_index, count in enumerate(counts):
        size_class = f"synthetic_{count:g}"
        for replicate in range(labels_per_size):
            cases.append(
                _synthetic_case(
                    next_label,
                    count,
                    voxel_size,
                    seed=10_000 * class_index + replicate,
                    size_class=size_class,
                )
            )
            next_label += 1
    return cases


def _real_size_class(point_count: int) -> str:
    if point_count < 10_000:
        return "real_~1e3"
    if point_count < 300_000:
        return "real_~1e5"
    return "real_~1e6"


def load_real_cases(
    store_path: str,
    labels: Sequence[int],
    voxel_size: Sequence[float],
    budget: int,
    soma_threshold: float,
    sampling_fraction: Optional[float],
    query_threads: Optional[int],
) -> list[BenchmarkCase]:
    config = StorageConfig(base_path=store_path)
    query = Query(config, threads=query_threads, cache_size=0)
    cases = []
    try:
        for label in labels:
            points = query.get_points(label, use_cache=False)
            plan = plan_gross_budgeting(
                points,
                budget,
                voxel_size,
                soma_threshold=soma_threshold,
                sampling_fraction=sampling_fraction,
            )
            soma_mask = plan.soma_mask[plan.partition_ids]
            cases.append(
                BenchmarkCase(
                    label=label,
                    size_class=_real_size_class(len(points)),
                    points=points,
                    soma_mask=soma_mask,
                    soma_truth_source="gross_budgeting detector-relative mask",
                )
            )
    finally:
        query.close()
    return cases


def write_synthetic_store(cases: Sequence[BenchmarkCase], store_path: str) -> None:
    config = StorageConfig(base_path=store_path)
    ingestor = Ingestor(
        config,
        worker_id="benchmark",
        max_points_per_file=1_000_000,
    )
    for case in cases:
        ingestor.write(case.label, f"synthetic_{case.label}", case.points)
    ingestor.finalize()
    Ingestor.consolidate_indexes(config)


def _method_specs(ratios: Sequence[float]) -> list[tuple[str, Optional[float]]]:
    return [("gross_budgeting", None)] + [
        (f"old_ratio_{ratio:.2f}", ratio) for ratio in ratios
    ]


def _select_compute(
    case: BenchmarkCase,
    method: str,
    ratio: Optional[float],
    seed: int,
    args,
) -> tuple[np.ndarray, dict]:
    if method == "gross_budgeting":
        return gross_budgeting_indices(
            case.points,
            args.budget,
            args.voxel_size,
            seed,
            args.soma_threshold,
            args.sampling_fraction,
        )
    return old_path_indices(case.points, args.budget, ratio, seed)


def coverage_fraction(
    raw_xyz: np.ndarray,
    selected_xyz: np.ndarray,
    voxel_size: Sequence[float],
    edge_nm: float,
) -> float:
    if len(raw_xyz) == 0:
        return 1.0
    voxel_size = np.asarray(voxel_size, dtype=np.float64)
    raw_nm = raw_xyz.astype(np.float64, copy=False) * voxel_size
    selected_nm = selected_xyz.astype(np.float64, copy=False) * voxel_size
    bbox_min = np.min(raw_nm, axis=0)
    raw_bins, raw_grid_shape = bin_coordinates_nm(raw_nm, bbox_min, edge_nm)
    # The subset must use the raw cloud's extent. Deriving a smaller extent
    # from selected points would clamp a selected boundary point into a
    # different cell than the identical point occupies in the raw cloud.
    selected_bins = np.floor((selected_nm - bbox_min) / edge_nm).astype(np.int64)
    np.clip(selected_bins, 0, raw_grid_shape - 1, out=selected_bins)
    raw_occupied = {tuple(row) for row in np.unique(raw_bins, axis=0)}
    selected_occupied = {tuple(row) for row in np.unique(selected_bins, axis=0)}
    return len(raw_occupied & selected_occupied) / len(raw_occupied)


def spatial_recall_distances_nm(
    raw_xyz: np.ndarray,
    selected_xyz: np.ndarray,
    voxel_size: Sequence[float],
) -> np.ndarray:
    if len(raw_xyz) == 0:
        return np.empty(0, dtype=np.float64)
    nearest_neighbors = _nearest_neighbors_class()
    voxel_size = np.asarray(voxel_size, dtype=np.float64)
    raw_nm = raw_xyz.astype(np.float64, copy=False) * voxel_size
    selected_nm = selected_xyz.astype(np.float64, copy=False) * voxel_size
    distances, _ = nearest_neighbors(n_neighbors=1, n_jobs=1).fit(
        selected_nm
    ).kneighbors(raw_nm)
    return distances[:, 0]


def run_compute_benchmark(cases, args, timing_rows, property_rows, selections):
    methods = _method_specs(args.subsample_ratios)
    for case in cases:
        for seed in args.seeds:
            shuffled = methods.copy()
            random.Random((case.label << 16) + seed).shuffle(shuffled)
            for method, ratio in shuffled:
                start = time.perf_counter()
                selected_indices, step_timings = _select_compute(
                    case, method, ratio, seed, args
                )
                elapsed = time.perf_counter() - start
                timing_rows.append(
                    {
                        "mode": "compute_only",
                        "cache_state": "preloaded",
                        "method": method,
                        "label": case.label,
                        "size_class": case.size_class,
                        "seed": seed,
                        "raw_points": len(case.points),
                        "selected_points": len(selected_indices),
                        "seconds": elapsed,
                        **step_timings,
                    }
                )

                selections[(case.label, method)].append(
                    np.asarray(selected_indices, dtype=np.int64)
                )
                selected = case.points[selected_indices]
                for edge_nm in args.coverage_edges_nm:
                    property_rows.append(
                        {
                            "metric": "coverage_fraction",
                            "method": method,
                            "label": case.label,
                            "size_class": case.size_class,
                            "seed": seed,
                            "parameter_nm": edge_nm,
                            "value": coverage_fraction(
                                case.points[:, :3],
                                selected[:, :3],
                                args.voxel_size,
                                edge_nm,
                            ),
                        }
                    )
                property_rows.append(
                    {
                        "metric": "soma_budget_share",
                        "method": method,
                        "label": case.label,
                        "size_class": case.size_class,
                        "seed": seed,
                        "parameter_nm": None,
                        "value": float(np.mean(case.soma_mask[selected_indices]))
                        if len(selected_indices)
                        else 0.0,
                    }
                )
                recall_distances = spatial_recall_distances_nm(
                    case.points[:, :3], selected[:, :3], args.voxel_size
                )
                for distance_nm in args.recall_distances_nm:
                    property_rows.append(
                        {
                            "metric": "spatial_recall_fraction",
                            "method": method,
                            "label": case.label,
                            "size_class": case.size_class,
                            "seed": seed,
                            "parameter_nm": distance_nm,
                            "value": float(np.mean(recall_distances <= distance_nm))
                            if len(recall_distances)
                            else 1.0,
                        }
                    )


def _fresh_end_to_end(case, method, ratio, seed, args, store_path):
    start = time.perf_counter()
    query = Query(
        StorageConfig(base_path=store_path),
        threads=args.query_threads,
        cache_size=max(1, len(args.labels or [])),
    )
    try:
        if method == "gross_budgeting":
            selected = query.get_points(
                case.label,
                budget=args.budget,
                strategy="gross_budgeting",
                voxel_size=args.voxel_size,
                seed=seed,
                sampling_fraction=args.sampling_fraction,
                soma_threshold=args.soma_threshold,
            )
        else:
            raw = query.get_points(case.label)
            indices, _ = old_path_indices(raw, args.budget, ratio, seed)
            selected = raw[indices]
    finally:
        query.close()
    return time.perf_counter() - start, len(selected)


def run_end_to_end_benchmark(cases, args, store_path, timing_rows):
    methods = _method_specs(args.subsample_ratios)

    if "fresh" in args.cache_states:
        for case in cases:
            for seed in args.seeds:
                shuffled = methods.copy()
                random.Random((case.label << 16) + seed + 1).shuffle(shuffled)
                for method, ratio in shuffled:
                    elapsed, selected_count = _fresh_end_to_end(
                        case, method, ratio, seed, args, store_path
                    )
                    timing_rows.append(
                        {
                            "mode": "end_to_end",
                            "cache_state": "fresh_query_consumer_faithful",
                            "method": method,
                            "label": case.label,
                            "size_class": case.size_class,
                            "seed": seed,
                            "raw_points": len(case.points),
                            "selected_points": selected_count,
                            "seconds": elapsed,
                        }
                    )

    if "warm" in args.cache_states:
        query = Query(
            StorageConfig(base_path=store_path),
            threads=args.query_threads,
            cache_size=max(1, len(cases)),
        )
        try:
            for case in cases:
                query.get_points(case.label)
            for case in cases:
                for seed in args.seeds:
                    shuffled = methods.copy()
                    random.Random((case.label << 16) + seed + 2).shuffle(shuffled)
                    for method, ratio in shuffled:
                        start = time.perf_counter()
                        if method == "gross_budgeting":
                            selected = query.get_points(
                                case.label,
                                budget=args.budget,
                                strategy="gross_budgeting",
                                voxel_size=args.voxel_size,
                                seed=seed,
                                sampling_fraction=args.sampling_fraction,
                                soma_threshold=args.soma_threshold,
                            )
                        else:
                            raw = query.get_points(case.label)
                            indices, _ = old_path_indices(
                                raw, args.budget, ratio, seed
                            )
                            selected = raw[indices]
                        elapsed = time.perf_counter() - start
                        timing_rows.append(
                            {
                                "mode": "end_to_end",
                                "cache_state": "warm_lru_counterfactual",
                                "method": method,
                                "label": case.label,
                                "size_class": case.size_class,
                                "seed": seed,
                                "raw_points": len(case.points),
                                "selected_points": len(selected),
                                "seconds": elapsed,
                            }
                        )
        finally:
            query.close()


def add_overlap_properties(cases, selections, property_rows):
    case_by_label = {case.label: case for case in cases}
    for (label, method), draws in selections.items():
        overlaps = []
        for first, second in itertools.combinations(draws, 2):
            denominator = min(len(first), len(second))
            if denominator == 0:
                overlaps.append(1.0)
            else:
                overlaps.append(
                    len(np.intersect1d(first, second, assume_unique=True)) / denominator
                )
        if overlaps:
            case = case_by_label[label]
            for value in overlaps:
                property_rows.append(
                    {
                        "metric": "epoch_draw_overlap_fraction",
                        "method": method,
                        "label": label,
                        "size_class": case.size_class,
                        "seed": None,
                        "parameter_nm": None,
                        "value": value,
                    }
                )


def _distribution(values: Sequence[float]) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def summarize_rows(rows, value_key, group_keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row[value_key])
    return [
        {
            **dict(zip(group_keys, key)),
            **_distribution(values),
        }
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _git_sha(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _hardware_metadata(query_threads: Optional[int]) -> dict:
    thread_variables = {}
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        thread_variables[name] = os.environ.get(name)
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_model": cpu_model or platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "query_threads": query_threads,
        "thread_environment": thread_variables,
    }


def build_report(cases, args, timing_rows, property_rows, warnings):
    repo_root = Path(__file__).resolve().parent
    repos_root = repo_root.parent
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = None
    return {
        "metadata": {
            "created_unix_seconds": time.time(),
            "data_kind": "real store" if args.store_path else "synthetic fixtures",
            "pocaduck_sha": _git_sha(repo_root),
            "mo_gen_reference_sha": _git_sha(repos_root / "mo-gen")
            or REFERENCE_MOGEN_SHA,
            "nps_reference_sha": _git_sha(repos_root / "nps") or REFERENCE_NPS_SHA,
            "scikit_learn": sklearn_version,
            "hardware": _hardware_metadata(args.query_threads),
            "measurement_notes": [
                "all timings use external perf_counter calls; Query timing=True is not used",
                "method order is deterministically randomized for every label and seed",
                "OS page-cache state is uncontrolled",
                "fresh_query_consumer_faithful constructs, queries once, and closes per method",
                "warm_lru_counterfactual is reported separately and is not mo-gen's lifecycle",
                "cloud properties are stored-sample proxies, not morphology or model quality",
            ],
            "config": {
                "store_path": args.store_path,
                "labels": [case.label for case in cases],
                "cohort": [
                    {
                        "label": case.label,
                        "size_class": case.size_class,
                        "point_count": len(case.points),
                        "soma_truth_source": case.soma_truth_source,
                    }
                    for case in cases
                ],
                "budget": args.budget,
                "voxel_size_nm": args.voxel_size,
                "seeds": args.seeds,
                "subsample_ratios": args.subsample_ratios,
                "soma_threshold": args.soma_threshold,
                "sampling_fraction": args.sampling_fraction,
                "coverage_edges_nm": args.coverage_edges_nm,
                "recall_distances_nm": args.recall_distances_nm,
                "modes": args.modes,
                "cache_states": args.cache_states,
            },
            "warnings": warnings,
        },
        "timing_distributions": summarize_rows(
            timing_rows,
            "seconds",
            ("mode", "cache_state", "method", "size_class"),
        ),
        "property_distributions": summarize_rows(
            property_rows,
            "value",
            ("metric", "method", "size_class", "parameter_nm"),
        ),
    }


def print_report(report):
    metadata = report["metadata"]
    print("gross_budgeting benchmark")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("\nTiming distributions (seconds)")
    for row in report["timing_distributions"]:
        print(json.dumps(row, sort_keys=True))
    print("\nStored-sample property proxy distributions")
    for row in report["property_distributions"]:
        print(json.dumps(row, sort_keys=True))


def _parse_csv(value, converter):
    return [converter(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-path", help="Existing PoCADuck store; omit for synthetic")
    parser.add_argument("--labels", help="Comma-separated real-store labels")
    parser.add_argument("--voxel-size", default="16,16,40", help="x,y,z pitches in nm")
    parser.add_argument("--budget", type=int, default=4096)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--subsample-ratios",
        default=",".join(map(str, DEFAULT_RATIOS)),
    )
    parser.add_argument(
        "--synthetic-counts",
        default=",".join(map(str, DEFAULT_SYNTHETIC_COUNTS)),
        help="Point counts for the synthetic size classes",
    )
    parser.add_argument("--labels-per-size", type=int, default=3)
    parser.add_argument("--coverage-edges-nm", default="500,1000,2000,4000")
    parser.add_argument("--recall-distances-nm", default="100,250,500,1000")
    parser.add_argument("--soma-threshold", type=float, default=DEFAULT_SOMA_THRESHOLD)
    parser.add_argument("--sampling-fraction", type=float)
    parser.add_argument("--query-threads", type=int)
    parser.add_argument(
        "--modes",
        default="compute,end-to-end",
        help="Comma-separated: compute,end-to-end",
    )
    parser.add_argument(
        "--cache-states",
        default="fresh,warm",
        help="End-to-end cache states: fresh,warm",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args(argv)

    args.voxel_size = _parse_csv(args.voxel_size, float)
    args.seeds = _parse_csv(args.seeds, int)
    args.subsample_ratios = _parse_csv(args.subsample_ratios, float)
    args.synthetic_counts = _parse_csv(args.synthetic_counts, int)
    args.coverage_edges_nm = _parse_csv(args.coverage_edges_nm, float)
    args.recall_distances_nm = _parse_csv(args.recall_distances_nm, float)
    args.modes = _parse_csv(args.modes, str)
    args.cache_states = _parse_csv(args.cache_states, str)
    args.labels = _parse_csv(args.labels, int) if args.labels else None

    if len(args.voxel_size) != 3 or any(value <= 0 for value in args.voxel_size):
        parser.error("--voxel-size must contain three positive values")
    if args.budget <= 0:
        parser.error("--budget must be positive")
    if not args.seeds:
        parser.error("--seeds must not be empty")
    if any(not 0 < ratio <= 1 for ratio in args.subsample_ratios):
        parser.error("--subsample-ratios values must be in (0, 1]")
    if (args.store_path is None) != (args.labels is None):
        parser.error("--store-path and --labels must be supplied together")
    if args.labels_per_size <= 0 or any(count <= 0 for count in args.synthetic_counts):
        parser.error("synthetic counts and labels-per-size must be positive")
    if not set(args.modes) <= {"compute", "end-to-end"}:
        parser.error("--modes accepts only compute,end-to-end")
    if not set(args.cache_states) <= {"fresh", "warm"}:
        parser.error("--cache-states accepts only fresh,warm")
    return args


def main(argv=None):
    args = parse_args(argv)
    warnings = []
    if len(args.seeds) < 5:
        warnings.append("fewer than five seeds were requested")

    temporary_store = None
    if args.store_path:
        cases = load_real_cases(
            args.store_path,
            args.labels,
            args.voxel_size,
            args.budget,
            args.soma_threshold,
            args.sampling_fraction,
            args.query_threads,
        )
        store_path = args.store_path
    else:
        cases = generate_synthetic_cases(
            args.synthetic_counts, args.labels_per_size, args.voxel_size
        )
        temporary_store = tempfile.mkdtemp(prefix="pocaduck_gross_benchmark_")
        store_path = temporary_store
        if "end-to-end" in args.modes:
            write_synthetic_store(cases, store_path)

    counts_by_class = defaultdict(int)
    for case in cases:
        counts_by_class[case.size_class] += 1
    for size_class, count in sorted(counts_by_class.items()):
        if count < 3:
            warnings.append(f"{size_class} has only {count} label(s); target is at least 3")

    timing_rows = []
    property_rows = []
    selections = defaultdict(list)
    try:
        if "compute" in args.modes:
            run_compute_benchmark(
                cases, args, timing_rows, property_rows, selections
            )
            add_overlap_properties(cases, selections, property_rows)
        if "end-to-end" in args.modes:
            run_end_to_end_benchmark(cases, args, store_path, timing_rows)
        report = build_report(cases, args, timing_rows, property_rows, warnings)
        print_report(report)
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        if temporary_store is not None:
            shutil.rmtree(temporary_store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
