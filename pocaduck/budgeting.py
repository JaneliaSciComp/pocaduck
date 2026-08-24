"""Coverage-first point-budget allocation.

The ``gross_budgeting`` strategy has two deliberately separate passes:

* :func:`plan_gross_budgeting` chooses an isotropic partition grid in nm and
  assigns an integer quota to every occupied partition.
* :func:`sample_budget_plan` uses a NumPy generator to draw those quotas while
  carrying every point column through by row index.

The module is pure NumPy so callers and benchmarks can use the allocator on
already-loaded arrays without constructing a :class:`pocaduck.Query`.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Sequence

import numpy as np


CANDIDATE_RATIO = 1.25
DEFAULT_SOMA_THRESHOLD = 8.0


@dataclass(frozen=True)
class BudgetPlan:
    """Deterministic pass-1 result for a single stored point cloud.

    ``partition_ids`` maps each input row to the corresponding row of
    ``occupied_bins``, ``counts``, ``soma_mask``, and ``quotas``.
    """

    budget: int
    edge_length_nm: float
    bbox_min_nm: np.ndarray
    grid_shape: np.ndarray
    occupied_bins: np.ndarray
    partition_ids: np.ndarray
    counts: np.ndarray
    soma_mask: np.ndarray
    quotas: np.ndarray
    candidate_edge_lengths_nm: np.ndarray


def _validate_budget(budget: int) -> int:
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, Integral):
        raise ValueError("budget must be a positive integer")
    budget = int(budget)
    if budget <= 0:
        raise ValueError("budget must be a positive integer")
    return budget


def _validate_voxel_size(voxel_size: Sequence[float]) -> np.ndarray:
    try:
        result = np.asarray(voxel_size, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("voxel_size must contain three positive finite values") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("voxel_size must contain three positive finite values")
    return result


def _validate_soma_parameters(
    soma_threshold: float, sampling_fraction: Optional[float]
) -> tuple[float, Optional[float]]:
    try:
        threshold = float(soma_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("soma_threshold must be a finite value greater than 1") from exc
    if not np.isfinite(threshold) or threshold <= 1:
        raise ValueError("soma_threshold must be a finite value greater than 1")

    if sampling_fraction is None:
        return threshold, None

    try:
        fraction = float(sampling_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampling_fraction must be in the interval (0, 1]") from exc
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("sampling_fraction must be in the interval (0, 1]")
    return threshold, fraction


def candidate_edge_lengths(
    bbox_span_nm: Sequence[float],
    voxel_size: Sequence[float],
    *,
    ratio: float = CANDIDATE_RATIO,
) -> np.ndarray:
    """Return the pinned finite geometric ladder, largest edge first.

    The normal ladder starts at the largest bounding-box span and ends at the
    largest voxel pitch. A cloud smaller than one pitch, including a one-point
    cloud, has the single floor candidate.
    """

    span = np.asarray(bbox_span_nm, dtype=np.float64)
    if span.shape != (3,) or not np.all(np.isfinite(span)) or np.any(span < 0):
        raise ValueError("bbox_span_nm must contain three non-negative finite values")
    pitches = _validate_voxel_size(voxel_size)
    if not np.isfinite(ratio) or ratio <= 1:
        raise ValueError("ratio must be a finite value greater than 1")

    s_min = float(np.max(pitches))
    s_max = float(np.max(span))
    if s_max <= s_min:
        return np.array([s_min], dtype=np.float64)

    candidates = [s_max]
    edge = s_max
    while edge / ratio > s_min:
        edge /= ratio
        candidates.append(edge)
    if not np.isclose(candidates[-1], s_min, rtol=1e-12, atol=0.0):
        candidates.append(s_min)
    else:
        candidates[-1] = s_min
    return np.asarray(candidates, dtype=np.float64)


def bin_coordinates_nm(
    coordinates_nm: np.ndarray, bbox_min_nm: np.ndarray, edge_length_nm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bin nm coordinates on a bbox-anchored grid with upper-face clamping."""

    coordinates_nm = np.asarray(coordinates_nm, dtype=np.float64)
    bbox_min_nm = np.asarray(bbox_min_nm, dtype=np.float64)
    if coordinates_nm.ndim != 2 or coordinates_nm.shape[1] != 3:
        raise ValueError("coordinates_nm must have shape (N, 3)")
    if bbox_min_nm.shape != (3,):
        raise ValueError("bbox_min_nm must have shape (3,)")
    if not np.isfinite(edge_length_nm) or edge_length_nm <= 0:
        raise ValueError("edge_length_nm must be positive and finite")
    if len(coordinates_nm) == 0:
        return np.empty((0, 3), dtype=np.int64), np.ones(3, dtype=np.int64)

    spans = np.max(coordinates_nm, axis=0) - bbox_min_nm
    grid_shape = np.maximum(1, np.ceil(spans / edge_length_nm).astype(np.int64))
    bin_indices = np.floor(
        (coordinates_nm - bbox_min_nm) / np.float64(edge_length_nm)
    ).astype(np.int64)
    # The upper bbox face is exactly one edge away at s_max. Clamp it back
    # into the last grid cell so s_max remains the guaranteed one-cell rung.
    np.clip(bin_indices, 0, grid_shape - 1, out=bin_indices)
    return bin_indices, grid_shape


def _water_fill(
    quotas: np.ndarray,
    capacities: np.ndarray,
    eligible: np.ndarray,
    remaining: int,
) -> int:
    """Spread integer quota evenly over eligible partitions with capacity."""

    while remaining > 0:
        active = np.flatnonzero(eligible & (quotas < capacities))
        if len(active) == 0:
            break
        if remaining < len(active):
            quotas[active[:remaining]] += 1
            return 0

        share = remaining // len(active)
        additions = np.minimum(capacities[active] - quotas[active], share)
        used = int(np.sum(additions, dtype=np.int64))
        quotas[active] += additions
        remaining -= used

    return remaining


def allocate_partition_quotas(
    counts: np.ndarray, soma_mask: np.ndarray, budget: int
) -> np.ndarray:
    """Allocate exact integer quotas with the soma-cap relaxation contract."""

    budget = _validate_budget(budget)
    counts = np.asarray(counts)
    soma_mask = np.asarray(soma_mask)
    if (
        counts.ndim != 1
        or soma_mask.shape != counts.shape
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
    ):
        raise ValueError("counts must be positive integers with a matching soma_mask")
    counts = counts.astype(np.int64, copy=False)
    soma_mask = soma_mask.astype(bool, copy=False)
    partition_count = len(counts)
    if partition_count == 0:
        return np.empty(0, dtype=np.int64)
    if partition_count > budget:
        raise ValueError("occupied partition count exceeds budget")
    if int(np.sum(counts, dtype=np.int64)) < budget:
        raise ValueError("partition capacity is smaller than budget")

    base_quota = max(1, budget // partition_count)
    quotas = np.minimum(counts, base_quota).astype(np.int64, copy=False)
    remaining = budget - int(np.sum(quotas, dtype=np.int64))

    # Soma-like partitions stop at the base quota while non-soma capacity is
    # available. If that capacity is exhausted, relaxing the soma cap is what
    # preserves the exact-N contract for all-soma and soma-heavy fragments.
    remaining = _water_fill(quotas, counts, ~soma_mask, remaining)
    remaining = _water_fill(quotas, counts, soma_mask, remaining)
    if remaining:
        raise RuntimeError("quota allocation could not satisfy a feasible budget")
    return quotas


def _soma_partitions(
    counts: np.ndarray,
    edge_length_nm: float,
    voxel_size: np.ndarray,
    soma_threshold: float,
    sampling_fraction: Optional[float],
) -> np.ndarray:
    if len(counts) == 0:
        return np.empty(0, dtype=bool)

    if sampling_fraction is None:
        median_count = float(np.median(counts))
        return counts >= soma_threshold * median_count

    # nps stores approximately ``fraction * labeled_voxels`` points. Undoing
    # that rate gives an absolute labeled-voxel estimate, which is compared
    # with the full cubic partition capacity. The configurable multiplier is
    # reused as an inverse fill threshold: the default 8x relative threshold
    # corresponds to flagging cells estimated at least 1/8 full.
    cell_capacity_voxels = edge_length_nm**3 / float(np.prod(voxel_size))
    estimated_fill = counts / (sampling_fraction * cell_capacity_voxels)
    return estimated_fill >= 1.0 / soma_threshold


def plan_gross_budgeting(
    points: np.ndarray,
    budget: int,
    voxel_size: Sequence[float],
    *,
    soma_threshold: float = DEFAULT_SOMA_THRESHOLD,
    sampling_fraction: Optional[float] = None,
) -> BudgetPlan:
    """Build the deterministic partition grid and quota allocation.

    Coordinates are read from the first three columns in voxel units. The
    smallest feasible rung on the explicit 1.25-ratio ladder is selected by
    scanning every candidate because occupied-cell count is not monotone for
    a fixed-origin grid.
    """

    budget = _validate_budget(budget)
    pitches = _validate_voxel_size(voxel_size)
    threshold, fraction = _validate_soma_parameters(
        soma_threshold, sampling_fraction
    )
    points = np.asarray(points)

    if points.size == 0:
        edge = float(np.max(pitches))
        return BudgetPlan(
            budget=budget,
            edge_length_nm=edge,
            bbox_min_nm=np.zeros(3, dtype=np.float64),
            grid_shape=np.ones(3, dtype=np.int64),
            occupied_bins=np.empty((0, 3), dtype=np.int64),
            partition_ids=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int64),
            soma_mask=np.empty(0, dtype=bool),
            quotas=np.empty(0, dtype=np.int64),
            candidate_edge_lengths_nm=np.array([edge], dtype=np.float64),
        )
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, D) with D >= 3")

    coordinates_nm = points[:, :3].astype(np.float64, copy=False) * pitches
    if not np.all(np.isfinite(coordinates_nm)):
        raise ValueError("point coordinates must be finite")
    bbox_min_nm = np.min(coordinates_nm, axis=0)
    bbox_span_nm = np.max(coordinates_nm, axis=0) - bbox_min_nm
    candidates = candidate_edge_lengths(bbox_span_nm, pitches)

    selected = None
    target = min(budget, len(points))
    for edge in candidates:
        bins, grid_shape = bin_coordinates_nm(coordinates_nm, bbox_min_nm, edge)
        occupied_bins, partition_ids, counts = np.unique(
            bins, axis=0, return_inverse=True, return_counts=True
        )
        if len(occupied_bins) <= target:
            selected = (
                float(edge),
                grid_shape,
                occupied_bins,
                partition_ids.astype(np.int64, copy=False),
                counts.astype(np.int64, copy=False),
            )

    if selected is None:  # s_max is guaranteed feasible for every non-empty cloud.
        raise RuntimeError("candidate ladder contained no feasible partition size")

    edge, grid_shape, occupied_bins, partition_ids, counts = selected
    soma_mask = _soma_partitions(
        counts, edge, pitches, threshold, fraction
    )
    quotas = allocate_partition_quotas(counts, soma_mask, target)
    return BudgetPlan(
        budget=budget,
        edge_length_nm=edge,
        bbox_min_nm=bbox_min_nm,
        grid_shape=grid_shape,
        occupied_bins=occupied_bins,
        partition_ids=partition_ids,
        counts=counts,
        soma_mask=soma_mask,
        quotas=quotas,
        candidate_edge_lengths_nm=candidates,
    )


def sample_budget_plan(
    points: np.ndarray, plan: BudgetPlan, *, seed: Optional[int] = None
) -> np.ndarray:
    """Draw pass-2 rows for ``plan`` with a seedable NumPy Generator."""

    points = np.asarray(points)
    if points.size == 0:
        return points.copy()
    if points.ndim != 2 or len(points) != len(plan.partition_ids):
        raise ValueError("points must be the same (N, D) array used for the plan")
    if int(np.sum(plan.quotas, dtype=np.int64)) == len(points):
        return points.copy()

    rng = np.random.default_rng(seed)
    selections = []
    for partition_id, quota in enumerate(plan.quotas):
        indices = np.flatnonzero(plan.partition_ids == partition_id)
        quota = int(quota)
        if quota == len(indices):
            selections.append(indices)
        else:
            selections.append(rng.choice(indices, size=quota, replace=False))
    selected_indices = np.concatenate(selections)
    return points[selected_indices]


def gross_budget_points(
    points: np.ndarray,
    budget: int,
    voxel_size: Sequence[float],
    *,
    seed: Optional[int] = None,
    sampling_fraction: Optional[float] = None,
    soma_threshold: float = DEFAULT_SOMA_THRESHOLD,
) -> np.ndarray:
    """Convenience wrapper that runs both ``gross_budgeting`` passes."""

    plan = plan_gross_budgeting(
        points,
        budget,
        voxel_size,
        soma_threshold=soma_threshold,
        sampling_fraction=sampling_fraction,
    )
    return sample_budget_plan(points, plan, seed=seed)
