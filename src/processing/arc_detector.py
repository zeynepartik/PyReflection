import pandas as pd
import numpy as np

MIN_ARC_POINTS = 3
TIME_GAP_HOURS = 4.0


def _is_time_gap(epochs: np.ndarray, index: int) -> bool:
    if index <= 0:
        return False
    dt_hours = (epochs[index] - epochs[index - 1]) / np.timedelta64(1, "h")
    return dt_hours >= TIME_GAP_HOURS


def _local_elevation_sign(
    elevations: np.ndarray,
    epochs: np.ndarray,
    index: int,
    default: int = 1,
) -> int:
    """Use in-segment neighbors only; skip diffs that cross time gaps."""
    n = len(elevations)

    if index + 1 < n and not _is_time_gap(epochs, index + 1):
        sign = np.sign(elevations[index + 1] - elevations[index])
        if sign != 0:
            return int(sign)

    if index > 0 and not _is_time_gap(epochs, index):
        sign = np.sign(elevations[index] - elevations[index - 1])
        if sign != 0:
            return int(sign)

    return default


def _compute_full_signs(elevations: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    n = len(elevations)
    full_signs = np.ones(n, dtype=int)

    for i in range(n):
        if i == 0 or _is_time_gap(epochs, i):
            default = full_signs[i - 1] if i > 0 else 1
            full_signs[i] = _local_elevation_sign(elevations, epochs, i, default=default)
        else:
            sign = np.sign(elevations[i] - elevations[i - 1])
            full_signs[i] = int(sign) if sign != 0 else full_signs[i - 1]

    return full_signs


def _renumber_arcs(arc_nos: np.ndarray) -> np.ndarray:
    unique_arcs = sorted(set(arc_nos.tolist()))
    mapping = {old: new for new, old in enumerate(unique_arcs, start=1)}
    return np.array([mapping[arc_no] for arc_no in arc_nos], dtype=int)


def _merge_short_arcs(
    arc_nos: np.ndarray,
    arc_types: np.ndarray,
    min_points: int = MIN_ARC_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    arc_nos = arc_nos.copy()
    arc_types = arc_types.copy()

    while True:
        unique_arcs = sorted(set(arc_nos.tolist()))
        arc_sizes = {arc_no: int(np.sum(arc_nos == arc_no)) for arc_no in unique_arcs}
        short_arcs = [arc_no for arc_no, size in arc_sizes.items() if size < min_points]

        if not short_arcs:
            break

        short_arc = min(short_arcs, key=lambda arc_no: arc_sizes[arc_no])
        arc_index = unique_arcs.index(short_arc)
        neighbors: list[int] = []

        if arc_index > 0:
            neighbors.append(unique_arcs[arc_index - 1])
        if arc_index + 1 < len(unique_arcs):
            neighbors.append(unique_arcs[arc_index + 1])

        if not neighbors:
            break

        short_type = arc_types[arc_nos == short_arc][0]
        matching_neighbors = [
            neighbor
            for neighbor in neighbors
            if arc_types[arc_nos == neighbor][0] == short_type
        ]

        if matching_neighbors:
            target_arc = max(matching_neighbors, key=lambda arc_no: arc_sizes[arc_no])
        elif arc_index + 1 < len(unique_arcs):
            target_arc = unique_arcs[arc_index + 1]
        else:
            target_arc = unique_arcs[arc_index - 1]

        target_type = arc_types[arc_nos == target_arc][0]
        short_mask = arc_nos == short_arc
        arc_nos[short_mask] = target_arc
        arc_types[short_mask] = target_type
        arc_nos = _renumber_arcs(arc_nos)

    return arc_nos, arc_types


def _assign_arc_columns(obs_type_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply elevation-based arc numbering to a single obsType time series.

    Expects rows for one satellite and one obsType, sorted by epoch.
    """
    obs_type_df = obs_type_df.sort_values("epoch").reset_index(drop=True)
    n = len(obs_type_df)

    if n == 0:
        return obs_type_df

    if n < MIN_ARC_POINTS:
        obs_type_df["arcNo"] = 1
        obs_type_df["arcType"] = _local_elevation_sign(
            obs_type_df["elevation"].to_numpy(dtype=float),
            obs_type_df["epoch"].to_numpy(),
            0,
        )
        return obs_type_df

    elevations = obs_type_df["elevation"].to_numpy(dtype=float)
    epochs = obs_type_df["epoch"].to_numpy()

    full_signs = _compute_full_signs(elevations, epochs)

    time_gaps = np.array([_is_time_gap(epochs, i) for i in range(n)], dtype=bool)
    sign_changes = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not time_gaps[i]:
            sign_changes[i] = full_signs[i] != full_signs[i - 1]

    arc_triggers = time_gaps | sign_changes
    arc_triggers[0] = False

    arc_nos = np.ones(n, dtype=int)
    arc_types = np.ones(n, dtype=int)
    current_arc = 1

    for i in range(n):
        if arc_triggers[i]:
            current_arc += 1
        arc_nos[i] = current_arc
        arc_types[i] = 1 if full_signs[i] >= 0 else -1

    arc_nos, arc_types = _merge_short_arcs(arc_nos, arc_types)

    obs_type_df["arcNo"] = arc_nos
    obs_type_df["arcType"] = arc_types

    return obs_type_df


def detect_arcs_for_satellite(sat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign arcNo and arcType for one satellite.

    Each obsType is evaluated independently because multiple obsType rows can
    share the same epoch. Arc numbering restarts per obsType series.
    """
    if sat_df.empty:
        return sat_df

    processed_groups = [
        _assign_arc_columns(group)
        for _, group in sat_df.groupby("obsType", sort=False)
    ]

    return (
        pd.concat(processed_groups, ignore_index=True)
        .sort_values(["epoch", "obsType"], kind="mergesort")
        .reset_index(drop=True)
    )
