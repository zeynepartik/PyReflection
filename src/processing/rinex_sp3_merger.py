"""
Merge SP3 orbit data with RINEX SNR observations via Lagrange interpolation.

Reads SP3 and RINEX files, interpolates satellite ECEF coordinates at RINEX
epochs, computes elevation/azimuth, and writes a sorted dataset for downstream analysis.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Literal

import numpy as np
import pandas as pd

from src.geometry.coordinates import (
    calculate_elevation_azimuth,
    ecef_to_geodetic,
)
from src.parsers.rinex_parser import parse_rinex_obs
from src.parsers.sp3_parser import parse_sp3

DEFAULT_OUTPUT_DIR = os.path.join("data", "processed")
DEFAULT_OUTPUT_STEM = "final_rnx_sp3_merged"
OUTPUT_COLUMNS = [
    "epoch",
    "constellation",
    "satID",
    "elevation",
    "azimuth",
    "obsType",
    "obsValue",
]
OutputFormat = Literal["parquet", "csv"]


def build_output_stem_from_rinex_path(rinex_path: str) -> str:
    """
    Build an output filename stem from a RINEX v3 observation filename.

    Example
    -------
    PTLD00AUS_R_20220010000_01D_30S_MO.rnx -> PTLD00AUS_2022001_30S
    """
    basename = os.path.splitext(os.path.basename(rinex_path))[0]
    parts = basename.split("_")

    if len(parts) >= 5:
        site = parts[0]
        start_time = parts[2]
        interval = parts[4]

        if (
            len(start_time) >= 7
            and start_time[:4].isdigit()
            and start_time[4:7].isdigit()
        ):
            year_doy = f"{start_time[:4]}{start_time[4:7]}"
            return f"{site}_{year_doy}_{interval}"

    return DEFAULT_OUTPUT_STEM


def _lagrange_interpolate_xyz(
    x_nodes: np.ndarray,
    coords: np.ndarray,
    x_target: float,
) -> tuple[float, float, float]:
    """Interpolate X, Y, Z at once using the same Lagrange nodes."""
    n = len(x_nodes)
    result = np.zeros(3, dtype=float)

    for i in range(n):
        weight = 1.0
        for j in range(n):
            if i != j:
                weight *= (x_target - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += coords[:, i] * weight

    return float(result[0]), float(result[1]), float(result[2])


def _select_neighbor_indices(n_low: int, n_high: int, total_sp3: int) -> list[int] | None:
    if n_low >= 5 and n_high >= 5:
        return list(range(n_low - 5, n_low)) + list(range(n_low, n_low + 5))

    if n_low < 5 and n_high >= 5:
        return list(range(0, 10))

    if n_low >= 5 and n_high < 5:
        return list(range(total_sp3 - 10, total_sp3))

    return None


def _prepare_antenna_enu_reference(ant_pos: tuple[float, float, float]) -> dict[str, float]:
    lat, lon, _ = ecef_to_geodetic(ant_pos[0], ant_pos[1], ant_pos[2])
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    return {
        "x_ant": ant_pos[0],
        "y_ant": ant_pos[1],
        "z_ant": ant_pos[2],
        "sin_lat": np.sin(lat_rad),
        "cos_lat": np.cos(lat_rad),
        "sin_lon": np.sin(lon_rad),
        "cos_lon": np.cos(lon_rad),
    }


def _ecef_to_enu_fast(
    x_sat: float,
    y_sat: float,
    z_sat: float,
    ref: dict[str, float],
) -> tuple[float, float, float]:
    dx = x_sat - ref["x_ant"]
    dy = y_sat - ref["y_ant"]
    dz = z_sat - ref["z_ant"]

    e = -ref["sin_lon"] * dx + ref["cos_lon"] * dy
    n = (
        -ref["sin_lat"] * ref["cos_lon"] * dx
        - ref["sin_lat"] * ref["sin_lon"] * dy
        + ref["cos_lat"] * dz
    )
    u = (
        ref["cos_lat"] * ref["cos_lon"] * dx
        + ref["cos_lat"] * ref["sin_lon"] * dy
        + ref["sin_lat"] * dz
    )
    return e, n, u


def _process_satellite(
    target_sat: str,
    sat_sp3_df: pd.DataFrame,
    sat_rnx_df: pd.DataFrame,
    enu_ref: dict[str, float],
) -> list[dict]:
    if sat_rnx_df.empty:
        return []

    sp3_base_time = sat_sp3_df["epoch"].iloc[0]
    sp3_seconds = (sat_sp3_df["epoch"] - sp3_base_time).dt.total_seconds().to_numpy()
    sp3_coords = sat_sp3_df[["X", "Y", "Z"]].to_numpy(dtype=float).T
    total_sp3 = len(sp3_seconds)

    rnx_by_epoch = {
        epoch: group[["constellation", "obsType", "obsValue"]].to_numpy()
        for epoch, group in sat_rnx_df.groupby("epoch", sort=True)
    }

    results: list[dict] = []

    for rnx_epoch, rnx_rows in rnx_by_epoch.items():
        target_seconds = (pd.Timestamp(rnx_epoch) - sp3_base_time).total_seconds()

        n_low = int(np.searchsorted(sp3_seconds, target_seconds, side="right"))
        n_high = total_sp3 - n_low

        chosen_indices = _select_neighbor_indices(n_low, n_high, total_sp3)
        if chosen_indices is None or len(chosen_indices) < 10:
            continue

        x_nodes = sp3_seconds[chosen_indices]
        y_matrix = sp3_coords[:, chosen_indices]

        interp_x, interp_y, interp_z = _lagrange_interpolate_xyz(
            x_nodes, y_matrix, target_seconds
        )

        e, n, u = _ecef_to_enu_fast(interp_x, interp_y, interp_z, enu_ref)
        elevation, azimuth = calculate_elevation_azimuth(e, n, u)

        for constellation, obs_type, obs_value in rnx_rows:
            results.append(
                {
                    "epoch": rnx_epoch,
                    "constellation": constellation,
                    "satID": target_sat,
                    "elevation": elevation,
                    "azimuth": azimuth,
                    "obsType": obs_type,
                    "obsValue": obs_value,
                }
            )

    return results


def _resolve_output_path(
    output_path: str | None,
    output_format: OutputFormat,
    output_stem: str,
) -> str:
    extension = ".parquet" if output_format == "parquet" else ".csv"

    if output_path is None:
        return os.path.join(DEFAULT_OUTPUT_DIR, f"{output_stem}{extension}")

    root, ext = os.path.splitext(output_path)
    if not ext or ext.lower() != extension:
        return root + extension

    return output_path


def _save_merged_dataframe(
    df: pd.DataFrame,
    output_path: str,
    output_format: OutputFormat,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_format == "parquet":
        df.to_parquet(output_path, index=False, compression="zstd")
    else:
        df.to_csv(output_path, index=False)


def merge_rinex_sp3(
    sp3_path: str,
    rinex_path: str,
    output_path: str | None = None,
    *,
    output_format: OutputFormat = "parquet",
    verbose: bool = True,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Interpolate SP3 orbits onto RINEX epochs and merge with SNR observations.

    Parameters
    ----------
    sp3_path : str
        Path to the SP3 orbit file.
    rinex_path : str
        Path to the RINEX observation file (.rnx).
    output_path : str, optional
        Destination file path. Defaults to
        ``data/processed/<rinex-derived-name>.<format>``.
    output_format : {"parquet", "csv"}, default "parquet"
        Output file format. Use ``"csv"`` for plain CSV export.
    verbose : bool
        Print progress and timing information.
    progress_callback : callable, optional
        Called as ``callback(sat_id, current_index, total_count)`` for each satellite.

    Returns
    -------
    pandas.DataFrame
        Merged dataset with columns ``epoch``, ``constellation``, ``satID``,
        ``elevation``, ``azimuth``, ``obsType``, and ``obsValue``,
        sorted by epoch, satID, and obsType.
    """
    start_time = time.perf_counter()

    if verbose:
        print(">>> SP3 dosyası okunuyor...")
    sp3_df = parse_sp3(sp3_path)
    sp3_df = sp3_df.sort_values(["satID", "epoch"]).reset_index(drop=True)

    if verbose:
        print(">>> RINEX dosyası okunuyor...")
    rnx_df, rinex_header = parse_rinex_obs(rinex_path)
    rnx_df = rnx_df.sort_values(["satID", "epoch"]).reset_index(drop=True)

    ant_pos = rinex_header["approx_pos"]
    enu_ref = _prepare_antenna_enu_reference(ant_pos)

    sp3_by_sat = {
        sat_id: group.reset_index(drop=True)
        for sat_id, group in sp3_df.groupby("satID", sort=False)
    }
    rnx_by_sat = {
        sat_id: group.reset_index(drop=True)
        for sat_id, group in rnx_df.groupby("satID", sort=False)
    }

    unique_sp3_sat_list = sp3_df["satID"].unique()
    total_sats = len(unique_sp3_sat_list)

    if verbose:
        print(f"Tespit edilen benzersiz uydu sayısı: {total_sats}")

    interpolated_results: list[dict] = []

    for sat_idx, target_sat in enumerate(unique_sp3_sat_list, start=1):
        if progress_callback is not None:
            progress_callback(target_sat, sat_idx, total_sats)

        sat_sp3_df = sp3_by_sat.get(target_sat)
        if sat_sp3_df is None:
            continue

        sat_rnx_df = rnx_by_sat.get(target_sat, pd.DataFrame())
        interpolated_results.extend(
            _process_satellite(target_sat, sat_sp3_df, sat_rnx_df, enu_ref)
        )

    final_df = pd.DataFrame(interpolated_results, columns=OUTPUT_COLUMNS)
    final_df = final_df.sort_values(
        by=["epoch", "satID", "obsType"],
        kind="mergesort",
    ).reset_index(drop=True)

    output_stem = build_output_stem_from_rinex_path(rinex_path)
    resolved_output_path = _resolve_output_path(
        output_path, output_format, output_stem
    )
    _save_merged_dataframe(final_df, resolved_output_path, output_format)

    if verbose:
        elapsed = time.perf_counter() - start_time
        print("\n--- ENTERPOLASYON VE BİRLEŞTİRME TAMAMLANDI ---")
        print(final_df.head())
        print(
            f"\nBaşarılı! Toplam {len(final_df)} adet saniyelik veri enterpole edilerek SNR ile birleştirildi."
        )
        print(f"Çıktı dosyası ({output_format}): {resolved_output_path}")
        print(f"Toplam çalışma süresi: {elapsed:.2f} saniye ({elapsed / 60:.2f} dakika)")

    return final_df
