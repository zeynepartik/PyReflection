"""
GNSS-IR analysis pipeline over processed arc datasets.

Loads a processed parquet/CSV file (output of the RINEX/SP3 merger), applies
user-defined filters, then runs a Lomb-Scargle reflector-height analysis for
every unique (satID, obsType, arcNo) arc. The per-arc results are collected
into a tidy DataFrame and (optionally) written to ``data/results/``.

This module is intended to grow: additional analysis/aggregation helpers will
be added next to :func:`analyze_arcs`.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.analysis.filters import filter_dataset
from src.analysis.lomb_scargle import calculate_lombscargle, detrend_snr

OutputFormat = Literal["parquet", "csv"]

DEFAULT_RESULTS_DIR = os.path.join("data", "results")
RESULT_PREFIX = "results_"

# Columns that uniquely identify a single arc time series.
GROUP_KEYS = ["satID", "obsType", "arcNo"]

# Final result schema, in the exact requested order (epoch fields first).
RESULT_COLUMNS = [
    "epochMean",
    "epochStart",
    "epochEnd",
    "epochRange",  # arc duration in seconds (epochEnd - epochStart)
    "constellation",
    "satID",
    "obsType",
    "arcNo",
    "arcType",
    "azimStart",
    "azimEnd",
    "azimMean",
    "elevStart",
    "elevEnd",
    "elevRange",  # elevation span in degrees (max - min)
    "h",          # dominant reflector height from LSP (m)
    "peakAmp",    # maximum spectral amplitude from LSP
    "meanAmp",    # mean spectral amplitude over the grid
    "PBNR",       # peak-to-background noise ratio (peakAmp / meanAmp)
]

# Minimum number of points required to attempt a Lomb-Scargle analysis.
MIN_ARC_POINTS = 5


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """
    Circular mean of angles in degrees (robust to the 0/360 wrap-around).

    Reduces to the arithmetic mean when the angles do not cross the wrap point,
    which is the common case for a single GNSS arc azimuth track.
    """
    rad = np.radians(np.asarray(angles_deg, dtype=float))
    mean_rad = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.degrees(mean_rad) % 360.0)


def _analyze_single_arc(
    arc_df: pd.DataFrame,
    *,
    poly_degree: int,
    h_min: float,
    h_max: float,
    precision: float,
) -> dict[str, Any]:
    """
    Compute epoch/geometry aggregates and Lomb-Scargle metrics for one arc.

    Expects rows belonging to a single (satID, obsType, arcNo) arc. The arc is
    sorted by epoch so that the *Start/*End values follow the time direction.
    When the arc has too few points (or the LSP fails) the spectral metrics
    (``h``, ``peakAmp``, ``meanAmp``, ``PBNR``) are returned as NaN.
    """
    arc_df = arc_df.sort_values("epoch").reset_index(drop=True)

    epochs = arc_df["epoch"]
    elevations = arc_df["elevation"].to_numpy(dtype=float)
    azimuths = arc_df["azimuth"].to_numpy(dtype=float)

    epoch_start = epochs.iloc[0]
    epoch_end = epochs.iloc[-1]

    record: dict[str, Any] = {
        "epochMean": epochs.mean(),
        "epochStart": epoch_start,
        "epochEnd": epoch_end,
        "epochRange": (epoch_end - epoch_start).total_seconds(),
        "constellation": arc_df["constellation"].iloc[0],
        "satID": arc_df["satID"].iloc[0],
        "obsType": arc_df["obsType"].iloc[0],
        "arcNo": int(arc_df["arcNo"].iloc[0]),
        "arcType": int(arc_df["arcType"].iloc[0]),
        "azimStart": float(azimuths[0]),
        "azimEnd": float(azimuths[-1]),
        "azimMean": _circular_mean_deg(azimuths),
        "elevStart": float(elevations[0]),
        "elevEnd": float(elevations[-1]),
        "elevRange": float(elevations.max() - elevations.min()),
        "h": np.nan,
        "peakAmp": np.nan,
        "meanAmp": np.nan,
        "PBNR": np.nan,
    }

    if len(arc_df) < MIN_ARC_POINTS:
        return record

    wavelength = float(arc_df["wavelength"].iloc[0])
    snr_values = arc_df["obsValue"].to_numpy(dtype=float)

    detrended_snr = detrend_snr(elevations, snr_values, poly_degree=poly_degree)
    h_grid, spectral_amplitude = calculate_lombscargle(
        elevation=elevations,
        detrended_snr=detrended_snr,
        wavelength=wavelength,
        h_min=h_min,
        h_max=h_max,
        precision=precision,
    )

    if spectral_amplitude.size == 0:
        return record

    peak_idx = int(np.argmax(spectral_amplitude))
    peak_amp = float(spectral_amplitude[peak_idx])
    mean_amp = float(np.mean(spectral_amplitude))

    record["h"] = float(h_grid[peak_idx])
    record["peakAmp"] = peak_amp
    record["meanAmp"] = mean_amp
    record["PBNR"] = peak_amp / mean_amp if mean_amp != 0 else np.nan

    return record


def _resolve_result_path(
    input_path: str,
    output_dir: str,
    output_format: OutputFormat,
) -> str:
    """Build ``<output_dir>/results_<input-stem>.<ext>`` from the input file."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    extension = ".parquet" if output_format == "parquet" else ".csv"
    return os.path.join(output_dir, f"{RESULT_PREFIX}{stem}{extension}")


def _save_results(
    df: pd.DataFrame,
    output_path: str,
    output_format: OutputFormat,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_format == "parquet":
        df.to_parquet(output_path, index=False, compression="zstd")
    else:
        df.to_csv(output_path, index=False)


def analyze_arcs(
    input_path: str,
    *,
    filters: dict[str, Any] | None = None,
    poly_degree: int = 2,
    h_min: float = 0.0,
    h_max: float = 10.0,
    precision: float = 0.01,
    output_format: OutputFormat = "parquet",
    output_dir: str = DEFAULT_RESULTS_DIR,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a Lomb-Scargle reflector-height analysis over a processed arc dataset.

    Parameters
    ----------
    input_path : str
        Path to a processed dataset (parquet or CSV) produced by the
        RINEX/SP3 merger. Must contain at least the columns ``epoch``,
        ``satID``, ``obsType``, ``arcNo``, ``arcType``, ``wavelength``,
        ``obsValue``, ``elevation`` and ``azimuth``.
    filters : dict, optional
        Keyword arguments forwarded to :func:`src.analysis.filters.filter_dataset`
        (e.g. ``{"elev_ranges": [(5, 25)], "obs_types_include": ["S1C"]}``).
        When ``None`` no filtering is applied.
    poly_degree : int, default 2
        Polynomial degree used by :func:`detrend_snr`.
    h_min, h_max : float
        Reflector-height search bounds (metres) for the Lomb-Scargle grid.
    precision : float, default 0.01
        Reflector-height grid resolution (metres).
    output_format : {"parquet", "csv"}, default "parquet"
        Format of the saved result file.
    output_dir : str, default ``data/results``
        Directory where the result file is written.
    save : bool, default True
        When ``True`` the result DataFrame is written to disk as
        ``<output_dir>/results_<input-stem>.<ext>``.
    verbose : bool, default True
        Print progress information.

    Returns
    -------
    pandas.DataFrame
        One row per unique (satID, obsType, arcNo) arc, using ``RESULT_COLUMNS``.
    """
    if verbose:
        print(f">>> Isleniyor: {input_path}")

    extension = os.path.splitext(input_path)[1].lower()
    if extension == ".csv":
        df = pd.read_csv(input_path, parse_dates=["epoch"])
    else:
        df = pd.read_parquet(input_path)

    if filters:
        df = filter_dataset(df, **filters)
        if verbose:
            print(f">>> Filtreleme sonrasi satir sayisi: {len(df)}")

    if df.empty:
        if verbose:
            print(">>> Uyari: Filtreleme sonrasi veri kalmadi, bos sonuc donduruluyor.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    records: list[dict[str, Any]] = []
    grouped = df.groupby(GROUP_KEYS, sort=True)

    if verbose:
        print(f">>> {grouped.ngroups} adet benzersiz arc (satID/obsType/arcNo) analiz ediliyor...")

    for _, arc_df in grouped:
        records.append(
            _analyze_single_arc(
                arc_df,
                poly_degree=poly_degree,
                h_min=h_min,
                h_max=h_max,
                precision=precision,
            )
        )

    result_df = pd.DataFrame(records, columns=RESULT_COLUMNS)
    result_df = result_df.sort_values("epochMean", kind="mergesort").reset_index(drop=True)

    if save:
        output_path = _resolve_result_path(input_path, output_dir, output_format)
        _save_results(result_df, output_path, output_format)
        if verbose:
            print(f">>> Sonuclar kaydedildi ({output_format}): {output_path}")

    return result_df
