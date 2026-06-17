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
from datetime import date, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.analysis.filters import filter_dataset
from src.analysis.lomb_scargle import calculate_lombscargle, detrend_snr
from src.processing.rinex_sp3_merger import (
    build_output_stem_from_rinex_path,
    merge_rinex_sp3,
)

OutputFormat = Literal["parquet", "csv"]

DEFAULT_RESULTS_DIR = os.path.join("data", "results")
RESULT_PREFIX = "results_"

# Default locations for raw inputs and processed outputs.
DEFAULT_DATA_DIR = "data"
DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")

# Default observation/orbit sampling intervals encoded in the file names.
DEFAULT_RINEX_INTERVAL = "30S"
DEFAULT_SP3_INTERVAL = "05M"

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

# Default quality-control thresholds applied to a results file.
DEFAULT_QC_PBNR_MIN = 4.0
DEFAULT_QC_ELEV_RANGE_MIN = 5.0
QC_SUFFIX = "_QC"


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


# ---------------------------------------------------------------------------
# Batch RINEX/SP3 -> parquet conversion
# ---------------------------------------------------------------------------


def _parse_date(value: str | date | datetime) -> date:
    """Normalize a date-like input to a :class:`datetime.date`.

    Accepts ``datetime``/``date`` objects or ISO strings such as
    ``"2022-01-01"``.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    raise TypeError(f"Desteklenmeyen tarih turu: {type(value)!r}")


def _iter_year_doy(start_date: date, end_date: date) -> list[tuple[int, int]]:
    """Return a list of ``(year, day_of_year)`` for every day in the range.

    The range is inclusive on both ends.
    """
    if end_date < start_date:
        raise ValueError(
            f"Bitis tarihi ({end_date}) baslangic tarihinden ({start_date}) once olamaz."
        )

    days: list[tuple[int, int]] = []
    current = start_date
    while current <= end_date:
        days.append((current.year, current.timetuple().tm_yday))
        current += timedelta(days=1)
    return days


def _build_rinex_filename(station: str, year: int, doy: int, interval: str) -> str:
    """Build a RINEX v3 file name, e.g. ``PTLD00AUS_R_20220010000_01D_30S_MO.rnx``."""
    return f"{station}_R_{year}{doy:03d}0000_01D_{interval}_MO.rnx"


def _build_sp3_filename(product: str, year: int, doy: int, interval: str) -> str:
    """Build an SP3 file name, e.g. ``COD0MGXFIN_20220010000_01D_05M_ORB.SP3``."""
    return f"{product}_{year}{doy:03d}0000_01D_{interval}_ORB.SP3"


def batch_merge_to_parquet(
    station: str,
    sp3_product: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_PROCESSED_DIR,
    output_format: OutputFormat = "parquet",
    rinex_interval: str = DEFAULT_RINEX_INTERVAL,
    sp3_interval: str = DEFAULT_SP3_INTERVAL,
    skip_existing: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Batch-convert RINEX/SP3 pairs into processed parquet (or CSV) files.

    For each day in the inclusive ``[start_date, end_date]`` range, the matching
    RINEX observation file and SP3 orbit file are located under ``data_dir`` and
    merged via :func:`src.processing.rinex_sp3_merger.merge_rinex_sp3`. Each
    merged dataset is written to ``output_dir`` (``data/processed`` by default).

    Parameters
    ----------
    station : str
        RINEX station/site name, e.g. ``"PTLD00AUS"``.
    sp3_product : str
        SP3 product name, e.g. ``"COD0MGXFIN"``.
    start_date, end_date : str | datetime.date | datetime.datetime
        Inclusive date range. Strings must be ISO formatted (``"YYYY-MM-DD"``).
    data_dir : str, default ``"data"``
        Directory holding the raw RINEX (``.rnx``) and SP3 (``.SP3``) files.
    output_dir : str, default ``data/processed``
        Directory where the processed files are written.
    output_format : {"parquet", "csv"}, default "parquet"
        Output file format.
    rinex_interval, sp3_interval : str
        Sampling interval tokens encoded in the file names (``"30S"``/``"05M"``).
    skip_existing : bool, default False
        When ``True`` days whose output file already exists are skipped.
    verbose : bool, default True
        Print per-day progress information.

    Returns
    -------
    list[str]
        Paths of the processed files that were written (or already existed when
        ``skip_existing`` is ``True``).
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    days = _iter_year_doy(start, end)

    extension = ".parquet" if output_format == "parquet" else ".csv"
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(
            f">>> Toplu donusum baslatiliyor: istasyon={station}, "
            f"sp3={sp3_product}, {start} -> {end} ({len(days)} gun)"
        )

    written_paths: list[str] = []

    for index, (year, doy) in enumerate(days, start=1):
        rinex_name = _build_rinex_filename(station, year, doy, rinex_interval)
        sp3_name = _build_sp3_filename(sp3_product, year, doy, sp3_interval)
        rinex_path = os.path.join(data_dir, rinex_name)
        sp3_path = os.path.join(data_dir, sp3_name)

        if verbose:
            print(f"\n[{index}/{len(days)}] {year} DOY {doy:03d}")

        output_stem = build_output_stem_from_rinex_path(rinex_path)
        output_path = os.path.join(output_dir, f"{output_stem}{extension}")

        if skip_existing and os.path.exists(output_path):
            if verbose:
                print(f">>> Atlandi (zaten mevcut): {output_path}")
            written_paths.append(output_path)
            continue

        missing = [p for p in (rinex_path, sp3_path) if not os.path.exists(p)]
        if missing:
            if verbose:
                for path in missing:
                    print(f">>> Uyari: Dosya bulunamadi, gun atlaniyor: {path}")
            continue

        merge_rinex_sp3(
            sp3_path=sp3_path,
            rinex_path=rinex_path,
            output_path=output_path,
            output_format=output_format,
            verbose=verbose,
        )
        written_paths.append(output_path)

    if verbose:
        print(
            f"\n>>> Toplu donusum tamamlandi: {len(written_paths)}/{len(days)} "
            f"dosya islendi. Cikti dizini: {output_dir}"
        )

    return written_paths


# ---------------------------------------------------------------------------
# Batch analysis over processed parquet files -> single combined results file
# ---------------------------------------------------------------------------


def _build_processed_stem(station: str, year: int, doy: int, interval: str) -> str:
    """Build a processed-file stem, e.g. ``PTLD00AUS_2022001_30S``.

    Mirrors :func:`src.processing.rinex_sp3_merger.build_output_stem_from_rinex_path`.
    """
    return f"{station}_{year}{doy:03d}_{interval}"


def _resolve_combined_result_path(
    station: str,
    days: list[tuple[int, int]],
    interval: str,
    output_dir: str,
    output_format: OutputFormat,
    output_name: str | None,
) -> str:
    """Build the path of the single combined results file for a date range."""
    extension = ".parquet" if output_format == "parquet" else ".csv"

    if output_name is not None:
        stem = os.path.splitext(output_name)[0]
        return os.path.join(output_dir, f"{stem}{extension}")

    start_year, start_doy = days[0]
    end_year, end_doy = days[-1]
    stem = (
        f"{RESULT_PREFIX}{station}_"
        f"{start_year}{start_doy:03d}_{end_year}{end_doy:03d}_{interval}"
    )
    return os.path.join(output_dir, f"{stem}{extension}")


def batch_analyze_arcs(
    station: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    input_dir: str = DEFAULT_PROCESSED_DIR,
    input_format: OutputFormat = "parquet",
    rinex_interval: str = DEFAULT_RINEX_INTERVAL,
    filters: dict[str, Any] | None = None,
    poly_degree: int = 2,
    h_min: float = 0.0,
    h_max: float = 10.0,
    precision: float = 0.01,
    output_format: OutputFormat = "parquet",
    output_dir: str = DEFAULT_RESULTS_DIR,
    output_name: str | None = None,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze processed parquet files over a date range into one results file.

    For each day in the inclusive ``[start_date, end_date]`` range, the matching
    processed file (output of :func:`batch_merge_to_parquet`) is located under
    ``input_dir`` and analyzed via :func:`analyze_arcs`. The per-day results are
    concatenated into a single DataFrame, sorted by ``epochMean``, and written as
    one combined results file to ``output_dir`` (``data/results`` by default).

    Parameters
    ----------
    station : str
        RINEX station/site name, e.g. ``"PTLD00AUS"``.
    start_date, end_date : str | datetime.date | datetime.datetime
        Inclusive date range. Strings must be ISO formatted (``"YYYY-MM-DD"``).
    input_dir : str, default ``data/processed``
        Directory holding the processed parquet/CSV files to analyze.
    input_format : {"parquet", "csv"}, default "parquet"
        Format of the processed input files.
    rinex_interval : str, default ``"30S"``
        Sampling interval token encoded in the processed file names.
    filters : dict, optional
        Forwarded to :func:`src.analysis.filters.filter_dataset` (e.g.
        ``{"elev_ranges": [(2, 15)], "azim_ranges": [(130, 270)]}``).
    poly_degree : int, default 2
        Polynomial degree used by the SNR detrending step.
    h_min, h_max : float
        Reflector-height search bounds (metres) for the Lomb-Scargle grid.
    precision : float, default 0.01
        Reflector-height grid resolution (metres).
    output_format : {"parquet", "csv"}, default "parquet"
        Format of the combined results file.
    output_dir : str, default ``data/results``
        Directory where the combined results file is written.
    output_name : str, optional
        Custom file name (extension optional) for the combined results file.
        When ``None`` a name like
        ``results_<station>_<startYearDOY>_<endYearDOY>_<interval>`` is used.
    save : bool, default True
        When ``True`` the combined results DataFrame is written to disk.
    verbose : bool, default True
        Print per-day progress information.

    Returns
    -------
    pandas.DataFrame
        Combined per-arc results for every analyzed day, using ``RESULT_COLUMNS``.
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    days = _iter_year_doy(start, end)

    input_extension = ".parquet" if input_format == "parquet" else ".csv"

    if verbose:
        print(
            f">>> Toplu analiz baslatiliyor: istasyon={station}, "
            f"{start} -> {end} ({len(days)} gun)"
        )

    per_day_results: list[pd.DataFrame] = []

    for index, (year, doy) in enumerate(days, start=1):
        stem = _build_processed_stem(station, year, doy, rinex_interval)
        input_path = os.path.join(input_dir, f"{stem}{input_extension}")

        if verbose:
            print(f"\n[{index}/{len(days)}] {year} DOY {doy:03d}")

        if not os.path.exists(input_path):
            if verbose:
                print(f">>> Uyari: Dosya bulunamadi, gun atlaniyor: {input_path}")
            continue

        day_result = analyze_arcs(
            input_path,
            filters=filters,
            poly_degree=poly_degree,
            h_min=h_min,
            h_max=h_max,
            precision=precision,
            save=False,
            verbose=verbose,
        )
        per_day_results.append(day_result)

    if not per_day_results:
        if verbose:
            print(">>> Uyari: Hicbir gun analiz edilemedi, bos sonuc donduruluyor.")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    combined_df = pd.concat(per_day_results, ignore_index=True)
    combined_df = combined_df.sort_values(
        "epochMean", kind="mergesort"
    ).reset_index(drop=True)

    if save:
        output_path = _resolve_combined_result_path(
            station, days, rinex_interval, output_dir, output_format, output_name
        )
        _save_results(combined_df, output_path, output_format)
        if verbose:
            print(
                f"\n>>> Birlesik sonuclar kaydedildi ({output_format}): {output_path}"
            )

    if verbose:
        print(
            f">>> Toplu analiz tamamlandi: {len(per_day_results)}/{len(days)} gun, "
            f"toplam {len(combined_df)} arc."
        )

    return combined_df


# ---------------------------------------------------------------------------
# Quality control over a results file
# ---------------------------------------------------------------------------


def _read_results_file(input_path: str) -> pd.DataFrame:
    """Read a results file as a DataFrame, inferring format from the extension."""
    extension = os.path.splitext(input_path)[1].lower()
    if extension == ".csv":
        return pd.read_csv(input_path)
    return pd.read_parquet(input_path)


def _resolve_qc_path(input_path: str, output_dir: str | None) -> str:
    """Insert the ``_QC`` suffix before the extension of ``input_path``."""
    directory, filename = os.path.split(input_path)
    stem, extension = os.path.splitext(filename)
    target_dir = output_dir if output_dir is not None else directory
    return os.path.join(target_dir, f"{stem}{QC_SUFFIX}{extension}")


def apply_quality_control(
    input_path: str,
    *,
    pbnr_min: float = DEFAULT_QC_PBNR_MIN,
    elev_range_min: float = DEFAULT_QC_ELEV_RANGE_MIN,
    output_dir: str | None = None,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter a results file by quality thresholds and save a ``_QC`` copy.

    Keeps only the rows where ``PBNR >= pbnr_min`` **and**
    ``elevRange >= elev_range_min``. Rows with NaN in either column are dropped
    (a NaN comparison evaluates to ``False``). The filtered table is written next
    to the input file (or under ``output_dir``) with a ``_QC`` suffix added
    before the extension, keeping the original file format.

    Parameters
    ----------
    input_path : str
        Path to a results file (parquet or CSV) produced by the analysis steps.
    pbnr_min : float, default 4.0
        Minimum peak-to-background noise ratio to keep a row.
    elev_range_min : float, default 5.0
        Minimum elevation span (degrees) to keep a row.
    output_dir : str, optional
        Directory for the ``_QC`` output. Defaults to the input file directory.
    save : bool, default True
        When ``True`` the filtered DataFrame is written to disk.
    verbose : bool, default True
        Print progress information.

    Returns
    -------
    pandas.DataFrame
        The quality-controlled subset of the input results.
    """
    if verbose:
        print(f">>> Kalite kontrol uygulaniyor: {input_path}")

    df = _read_results_file(input_path)
    total_rows = len(df)

    mask = (df["PBNR"] >= pbnr_min) & (df["elevRange"] >= elev_range_min)
    qc_df = df[mask].reset_index(drop=True)

    if verbose:
        print(
            f">>> Kalite kontrol sonrasi: {len(qc_df)}/{total_rows} satir tutuldu "
            f"(PBNR >= {pbnr_min}, elevRange >= {elev_range_min})."
        )

    if save:
        output_format: OutputFormat = (
            "csv" if os.path.splitext(input_path)[1].lower() == ".csv" else "parquet"
        )
        output_path = _resolve_qc_path(input_path, output_dir)
        _save_results(qc_df, output_path, output_format)
        if verbose:
            print(f">>> Kalite kontrol dosyasi kaydedildi: {output_path}")

    return qc_df
