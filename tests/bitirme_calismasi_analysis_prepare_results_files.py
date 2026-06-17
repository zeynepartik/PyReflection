import os
import sys

# Ensure the project root is importable when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.process_data import apply_quality_control, batch_analyze_arcs

STATION = "PTLD00AUS"
START_DATE = "2022-01-01"
END_DATE = "2022-01-07"
RESULTS_DIR = "data/results"

# Filters shared by both wavelength groups.
COMMON_FILTERS = {
    "elev_ranges": [(2, 15)],
    "azim_ranges": [(130, 270)],
    "obs_types_exclude": ["S1W", "S2W"],
}

# Common Lomb-Scargle / detrending settings.
LSP_SETTINGS = {
    "poly_degree": 2,
    "h_min": 0.0,
    "h_max": 10.0,
    "precision": 0.01,
}

# Two wavelength groups: short (SWG) and long (LWG).
WAVELENGTH_GROUPS = {
    "SWG": [(0.00, 0.20)],
    "LWG": [(0.20, 0.50)],
}


def run_group(group_name: str, wavelength_ranges: list[tuple[float, float]]) -> None:
    output_name = (
        f"results_{STATION}_2022001_2022007_30S_{group_name}"
    )

    batch_analyze_arcs(
        station=STATION,
        start_date=START_DATE,
        end_date=END_DATE,
        filters={**COMMON_FILTERS, "wavelength_ranges": wavelength_ranges},
        output_format="csv",
        output_dir=RESULTS_DIR,
        output_name=output_name,
        **LSP_SETTINGS,
    )

    results_path = os.path.join(RESULTS_DIR, f"{output_name}.csv")
    apply_quality_control(results_path, pbnr_min=4.0, elev_range_min=5.0)


if __name__ == "__main__":
    for name, ranges in WAVELENGTH_GROUPS.items():
        print(f"\n===== {name} grubu analizi =====")
        run_group(name, ranges)
