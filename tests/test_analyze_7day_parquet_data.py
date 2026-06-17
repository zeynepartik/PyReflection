import os
import sys

# Ensure the project root is importable when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.process_data import batch_analyze_arcs

df = batch_analyze_arcs(
    station="PTLD00AUS",
    start_date="2022-01-01",
    end_date="2022-01-07",
    filters={"elev_ranges": [(2, 15)], "azim_ranges": [(130, 270)]},
    h_max=10.0,
    output_format="csv",  # "csv" de olabilir
)
# -> data/results/results_PTLD00AUS_2022001_2022007_30S.csv
