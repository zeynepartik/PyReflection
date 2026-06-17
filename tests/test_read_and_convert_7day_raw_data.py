import os
import sys

# Ensure the project root is importable when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.analysis.process_data import batch_merge_to_parquet

paths = batch_merge_to_parquet(
    station="PTLD00AUS",
    sp3_product="COD0MGXFIN",
    start_date="2022-01-01",
    end_date="2022-01-07",
)
# -> data/processed/PTLD00AUS_2022001_30S.parquet ... 2022007 (7 dosya)