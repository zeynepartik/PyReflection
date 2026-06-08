import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.processing.rinex_sp3_merger import (
    DEFAULT_OUTPUT_STEM,
    build_output_stem_from_rinex_path,
)


def test_build_output_stem_from_rinex_v3_filename():
    rinex_path = "data/PTLD00AUS_R_20220010000_01D_30S_MO.rnx"
    assert build_output_stem_from_rinex_path(rinex_path) == "PTLD00AUS_2022001_30S"


def test_build_output_stem_fallback_for_non_standard_name():
    rinex_path = "data/custom_observation.rnx"
    assert build_output_stem_from_rinex_path(rinex_path) == DEFAULT_OUTPUT_STEM


if __name__ == "__main__":
    test_build_output_stem_from_rinex_v3_filename()
    test_build_output_stem_fallback_for_non_standard_name()
    print("All naming tests passed.")
