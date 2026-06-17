from src.analysis.process_data import analyze_arcs

df = analyze_arcs(
    "data/processed/PTLD00AUS_2022001_30S.parquet",
    filters={"elev_ranges": [(2, 15)], "azim_ranges": [(130, 270)]},
    h_max=10.0,
    output_format="csv",  # "csv" de olabilir
)