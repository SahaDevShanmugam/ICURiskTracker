"""Canonical column names for parquet time-series tables (MIMIC CSV-derived)."""

TS_COLUMNS = [
    "stay_id",
    "subject_id",
    "charttime",
    "map",
    "sbp",
    "hr",
    "rr",
    "lactate",
    "age",
    "sex",  # 0/1 encoded
]

# Present in processed parquets for label derivation only — never use as model features.
LABEL_HELPER_COLUMNS = ["vaso_marker"]
