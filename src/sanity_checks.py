"""Lightweight guardrails for downstream analysis notebooks.

Every notebook after 01_main_model.ipynb reads CSV exports produced by an
upstream step. These checks catch the most common failure mode: running a
downstream notebook against a stale export from a previous model version
(e.g. an old 102-MSA run instead of the current 100-MSA panel), which
produces numbers that are wrong but don't obviously look wrong.
"""

import pandas as pd


def check_required_columns(df: pd.DataFrame, required: set[str], source_name: str) -> None:
    """Raise with a clear message if required columns are missing."""
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"STOPPING -- {source_name} is missing columns: {missing}. "
            f"Re-run the upstream notebook that produces this file."
        )


def check_msa_panel_size(
    df: pd.DataFrame,
    expected_n: int,
    source_name: str,
    msa_col: str = "MSA_Name",
) -> None:
    """Warn loudly if a CSV export doesn't reflect the expected MSA panel size.

    Doesn't raise, since a mismatch is sometimes intentional -- but it should
    never be silent.
    """
    n_msas = df[msa_col].nunique()
    if n_msas != expected_n:
        print(
            f"WARNING -- {source_name} contains {n_msas} MSAs, expected "
            f"{expected_n}. This may be a stale export from a different "
            f"model run -- verify before trusting downstream results."
        )
