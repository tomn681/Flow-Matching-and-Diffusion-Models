from __future__ import annotations

import os

import pandas as pd


def lot_id(df: pd.DataFrame, case_column: str, number_column: str) -> pd.DataFrame:
    """
    Generate unique identifiers for each multi-file image split lot.
    """
    df = df.copy()
    grouped = df.groupby(case_column)

    for case, group in grouped:
        for idx, row in group.iterrows():
            files = row[number_column]
            if not isinstance(files, (list, tuple)) or not files:
                continue
            first_elem = os.path.basename(files[0]).split(".")[0]
            last_elem = os.path.basename(files[-1]).split(".")[0]
            new_name = f"I{case}S{idx}F{first_elem}T{last_elem}C{len(files)}"
            df.at[idx, case_column] = new_name
    return df
