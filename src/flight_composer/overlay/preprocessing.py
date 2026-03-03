"""Preprocessing utilities for telemetry DataFrames.

This module contains functions that compute derived metrics from raw
telemetry data.  It is intended to be called once, right after CSV
ingestion and schema validation, so that all downstream overlay widgets
can rely on the enriched columns being present.
"""

from __future__ import annotations

import pandas as pd


def preprocess_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metric columns to a telemetry DataFrame.

    Currently computes:

    * **speedup_raw** – the ratio of *simulation* time elapsed to *video*
      time elapsed between consecutive frames
      (``timestamp_s.diff() / video_time_s.diff()``).  The first row is
      back-filled from the second row because ``diff()`` produces NaN for
      the initial entry.

    Parameters
    ----------
    df:
        A validated telemetry DataFrame that contains at least
        ``timestamp_s`` and ``video_time_s`` columns.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with the new column(s) added in-place.
    """
    df["speedup_raw"] = df["timestamp_s"].diff() / df["video_time_s"].diff()
    df["speedup_raw"] = df["speedup_raw"].bfill()

    return df
