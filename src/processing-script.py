"""
processing.py
=============
Reusable data-loading, cleaning, and feature-engineering functions for the
Heart Disease Prevalence Predictor project.

Usage
-----
    from src.processing import load_data, clean_data, engineer_features, get_X_y

These functions are called from the main notebook and can also be used
independently in prediction scripts or future pipelines.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATA_URL = (
    "https://breathecode.herokuapp.com/asset/internal-link"
    "?id=733&path=demographic_health_data.csv"
)

TARGET_COL = "Heart disease_prevalence"

# Columns that identify rows but carry no predictive signal
IDENTIFIER_COLS = [
    "STATE_NAME",
    "COUNTY_NAME",
    # cnty_fips is redundant with fips — drop it too
    "cnty_fips",
]

# Fine-grained age-percentage columns replaced by broader grouped features
FINE_AGE_COLS = [
    "0-9 y/o % of total pop",
    "10-19 y/o % of total pop",
    "20-29 y/o % of total pop",
    "30-39 y/o % of total pop",
    "40-49 y/o % of total pop",
    "50-59 y/o % of total pop",
    "60-69 y/o % of total pop",
    "70-79 y/o % of total pop",
    "80+ y/o % of total pop",
]

# Redundant / high-collinearity columns identified during EDA
REDUNDANT_COLS = [
    "county_pop2018_18 and older",   # overlaps with age-group percentages
    "anycondition_prevalence",        # aggregate of all disease prevalences
    "Percent of Population Aged 60+", # captured by age 60-79 + 80+ groups
]

# Leakage columns (CI bounds of the target — would not be available at predict time)
TARGET_CI_COLS = [
    "Heart disease_Upper 95% CI",
    "Heart disease_Lower 95% CI",
]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """
    Download the raw demographic-health CSV and return it as a DataFrame.

    Parameters
    ----------
    url : str
        Direct URL to the CSV file. Defaults to the project dataset.

    Returns
    -------
    pd.DataFrame
        Raw data — 3 140 rows × 108 columns.
    """
    df = pd.read_csv(url)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier, redundant, and zero-variance columns.

    Steps
    -----
    1. Remove pure identifier columns (state/county names, duplicate FIPS).
    2. Assert no nulls or duplicates (dataset is expected to be clean).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by :func:`load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame. Does **not** mutate the input.
    """
    df = df.copy()

    # Drop identifier columns that exist in this particular dataframe
    cols_to_drop = [c for c in IDENTIFIER_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Sanity checks (fail fast if data source changes)
    assert df.isna().sum().sum() == 0, "Unexpected nulls found after cleaning."
    assert df.duplicated().sum() == 0, "Unexpected duplicate rows found."

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create grouped age-band features and drop fine-grained / redundant columns.

    Age bands created
    -----------------
    * age 0-19 pct %   = 0-9 + 10-19
    * age 20-39 pct %  = 20-29 + 30-39
    * age 40-59 pct %  = 40-49 + 50-59
    * age 60-79 pct %  = 60-69 + 70-79
    * age 80+ pct %    = 80+

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from :func:`clean_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features and reduced collinearity.
        Does **not** mutate the input.
    """
    df = df.copy()

    # --- Age grouping ---------------------------------------------------
    df["age 0-19 pct %"]  = df["0-9 y/o % of total pop"]  + df["10-19 y/o % of total pop"]
    df["age 20-39 pct %"] = df["20-29 y/o % of total pop"] + df["30-39 y/o % of total pop"]
    df["age 40-59 pct %"] = df["40-49 y/o % of total pop"] + df["50-59 y/o % of total pop"]
    df["age 60-79 pct %"] = df["60-69 y/o % of total pop"] + df["70-79 y/o % of total pop"]
    df["age 80+ pct %"]   = df["80+ y/o % of total pop"]

    # Drop fine-grained age columns that were just combined
    fine_age_present = [c for c in FINE_AGE_COLS if c in df.columns]
    df = df.drop(columns=fine_age_present)

    # Drop other redundant columns identified in EDA
    redundant_present = [c for c in REDUNDANT_COLS if c in df.columns]
    df = df.drop(columns=redundant_present)

    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the processed DataFrame into features (X) and target (y).

    Drops the target column and its confidence-interval bounds, which
    would constitute data leakage at inference time.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame (output of :func:`engineer_features`).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector — ``Heart disease_prevalence``.
    """
    cols_to_exclude = [TARGET_COL] + [c for c in TARGET_CI_COLS if c in df.columns]
    X = df.drop(columns=cols_to_exclude)
    y = df[TARGET_COL]
    return X, y


def run_pipeline(url: str = DATA_URL) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function: load → clean → engineer → split in one call.

    Parameters
    ----------
    url : str
        Dataset URL. Defaults to the project dataset.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    df = load_data(url)
    df = clean_data(df)
    df = engineer_features(df)
    X, y = get_X_y(df)
    return X, y
