"""CSV I/O, schema checks, train/test splits."""

import logging
import re
from typing import Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    TimeSeriesSplit,
)

logger = logging.getLogger(__name__)
NON_ASCII_COLUMN_MESSAGE = (
    "Some columns contain non-ASCII characters; use ASCII column names for best compatibility."
)


@dataclass
class SplitConfig:
    """Configuration for train/test data splitting."""

    method: str  # 'random', 'stratified', 'group', 'time'
    test_size: float = 0.2
    random_state: Optional[int] = None
    stratify_col: Optional[str] = None
    group_col: Optional[str] = None


@dataclass
class PreparedFeatureFrame:
    """Numeric feature frame prepared for analysis or GUI model fitting."""

    frame: pd.DataFrame
    used_inputs: list[str]
    skipped_identifier_inputs: list[str]
    skipped_non_numeric_inputs: list[str]
    rows_in: int
    rows_used: int


def is_likely_identifier_column(column: object) -> bool:
    """Return True for common identifier/index columns that should not be modeled."""
    text = str(column).strip()
    lower = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    compact = normalized.replace("_", "")

    if lower.startswith("unnamed:"):
        return True
    if normalized in {
        "id",
        "index",
        "row",
        "row_id",
        "sample_id",
        "specimen_id",
        "record_id",
        "run_id",
        "case_id",
    }:
        return True
    return normalized.endswith("_id") or compact in {"sampleid", "specimenid", "recordid", "runid", "caseid"}


def prepare_numeric_features(
    df: pd.DataFrame,
    inputs: list[str],
    target: str,
    task: str = "r",
) -> PreparedFeatureFrame:
    """Read all columns but keep only numeric non-ID inputs for analysis/model fitting.

    Classification targets may remain categorical strings. Regression targets are
    coerced to numeric so invalid target values are dropped before fitting.
    """
    task_key = (task or "r").lower()
    skipped_identifier: list[str] = []
    skipped_non_numeric: list[str] = []
    used_inputs: list[str] = []
    prepared: dict[str, pd.Series] = {}

    for column in inputs:
        if column not in df.columns:
            continue
        if is_likely_identifier_column(column):
            skipped_identifier.append(column)
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if int(numeric.notna().sum()) == 0:
            skipped_non_numeric.append(column)
            continue
        prepared[column] = numeric
        used_inputs.append(column)

    if target in df.columns:
        if task_key == "c":
            prepared[target] = df[target]
        else:
            prepared[target] = pd.to_numeric(df[target], errors="coerce")

    frame = pd.DataFrame(prepared, index=df.index)
    rows_in = len(frame)
    frame = frame.dropna(axis=0, how="any")
    rows_used = len(frame)
    numeric_inputs = [column for column in used_inputs if column in frame.columns]
    if numeric_inputs:
        frame.loc[:, numeric_inputs] = frame.loc[:, numeric_inputs].astype("float64")
    if task_key != "c" and target in frame.columns:
        frame.loc[:, target] = frame.loc[:, target].astype("float64")

    return PreparedFeatureFrame(
        frame=frame,
        used_inputs=used_inputs,
        skipped_identifier_inputs=skipped_identifier,
        skipped_non_numeric_inputs=skipped_non_numeric,
        rows_in=rows_in,
        rows_used=rows_used,
    )


def split_train_test(
    df: pd.DataFrame, target: str, cfg: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.

    Args:
        df: The DataFrame to split.
        target: The target column for prediction.
        cfg: Configuration for the split.

    Returns:
        A tuple containing the train and test DataFrames.

    Raises:
        ValueError: If the configuration is invalid or required columns are missing.
    """
    if cfg.method not in ["random", "stratified", "group", "time"]:
        raise ValueError(
            f"Invalid split method: {cfg.method}. Choose from 'random', 'stratified', 'group', 'time'."
        )

    if cfg.method == "stratified" and cfg.stratify_col is None:
        raise ValueError("Stratified split requires 'stratify_col' to be set.")

    if cfg.method == "group" and cfg.group_col is None:
        raise ValueError("Group split requires 'group_col' to be set.")

    if cfg.method == "random":
        stratify = (
            df[cfg.stratify_col]
            if cfg.stratify_col and cfg.stratify_col in df.columns
            else None
        )
        if stratify is not None:
            class_counts = stratify.value_counts(dropna=True)
            if len(class_counts) < 2 or int(class_counts.min()) < 2:
                stratify = None
        return train_test_split(
            df,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify,
        )

    elif cfg.method == "stratified":
        stratifier = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
        )
        train_idx, test_idx = next(stratifier.split(df, df[cfg.stratify_col]))
        return df.iloc[train_idx], df.iloc[test_idx]

    # Group/time methods are implemented in the helper.
    return split_group_or_time(df, cfg)


def find_non_ascii_columns(columns) -> list[str]:
    """Return column names containing non-ASCII characters."""
    return [str(column) for column in columns if not str(column).isascii()]


def warn_non_ascii_columns(columns) -> list[str]:
    """Log and return non-ASCII column names for caller-visible notices."""
    non_ascii = find_non_ascii_columns(columns)
    if non_ascii:
        logger.warning("%s Columns: %s", NON_ASCII_COLUMN_MESSAGE, ", ".join(non_ascii))
    return non_ascii


def align_to_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    One-hot encode categoricals (drop_first=False), reindex to the given features (fill_value=0),
    return DataFrame with columns ordered as features.
    """
    warn_non_ascii_columns(df.columns)
    df_dum = pd.get_dummies(df, drop_first=False)
    return df_dum.reindex(columns=features, fill_value=0)


# TODO: Implement additional data processing functions if needed.


def split_group_or_time(
    df: pd.DataFrame, cfg: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data using group or time method based on configuration."""
    if cfg.method == "group":
        grouper = GroupShuffleSplit(
            n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
        )
        train_idx, test_idx = next(grouper.split(df, groups=df[cfg.group_col]))
        return df.iloc[train_idx], df.iloc[test_idx]

    elif cfg.method == "time":
        tscv = TimeSeriesSplit(n_splits=int(1 / cfg.test_size))
        train_idx, test_idx = next(tscv.split(df))
        return df.iloc[train_idx], df.iloc[test_idx]
