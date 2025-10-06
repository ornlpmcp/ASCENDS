"""CSV I/O, schema checks, train/test splits."""

from typing import Tuple, Optional
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    TimeSeriesSplit,
)


@dataclass
class SplitConfig:
    """Configuration for train/test data splitting."""

    method: str  # 'random', 'stratified', 'group', 'time'
    test_size: float = 0.2
    random_state: Optional[int] = None
    stratify_col: Optional[str] = None
    group_col: Optional[str] = None


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
        return train_test_split(
            df, test_size=cfg.test_size, random_state=cfg.random_state
        )

    elif cfg.method == "stratified":
        stratifier = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
        )
        train_idx, test_idx = next(stratifier.split(df, df[cfg.stratify_col]))
        return df.iloc[train_idx], df.iloc[test_idx]


def align_to_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    One-hot encode categoricals (drop_first=False), reindex to the given features (fill_value=0),
    return DataFrame with columns ordered as features.
    """
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
