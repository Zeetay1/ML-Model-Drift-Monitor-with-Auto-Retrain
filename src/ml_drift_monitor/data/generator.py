from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from ml_drift_monitor.config import DataGenerationConfig, FeatureDriftSpec, get_default_config
from ml_drift_monitor.logging_utils.logger import get_logger


logger = get_logger()


def _month_drift_multiplier(month: int, spec: FeatureDriftSpec) -> float:
    if spec.drift_start_month is None or month < spec.drift_start_month:
        return 0.0
    # Linearly increase magnitude from start month onwards.
    month_index = month - spec.drift_start_month + 1
    return float(month_index)


def generate_feature_schema(config: DataGenerationConfig) -> pd.DataFrame:
    """
    Return a DataFrame capturing the feature metadata and configured drift specs.
    """
    records = [asdict(spec) for spec in config.drift_specs]
    schema = pd.DataFrame.from_records(records)
    logger.info("Generated feature schema with %d features", len(schema))
    return schema


def _generate_base_numerical(feature_name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if feature_name == "age":
        return rng.normal(loc=40.0, scale=10.0, size=size)
    if feature_name == "income":
        return rng.lognormal(mean=10.5, sigma=0.5, size=size)
    if feature_name == "tenure":
        return rng.normal(loc=3.0, scale=1.0, size=size)
    if feature_name == "transactions_last_month":
        return rng.poisson(lam=3.0, size=size).astype(float)
    if feature_name == "score":
        return rng.normal(loc=0.0, scale=1.0, size=size)
    return rng.normal(loc=0.0, scale=1.0, size=size)


def _apply_numerical_drift(
    base: np.ndarray, month: int, spec: FeatureDriftSpec, rng: np.random.Generator
) -> np.ndarray:
    mult = _month_drift_multiplier(month, spec)
    if mult == 0.0 or spec.drift_type == "none":
        return base

    if spec.drift_type == "mean_shift":
        shift = spec.drift_magnitude * mult
        return base + shift

    if spec.drift_type == "variance_shift":
        factor = 1.0 + spec.drift_magnitude * mult
        mean = np.mean(base)
        return mean + (base - mean) * factor

    # Non-numerical drift types are ignored here.
    return base


def _generate_base_categorical(feature_name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if feature_name == "region":
        categories = np.array(["north", "south", "east", "west"])
        probs = np.array([0.25, 0.25, 0.25, 0.25])
    elif feature_name == "channel":
        categories = np.array(["web", "mobile", "branch"])
        probs = np.array([0.4, 0.4, 0.2])
    else:
        categories = np.array(["A", "B"])
        probs = np.array([0.5, 0.5])
    return rng.choice(categories, size=size, p=probs)


def _apply_categorical_drift(
    base: np.ndarray, month: int, spec: FeatureDriftSpec, rng: np.random.Generator
) -> np.ndarray:
    mult = _month_drift_multiplier(month, spec)
    if mult == 0.0 or spec.drift_type != "category_shift":
        return base

    values, counts = np.unique(base, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    # Push probability mass toward the first category for simplicity.
    shift = spec.drift_magnitude * mult
    probs[0] = min(0.99, probs[0] + shift)
    remaining = 1.0 - probs[0]
    if len(probs) > 1:
        probs[1:] = remaining / (len(probs) - 1)
    probs = probs / probs.sum()
    return rng.choice(values, size=base.size, p=probs)


def _generate_features_for_month(
    month: int, config: DataGenerationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    data: Dict[str, np.ndarray] = {}
    for spec in config.drift_specs:
        if spec.feature_type == "numerical":
            base = _generate_base_numerical(spec.feature_name, config.rows_per_month, rng)
            data[spec.feature_name] = _apply_numerical_drift(base, month, spec, rng)
        else:
            base = _generate_base_categorical(spec.feature_name, config.rows_per_month, rng)
            data[spec.feature_name] = _apply_categorical_drift(base, month, spec, rng)
    return pd.DataFrame(data)


def _generate_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a binary target based on a logistic model over selected features.
    This function is kept constant across months; any change in label distribution
    is driven by feature drift.
    """
    # Select a subset of features for the signal.
    age = df.get("age", 0)
    income = np.log1p(df.get("income", 0))
    score = df.get("score", 0)
    transactions = df.get("transactions_last_month", 0)

    # Encode categoricals into simple effects.
    region = df.get("region", "north")
    channel = df.get("channel", "web")

    region_effect = np.where(region == "north", 0.1, 0.0)
    region_effect += np.where(region == "south", 0.05, 0.0)

    channel_effect = np.where(channel == "mobile", 0.15, 0.0)

    z = (
        -3.0
        + 0.02 * (age - 40.0)
        + 0.3 * (income - income.mean())
        + 0.5 * score
        + 0.1 * transactions
        + region_effect
        + channel_effect
    )
    probs = 1.0 / (1.0 + np.exp(-z))
    return rng.binomial(1, probs)


def generate_month_data(month: int, config: DataGenerationConfig) -> pd.DataFrame:
    """
    Generate a single month of synthetic data including features and target label.
    """
    base_seed = config.random_seed + month * 1000
    rng = np.random.default_rng(base_seed)
    features = _generate_features_for_month(month, config, rng)
    labels = _generate_labels(features, rng)
    df = features.copy()
    df["label"] = labels
    df["month"] = month
    logger.info("Generated data for month %d with %d rows", month, len(df))
    return df


def generate_all_months(config: DataGenerationConfig | None = None) -> Dict[int, pd.DataFrame]:
    """
    Generate data for all configured months and return as a mapping.
    """
    if config is None:
        project_cfg = get_default_config()
        config = project_cfg.data_generation
    month_to_df: Dict[int, pd.DataFrame] = {}
    for month in config.months:
        month_to_df[month] = generate_month_data(month, config)
    logger.info("Generated data for months: %s", list(month_to_df.keys()))
    return month_to_df

