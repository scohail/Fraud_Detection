"""Toolkit for studying data-resampling techniques on credit-card fraud detection.

Modules
-------
data        Loading the Kaggle ``creditcard.csv`` dataset and building the hold-out split.
samplers    Registry of over-, under- and hybrid resampling techniques (imbalanced-learn).
models      Registry of the classifiers compared in the study.
evaluation  Leakage-free cross-validation, the benchmark loop and imbalance-aware metrics.
tuning      Hyper-parameter search over a full ``scaler -> sampler -> model`` pipeline.
plots       Plotting helpers shared by the notebooks.
"""

from .data import load_data, train_test_split_stratified
from .samplers import SAMPLERS, get_sampler, resample
from .models import MODELS, get_model
from .evaluation import (
    build_pipeline,
    cross_validate_pipeline,
    leaky_cross_validate,
    benchmark,
    load_results,
    summarize,
    compute_metrics,
)
from .tuning import tune

__all__ = [
    "load_data",
    "train_test_split_stratified",
    "SAMPLERS",
    "get_sampler",
    "resample",
    "MODELS",
    "get_model",
    "build_pipeline",
    "cross_validate_pipeline",
    "leaky_cross_validate",
    "benchmark",
    "load_results",
    "summarize",
    "compute_metrics",
    "tune",
]
