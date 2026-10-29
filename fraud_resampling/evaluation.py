"""Leakage-free evaluation of ``scaler -> sampler -> classifier`` combinations.

The golden rule of resampling: **only the training data may be resampled.**
Validation and test folds must keep the real class distribution, otherwise the
model is scored on synthetic or duplicated frauds it has (almost) already seen.
Everything here applies the sampler inside each training fold only.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

from .data import RANDOM_STATE
from .models import MAX_TRAIN_ROWS, get_model
from .samplers import SAMPLERS, get_sampler

# Metrics reported everywhere. Accuracy is kept only to show how misleading it is.
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "mcc": "matthews_corrcoef",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}
METRICS = list(SCORING)


def make_cv(n_splits=5):
    """Stratified K-fold: every fold keeps the ~0.17 % fraud rate."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def scores_of(model, X):
    """Continuous fraud score (probability when available, else SVM decision value)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    """All evaluation metrics for one set of predictions."""
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def build_pipeline(sampler_name, model_name, **model_params):
    """``StandardScaler -> sampler -> model`` as an imbalanced-learn Pipeline.

    An imblearn ``Pipeline`` calls ``fit_resample`` during ``fit`` only; at
    prediction time the sampler is skipped, so validation data is never resampled.
    Scaling comes first because SMOTE, NearMiss, ENN, ... rely on distances and
    ``Time``/``Amount`` are not on the same scale as the PCA features ``V1..V28``.
    """
    class_weight = "balanced" if sampler_name == "Class weights" else None
    model = get_model(model_name, class_weight=class_weight)
    model.set_params(**model_params)
    steps = [("scaler", StandardScaler())]
    sampler = get_sampler(sampler_name)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return Pipeline(steps)


def _fold_table(raw):
    """Turn sklearn's ``cross_validate`` output into a per-fold table plus mean/std rows."""
    df = pd.DataFrame(raw)
    df.columns = [c.replace("test_", "") for c in df.columns]
    df.index = [f"fold {i + 1}" for i in range(len(df))]
    summary = df.agg(["mean", "std"])
    return pd.concat([df, summary])


def cross_validate_pipeline(sampler_name, model_name, X, y, n_splits=5, n_jobs=None):
    """Correct way: resampling happens inside each training fold."""
    pipe = build_pipeline(sampler_name, model_name)
    raw = cross_validate(pipe, X, y, cv=make_cv(n_splits), scoring=SCORING, n_jobs=n_jobs)
    return _fold_table(raw)


def leaky_cross_validate(sampler_name, model_name, X, y, n_splits=5, n_jobs=None):
    """Wrong way, kept for demonstration: resample everything, *then* cross-validate.

    The validation folds then contain synthetic/duplicated frauds that are near-copies
    of training points, and their class ratio is 50/50 instead of 0.17 %, so precision
    and every other score look far better than what the model achieves on real data.
    """
    X_scaled = StandardScaler().fit_transform(X)
    X_res, y_res = get_sampler(sampler_name).fit_resample(X_scaled, y)
    model = get_model(model_name)
    raw = cross_validate(model, X_res, y_res, cv=make_cv(n_splits), scoring=SCORING, n_jobs=n_jobs)
    return _fold_table(raw)


def benchmark(X, y, samplers=None, models=None, n_splits=5, checkpoint=None, verbose=True):
    """Evaluate every (sampler, model) pair with the same stratified folds.

    Equivalent to running ``cross_validate_pipeline`` for every pair, but each fold
    is resampled only once and shared by all models, which divides the cost of
    the slow samplers (SVMSMOTE, SMOTETomek, ...) by the number of models.

    Parameters
    ----------
    checkpoint : str or Path, optional
        CSV file where rows are appended as they are computed. Pairs already in
        the file are skipped, so an interrupted run can be resumed.

    Returns
    -------
    DataFrame with one row per (sampler, model, fold).
    """
    samplers = samplers or list(SAMPLERS)
    models = models or ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]
    X, y = np.asarray(X), np.asarray(y)

    done = set()
    rows = []
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if checkpoint.exists():
            previous = load_results(checkpoint)
            rows = previous.to_dict("records")
            done = set(zip(previous["sampler"], previous["model"], previous["fold"]))

    def log(msg):
        if verbose:
            print(msg, flush=True)

    for fold, (train_idx, val_idx) in enumerate(make_cv(n_splits).split(X, y), start=1):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_val = scaler.transform(X[train_idx]), scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        for sampler_name in samplers:
            todo = [m for m in models if (sampler_name, m, fold) not in done]
            if not todo:
                continue

            start = time.perf_counter()
            sampler = get_sampler(sampler_name)
            if sampler is None:
                X_res, y_res = X_train, y_train
            else:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
            resample_time = time.perf_counter() - start
            log(f"fold {fold} | {sampler_name:<18} {len(y_res):>7} rows, "
                f"{int(y_res.sum()):>6} frauds ({resample_time:.1f}s)")

            for model_name in todo:
                if len(y_res) > MAX_TRAIN_ROWS.get(model_name, np.inf):
                    continue  # too large for this model (kernel SVM)
                if sampler_name == "Class weights":
                    try:
                        model = get_model(model_name, class_weight="balanced")
                    except ValueError:
                        continue  # KNN has no class weights
                else:
                    model = get_model(model_name)

                start = time.perf_counter()
                model.fit(X_res, y_res)
                fit_time = time.perf_counter() - start
                metrics = compute_metrics(y_val, model.predict(X_val), scores_of(model, X_val))
                row = {
                    "sampler": sampler_name,
                    "family": SAMPLERS[sampler_name].family,
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(y_res),
                    "n_train_fraud": int(y_res.sum()),
                    "resample_time": resample_time,
                    "fit_time": fit_time,
                    **metrics,
                }
                rows.append(row)
                if checkpoint is not None:
                    pd.DataFrame([row]).to_csv(
                        checkpoint, mode="a", header=not checkpoint.exists(), index=False
                    )
                log(f"    {model_name:<20} PR-AUC={metrics['pr_auc']:.3f} "
                    f"recall={metrics['recall']:.3f} precision={metrics['precision']:.3f}")

    return pd.DataFrame(rows)


def load_results(path):
    """Read a benchmark CSV. ``keep_default_na=False`` stops pandas from turning the
    technique name "None" into NaN."""
    return pd.read_csv(path, keep_default_na=False)


def summarize(results, metrics=METRICS):
    """Mean and standard deviation over folds for every (sampler, model) pair."""
    grouped = results.groupby(["family", "sampler", "model"], sort=False)
    mean = grouped[metrics + ["n_train", "fit_time"]].mean()
    std = grouped[metrics].std().add_suffix("_std")
    return mean.join(std).reset_index()
