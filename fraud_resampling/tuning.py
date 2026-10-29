"""Hyper-parameter tuning of a full resampling pipeline."""

from sklearn.model_selection import GridSearchCV

from .evaluation import SCORING, build_pipeline, make_cv


def tune(sampler_name, model_name, param_grid, X, y, n_splits=5, refit="pr_auc", n_jobs=1):
    """Grid search over a ``scaler -> sampler -> model`` pipeline.

    Parameter names follow the pipeline step names, e.g.
    ``{"model__max_depth": [5, 10], "sampler__k_neighbors": [3, 5]}``.

    The search is scored on PR-AUC by default: with 0.17 % frauds, accuracy would
    pick a model that predicts "legitimate" for everything.
    """
    search = GridSearchCV(
        build_pipeline(sampler_name, model_name),
        param_grid=param_grid,
        scoring=SCORING,
        refit=refit,
        cv=make_cv(n_splits),
        n_jobs=n_jobs,
    )
    return search.fit(X, y)
