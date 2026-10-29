"""Registry of the resampling techniques studied in the project.

Every technique is described by a :class:`Technique` holding its family
(over / under / hybrid), a factory that builds a fresh imbalanced-learn object,
and a one-line explanation used in the notebooks and the README.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
    SVMSMOTE,
)
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)

from .data import RANDOM_STATE, TARGET


@dataclass(frozen=True)
class Technique:
    name: str
    family: str  # "none" | "cost-sensitive" | "over" | "under" | "hybrid"
    factory: Optional[Callable[[], object]]
    summary: str


_TECHNIQUES = [
    Technique(
        "None", "none", None,
        "Baseline: the model is trained on the original, imbalanced data.",
    ),
    Technique(
        "Class weights", "cost-sensitive", None,
        "No resampling; the loss of each class is weighted by its inverse frequency "
        "(class_weight='balanced').",
    ),
    # --- Over-sampling: create new minority (fraud) samples -------------------
    Technique(
        "RandomOverSampler", "over",
        lambda: RandomOverSampler(random_state=RANDOM_STATE),
        "Duplicates randomly chosen fraud transactions until both classes have the same size.",
    ),
    Technique(
        "SMOTE", "over",
        lambda: SMOTE(random_state=RANDOM_STATE),
        "Creates synthetic frauds by interpolating between a fraud and one of its k=5 "
        "nearest fraud neighbours.",
    ),
    Technique(
        "BorderlineSMOTE", "over",
        lambda: BorderlineSMOTE(random_state=RANDOM_STATE),
        "SMOTE restricted to 'danger' frauds, i.e. those whose neighbourhood is mostly "
        "legitimate (the decision border).",
    ),
    Technique(
        "SVMSMOTE", "over",
        lambda: SVMSMOTE(random_state=RANDOM_STATE),
        "Uses the support vectors of an SVM to locate the border, then applies SMOTE there.",
    ),
    Technique(
        "KMeansSMOTE", "over",
        # The default threshold finds no cluster with enough frauds on this data.
        lambda: KMeansSMOTE(random_state=RANDOM_STATE, cluster_balance_threshold=0.001),
        "Clusters the data with k-means and applies SMOTE inside clusters that contain frauds.",
    ),
    Technique(
        "ADASYN", "over",
        lambda: ADASYN(random_state=RANDOM_STATE),
        "Like SMOTE, but generates more samples for frauds that are harder to learn "
        "(surrounded by legitimate transactions).",
    ),
    # --- Under-sampling: remove majority (legitimate) samples -----------------
    Technique(
        "RandomUnderSampler", "under",
        lambda: RandomUnderSampler(random_state=RANDOM_STATE),
        "Randomly discards legitimate transactions until both classes have the same size.",
    ),
    Technique(
        "NearMiss-1", "under",
        lambda: NearMiss(version=1, n_neighbors=3),
        "Keeps the legitimate transactions whose mean distance to their 3 closest frauds "
        "is the smallest.",
    ),
    Technique(
        "NearMiss-2", "under",
        lambda: NearMiss(version=2, n_neighbors=3),
        "Keeps the legitimate transactions whose mean distance to their 3 farthest frauds "
        "is the smallest.",
    ),
    Technique(
        "NearMiss-3", "under",
        lambda: NearMiss(version=3, n_neighbors=3),
        "For every fraud, keeps its closest legitimate neighbours (two-step selection).",
    ),
    Technique(
        "TomekLinks", "under",
        lambda: TomekLinks(n_jobs=-1),
        "Removes legitimate transactions that form a Tomek link (mutual nearest neighbours "
        "of opposite classes) - a cleaning method, it barely changes the ratio.",
    ),
    Technique(
        "ENN", "under",
        lambda: EditedNearestNeighbours(n_jobs=-1),
        "Edited Nearest Neighbours: removes legitimate transactions misclassified by their "
        "3 nearest neighbours - also a cleaning method.",
    ),
    # --- Hybrid: over-sample, then clean ---------------------------------------
    Technique(
        "SMOTETomek", "hybrid",
        lambda: SMOTETomek(random_state=RANDOM_STATE, n_jobs=-1),
        "SMOTE followed by Tomek-link removal to clean the overlap created between classes.",
    ),
    Technique(
        "SMOTEENN", "hybrid",
        lambda: SMOTEENN(random_state=RANDOM_STATE, n_jobs=-1),
        "SMOTE followed by Edited Nearest Neighbours, a more aggressive cleaning than Tomek.",
    ),
]

SAMPLERS = {t.name: t for t in _TECHNIQUES}
FAMILY_ORDER = ["none", "cost-sensitive", "over", "under", "hybrid"]


def get_sampler(name):
    """Return a fresh imbalanced-learn sampler, or ``None`` for the no-resampling options."""
    if name not in SAMPLERS:
        raise ValueError(f"Unknown technique {name!r}. Available: {list(SAMPLERS)}")
    factory = SAMPLERS[name].factory
    return factory() if factory else None


def resample(df, name, verbose=True):
    """Resample a whole dataframe with technique ``name`` and return a new dataframe.

    Useful for visualisation and exploration. **Do not** resample a dataset and then
    cross-validate on it - see ``evaluation.leaky_cross_validate`` for why.
    """
    sampler = get_sampler(name)
    X, y = df.drop(columns=TARGET), df[TARGET]
    if sampler is None:
        X_res, y_res = X, y
    else:
        X_res, y_res = sampler.fit_resample(X, y)
    if verbose:
        print(f"{name}: {dict(Counter(y))} -> {dict(Counter(y_res))}")
    return pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=TARGET)], axis=1)
