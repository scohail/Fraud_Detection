"""Registry of the classifiers compared in the study."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data import RANDOM_STATE

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM": lambda: SVC(kernel="rbf", random_state=RANDOM_STATE),
}

# A kernel SVM scales roughly quadratically with the number of training rows:
# it is only trained when the resampled training set is small (under-sampling).
MAX_TRAIN_ROWS = {"SVM": 20_000}


def get_model(name, class_weight=None):
    """Return a fresh, unfitted classifier.

    ``class_weight='balanced'`` is applied when the estimator supports it;
    a ``ValueError`` is raised otherwise (e.g. KNN).
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Available: {list(MODELS)}")
    model = MODELS[name]()
    if class_weight is not None:
        if "class_weight" not in model.get_params():
            raise ValueError(f"{name} does not support class weights")
        model.set_params(class_weight=class_weight)
    return model
