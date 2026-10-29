"""Run the full resampling x classifier benchmark and save it to results/cv_benchmark.csv.

Usage (from the project root):
    python scripts/run_benchmark.py            # resumes if the CSV already exists
    python scripts/run_benchmark.py --fresh    # start from scratch
    python scripts/run_benchmark.py --folds 3 --models "Logistic Regression" "Random Forest"

Only the training part of the hold-out split is used; the test set stays untouched
for the final evaluation in notebooks/04_tuning_and_final_evaluation.ipynb.
Expect roughly one hour on a 12-core laptop for the full grid.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fraud_resampling import benchmark, load_data, train_test_split_stratified  # noqa: E402

OUTPUT = ROOT / "results" / "cv_benchmark.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--samplers", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fresh", action="store_true", help="delete previous results first")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and args.output.exists():
        args.output.unlink()

    X_train, _, y_train, _ = train_test_split_stratified(load_data())
    benchmark(X_train, y_train, samplers=args.samplers, models=args.models,
              n_splits=args.folds, checkpoint=args.output)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
