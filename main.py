"""Run the full pipeline in the same order as the notebook."""
import config
from src.data_preparation import load_raw, prepare_dataset
from src.models import get_classifiers
from src.nested_cv import run_nested_cv
from src.evaluation import print_summary


def main():
    # Data preparation
    df_week1 = load_raw(config.RAW_DATA_PATH)
    X, y = prepare_dataset(df_week1)

    # Classifiers
    classifiers, param_grids = get_classifiers(config.MODEL_NAME)

    # Nested CV
    outer_results, best_models_per_fold, inner_best_models = run_nested_cv(X, y, classifiers, param_grids)

    # Summary
    print_summary(outer_results)


if __name__ == "__main__":
    main()
