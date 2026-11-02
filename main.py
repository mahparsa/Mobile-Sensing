"""Run the full nested-CV pipeline.

Usage:
    python main.py
    python main.py --model xgboost --data data/other.csv
"""
import argparse
import os

import config
from src.data import load_data
from src.nested_cv import run_nested_cv
from src.evaluation import print_summary, plot_roc


def parse_args():
    parser = argparse.ArgumentParser(description="Nested CV pipeline")
    parser.add_argument("--data", default=config.DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--model", default=config.MODEL_NAME, help="Model name (see src/models.py)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    X, y = load_data(args.data)
    results_df, fold_models, roc_data = run_nested_cv(X, y, model_name=args.model)

    print_summary(results_df)
    results_df.to_csv(os.path.join(config.RESULTS_DIR, f"results_{args.model}.csv"), index=False)
    plot_roc(roc_data, save_path=os.path.join(config.RESULTS_DIR, f"roc_{args.model}.png"))


if __name__ == "__main__":
    main()
