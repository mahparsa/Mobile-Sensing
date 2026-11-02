"""Nested cross-validation: outer loop for evaluation, inner loop for tuning."""
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
from src.evaluation import compute_metrics, format_metrics
from src.feature_selection import select_features
from src.models import get_model
from src.preprocessing import preprocess_fold


def run_nested_cv(X, y, model_name=config.MODEL_NAME):
    """Returns (results_df, fold_models, roc_data)."""
    outer_cv = StratifiedKFold(n_splits=config.OUTER_FOLDS, shuffle=True, random_state=config.SEED)

    results, fold_models, roc_data = [], {}, []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"\n========= Fold {fold} =========")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 1. Preprocess (fit on train only)
        X_train, y_train, X_test = preprocess_fold(X_train, y_train, X_test)

        # 2. Feature selection (train only)
        fs_method, fs_info, _ = select_features(X_train, y_train)
        features = fs_info["features"]
        X_train, X_test = X_train[features], X_test[features]

        # 3. Inner CV hyperparameter tuning
        estimator, param_grid = get_model(model_name, config.SEED)
        inner_cv = StratifiedKFold(n_splits=int(fs_info["cv"]), shuffle=True, random_state=config.SEED)
        grid = GridSearchCV(
            estimator, param_grid, cv=inner_cv,
            scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
            refit="accuracy",
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  Best params ({model_name}): {grid.best_params_}")

        # 4. Evaluate on the untouched outer test fold
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print(f"  {format_metrics(metrics)}")

        results.append({"fold": fold, "model_name": model_name, "fs_method": fs_method, **metrics})
        fold_models[fold] = {"model": model, "params": grid.best_params_, "features": features}
        roc_data.append({"fold": fold, "model_name": model_name,
                         "y_true": y_test, "y_prob": y_prob, "auc": metrics["auc"]})

    return pd.DataFrame(results), fold_models, roc_data
