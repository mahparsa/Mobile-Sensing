"""BLOCK 2: Classifiers (one is used per run, chosen by config.MODEL_NAME)."""
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def get_classifiers(model_name):
    if model_name == "catboost":
        # CatBoostClassifier
        classifiers = [
            CatBoostClassifier(verbose=0),
        ]
        param_grids = [
            {
                'iterations': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                'learning_rate': [0.01, 0.05, 0.1],
                'depth': [1, 2, 3, 4, 5],
            },
        ]

    elif model_name == "xgboost":
        # XGBClassifier
        classifiers = [
            XGBClassifier(),
        ]
        param_grids = [
            {
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [1, 2, 3, 4, 5],
            },
        ]

    elif model_name == "random_forest":
        # RandomForestClassifier
        classifiers = [
            RandomForestClassifier(),
        ]
        param_grids = [
            {
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                'max_depth': [1, 2, 3, 4, 5],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2'],
            },
        ]

    elif model_name == "gradient_boosting":
        # GradientBoostingClassifier
        classifiers = [
            GradientBoostingClassifier(),
        ]
        param_grids = [
            {
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [1, 2, 3, 4, 5],
            },
        ]

    elif model_name == "extra_trees":
        # ExtraTreesClassifier
        classifiers = [
            ExtraTreesClassifier(),
        ]
        param_grids = [
            {
                'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                'max_depth': [1, 2, 3, 4, 5],
                'min_samples_split': [2, 3, 4],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2'],
            },
        ]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return classifiers, param_grids
