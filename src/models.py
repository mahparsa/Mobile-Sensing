"""Classifiers and their hyperparameter grids. Pick one via config.MODEL_NAME."""
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from xgboost import XGBClassifier

N_ESTIMATORS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
DEPTHS = [1, 2, 3, 4, 5]
LEARNING_RATES = [0.01, 0.05, 0.1]

TREE_GRID = {
    "n_estimators": N_ESTIMATORS,
    "max_depth": DEPTHS,
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2"],
}
BOOSTING_GRID = {
    "n_estimators": N_ESTIMATORS,
    "learning_rate": LEARNING_RATES,
    "max_depth": DEPTHS,
}

MODELS = {
    "catboost": (
        lambda seed: CatBoostClassifier(verbose=0, random_seed=seed),
        {"iterations": N_ESTIMATORS, "learning_rate": LEARNING_RATES, "depth": DEPTHS},
    ),
    "xgboost": (lambda seed: XGBClassifier(random_state=seed), BOOSTING_GRID),
    "gradient_boosting": (lambda seed: GradientBoostingClassifier(random_state=seed), BOOSTING_GRID),
    "random_forest": (lambda seed: RandomForestClassifier(random_state=seed), TREE_GRID),
    "extra_trees": (lambda seed: ExtraTreesClassifier(random_state=seed), TREE_GRID),
}


def get_model(name, seed):
    """Return (unfitted estimator, param_grid) for a model name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODELS)}")
    factory, grid = MODELS[name]
    return factory(seed), grid
