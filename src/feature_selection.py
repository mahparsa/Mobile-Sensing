"""Feature selection: try several methods and CV sizes, keep the best by macro-F1."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE, RFECV, SelectFromModel, SelectKBest, SelectPercentile,
    VarianceThreshold, mutual_info_classif,
)
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config


def build_selector(method, cv, scorer, seed):
    """Return an unfitted selector for the given method name."""
    rf = RandomForestClassifier(random_state=seed)
    if method == "RFECV":
        return RFECV(estimator=rf, step=1, cv=cv, scoring=scorer, n_jobs=-1)
    if method == "SelectFromModel":
        return SelectFromModel(estimator=rf)
    if method in ("MutualInfo", "SelectKBest", "MutualInfo_classif"):
        return SelectKBest(mutual_info_classif, k="all")
    if method == "RFE":
        return RFE(estimator=rf, n_features_to_select=config.FS_RFE_N_FEATURES)
    if method == "SelectPercentile":
        return SelectPercentile(mutual_info_classif, percentile=config.FS_PERCENTILE)
    if method == "VarianceThreshold":
        return VarianceThreshold()
    raise ValueError(f"Unknown feature selection method: {method}")


def select_features(X, y, verbose=config.VERBOSE):
    """Return (best_method, best_row, all_results_df).

    best_row contains: score, cv, n_features, features, rf_params.
    """
    scorer = make_scorer(f1_score, average="macro")
    rows = {}

    for method in config.FS_METHODS:
        best = {"score": -np.inf}
        for k in config.FS_CV_RANGE:
            cv = StratifiedKFold(n_splits=k)
            selector = build_selector(method, cv, scorer, config.SEED)
            selector.fit(X, y)
            X_sel = selector.transform(X)

            grid = GridSearchCV(
                RandomForestClassifier(random_state=config.SEED),
                config.FS_RF_PARAM_GRID, cv=cv, scoring=scorer, n_jobs=-1,
            )
            grid.fit(X_sel, y)

            if grid.best_score_ > best["score"]:
                best = {
                    "score": grid.best_score_,
                    "cv": k,
                    "n_features": X_sel.shape[1],
                    "features": list(X.columns[selector.get_support()]),
                    "rf_params": grid.best_params_,
                }
        rows[method] = best

    results = pd.DataFrame(rows).T
    results["score"] = results["score"].astype(float)
    results = results.sort_values("score", ascending=False)

    best_method = results.index[0]
    best_row = results.loc[best_method]
    if verbose:
        print(f"  Best FS method: {best_method} | CV={best_row['cv']} | "
              f"{best_row['n_features']} features | macro-F1={best_row['score']:.4f}")
        print(f"  Features: {best_row['features']}")
    return best_method, best_row, results
