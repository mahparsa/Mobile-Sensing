"""Feature selection (code from the notebook, BLOCK 1)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectKBest, SelectPercentile, SelectFromModel, VarianceThreshold
from sklearn.metrics import make_scorer, f1_score
from sklearn.feature_selection import mutual_info_classif


def select_features(X, y, X_test):
    # Feature selection
    scorer = make_scorer(f1_score, average='macro')
    Features_Name = X.columns
    feature_selection_methods = [
        "RFECV", "SelectFromModel", "MutualInfo", "RFE",
        "VarianceThreshold", "SelectKBest", "SelectPercentile", "MutualInfo_classif"
    ]
    results = {}
    rf_param_grid = {
        'n_estimators': [10, 20, 30, 40, 50],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf': [2, 3, 4],
        'max_features': ['sqrt', 'log2'],
    }
    for method in feature_selection_methods:
        best_score = -np.inf
        best_cv_value = None
        best_n_features = None
        best_features = None
        best_feature_names = None
        for cv_value in range(2, 11):
            cv = StratifiedKFold(n_splits=cv_value)
            if method == "RFECV":
                selector = RFECV(estimator=RandomForestClassifier(random_state=42), step=1, cv=cv, scoring=scorer, n_jobs=-1)
            elif method == "SelectFromModel":
                selector = SelectFromModel(estimator=RandomForestClassifier(random_state=42))
            elif method in ["MutualInfo", "SelectKBest", "MutualInfo_classif"]:
                selector = SelectKBest(mutual_info_classif, k='all')
            elif method == "RFE":
                selector = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
            elif method == "SelectPercentile":
                selector = SelectPercentile(mutual_info_classif, percentile=50)
            elif method == "VarianceThreshold":
                selector = VarianceThreshold()
            selector.fit(X, y)
            X_selected = selector.transform(X) if hasattr(selector, 'transform') else X
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=cv, scoring=scorer, n_jobs=-1)
            grid_search.fit(X_selected, y)
            mean_score = grid_search.best_score_
            if mean_score > best_score:
                best_score = mean_score
                best_cv_value = cv_value
                best_n_features = X_selected.shape[1]
                best_features = selector.get_support() if hasattr(selector, 'get_support') else None
                best_feature_names = Features_Name[best_features] if best_features is not None else None
        results[method] = {
            'Best CV Value': best_cv_value,
            'Best Number of Features': best_n_features,
            'Best Score': best_score,
            'Best Features': best_feature_names,
            'Best RF Params': grid_search.best_params_
        }
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values(by="Best Score", ascending=False)

    best_method = df_results.index[0]
    best_cv_value = df_results.loc[best_method, "Best CV Value"]
    best_features = df_results.loc[best_method, "Best Features"]
    best_rf_params = df_results.loc[best_method, "Best RF Params"]

    X_new = X[best_features] if best_features is not None else X
    X_test_new = X_test[best_features] if best_features is not None else X_test

    print(f"\nBest Method: {best_method}")
    print(f"Best CV Value: {best_cv_value}")
    print(f"Selected Features:\n{best_features}")
    print(f"Best RandomForest Parameters:\n{best_rf_params}")

    return X_new, X_test_new, best_cv_value
