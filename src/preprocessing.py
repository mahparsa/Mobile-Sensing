"""Per-fold preprocessing: scaling, outlier removal, oversampling.

Everything here is fitted on the training fold only, to avoid data leakage.
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

import config


def scale_per_platform(X_train, X_test, columns, platform_col):
    """Fit a RobustScaler per platform on train, apply to test. Drops platform column."""
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[columns] = X_train[columns].astype(float)
    X_test[columns] = X_test[columns].astype(float)

    for platform in X_train[platform_col].unique():
        train_mask = X_train[platform_col] == platform
        test_mask = X_test[platform_col] == platform

        scaler = RobustScaler()
        X_train.loc[train_mask, columns] = scaler.fit_transform(X_train.loc[train_mask, columns])
        if test_mask.any():
            X_test.loc[test_mask, columns] = scaler.transform(X_test.loc[test_mask, columns])

    return X_train.drop(columns=platform_col), X_test.drop(columns=platform_col)


def remove_consistent_outliers(X, y, neighbors_list, contamination, verbose=True):
    """Remove samples flagged as outliers by LOF for *every* n_neighbors value."""
    outlier_sets = []
    for n in neighbors_list:
        lof = LocalOutlierFactor(n_neighbors=n, contamination=contamination)
        preds = lof.fit_predict(X)
        idx = set(np.where(preds == -1)[0])
        outlier_sets.append(idx)
        if verbose:
            print(f"  LOF n_neighbors={n}: {len(idx)} outliers")

    consistent = set.intersection(*outlier_sets)
    if verbose:
        print(f"  Consistent outliers removed: {len(consistent)}")

    mask = np.ones(len(X), dtype=bool)
    mask[list(consistent)] = False
    return X[mask], y[mask]


def oversample(X, y, seed, params):
    """Balance classes with SVMSMOTE."""
    smote = SVMSMOTE(random_state=seed, **params)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def preprocess_fold(X_train, y_train, X_test, verbose=config.VERBOSE):
    """Full preprocessing for one outer fold. Returns (X_train, y_train, X_test)."""
    feature_cols = [c for c in X_train.columns if c != config.PLATFORM_COLUMN]

    X_train, X_test = scale_per_platform(X_train, X_test, feature_cols, config.PLATFORM_COLUMN)
    X_train, y_train = remove_consistent_outliers(
        X_train, y_train, config.LOF_NEIGHBORS, config.LOF_CONTAMINATION, verbose
    )
    X_train, y_train = oversample(X_train, y_train, config.SEED, config.SMOTE_PARAMS)

    if verbose:
        print("  Class counts after SVMSMOTE:", pd.Series(y_train).value_counts().to_dict())
    return X_train, y_train, X_test
