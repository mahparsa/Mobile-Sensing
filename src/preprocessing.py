"""BLOCK 1: Preprocessing inside each fold (code from the notebook, split into functions)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SVMSMOTE

import config
from src.feature_selection import select_features

columns_to_normalize = config.SCALE_COLUMNS


def scale_per_platform(X_train, X_test):
    # RobustScaler per platform (fitted on training fold only)
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns_to_normalize] = X_train[columns_to_normalize].astype(float)
    X_test[columns_to_normalize] = X_test[columns_to_normalize].astype(float)
    for p in [1, 0]:
        tr = X_train['platform'] == p
        te = X_test['platform'] == p
        scaler = RobustScaler()
        X_train.loc[tr, columns_to_normalize] = scaler.fit_transform(X_train.loc[tr, columns_to_normalize])
        if te.any():
            X_test.loc[te, columns_to_normalize] = scaler.transform(X_test.loc[te, columns_to_normalize])
    X_train = X_train.drop(columns='platform')
    X_test = X_test.drop(columns='platform')
    return X_train, X_test


def remove_outliers(X_train, y_train):
    # LOF
    X = X_train
    y = y_train
    outliers_sets = []
    neighbors_list = [5, 10, 15, 20, 25, 30]
    for n_neighbors in neighbors_list:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
        preds = lof.fit_predict(X)
        outliers_idx = np.where(preds == -1)[0]
        outliers_sets.append(set(outliers_idx))
        print(f"n_neighbors={n_neighbors}: Detected {len(outliers_idx)} outliers")
    consistent_outliers = set.intersection(*outliers_sets)
    print(f"Consistently identified outliers across all runs: {len(consistent_outliers)}")
    mask = np.ones(len(X), dtype=bool)
    mask[list(consistent_outliers)] = False
    X_clean = X[mask]
    y_clean = y[mask]
    X_final_cleaned = X_clean
    y_final_cleaned = y_clean
    return X_final_cleaned, y_final_cleaned


def oversample(X_final_cleaned, y_final_cleaned):
    # SVMSMOTE
    X = X_final_cleaned
    y = y_final_cleaned
    oversample = SVMSMOTE( sampling_strategy='auto',  random_state=None, k_neighbors=2, m_neighbors=4, svm_estimator=None, out_step=0.5)
    X_resampled, y_resampled= oversample.fit_resample(X, y)
    class_counts = pd.Series(y_resampled).value_counts()
    print(class_counts)
    X = X_resampled
    y = y_resampled
    return X, y


def preprocess_fold(X_train, X_test, y_train):
    X_train, X_test = scale_per_platform(X_train, X_test)
    X, y = remove_outliers(X_train, y_train)
    X, y = oversample(X, y)
    X_new, X_test_new, best_cv_value = select_features(X, y, X_test)
    return X_new, y, X_test_new, int(best_cv_value)
