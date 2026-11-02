"""Configuration for the nested CV pipeline (data preparation outside the CV loop)."""

# ---------- Raw data ----------
RAW_DATA_PATH = "Data/week1.csv"   # the file you load as df_week1
ID_COLUMN = "participantid"
PHONE_COLUMN = "phone"
PLATFORM_COLUMN = "platform"
PLATFORMS = {"iOS": 1, "Android": 0}   # phone value -> platform code

# ---------- Label ----------
LABEL_COLUMN = "disorder_name___10"
POSITIVE_LABEL = "Anxiety disorder"    # encoded as 1, everything else as 0
TARGET_COLUMN = "target"

# ---------- Numeric features (scaled per platform inside each fold) ----------
MOBILE_SENSING_FEATURES = [
    "total_haversine", "number_of_stay_points", "location_entropy", "normalized_entropy",
    "total_call_duration", "total_incoming_calls", "total_outgoing_calls", "total_screen_time",
    "total_unlocks", "total_evening_screen_time", "total_evening_unlocks", "total_daily_screen_time",
    "total_daily_unlocks", "total_nightly_screen_time", "total_nightly_unlocks",
    "total_screen_estimated_sleep_time_in_secs", "activity_entropy", "normalized_activity_entropy",
    "percent_time_spent_stationary", "percent_time_spent_stationary_evening",
    "percent_time_spent_stationary_night",
]
EMA_TIME_FEATURES = [
    "ema_3b_time_vigorous_activity", "ema_4b_time_moderate_activity", "ema_5b_time_walk",
]
AGE_FEATURES = ["age"]
SCALE_COLUMNS = MOBILE_SENSING_FEATURES + EMA_TIME_FEATURES + AGE_FEATURES

# ---------- Categorical EMA features (one-hot encoded) ----------
# column -> all possible answers (so every fold has the same columns)
EMA_CATEGORIES = {
    "ema_2_hangout_with_friends": [0, 1, 2, 3, 4, 5],
    "ema_3a_days_vigorous_activity": [0, 1, 2, 3, 4, 5, 6],
    "ema_4a_days_moderate_activity": [0, 1, 2, 3, 4, 5],
    "ema_5a_days_walk": [0, 1, 2, 3, 4, 5, 6, 7],
    "ema_8_sleep_disruption": [0, 1, 2, 3, 4, 5],
}

# ---------- Demographics (mapped to numbers; unknown -> -1) ----------
DEMOGRAPHIC_MAPPINGS = {
    "sex": {"Female": 1, "Male": 0},
    "education": {
        "Finished university": 4,
        "Further education": 3,
        "Finished high school": 2,
        "Student at university": 1,
        "Currently in high school": 0,
    },
    "sexual_orientation": {"Straight": 1, "Non-Straight": 0},
    "ethnicity": {"White": 1, "Non-White": 0},
}

# ---------- General ----------
SEED = 42
OUTER_FOLDS = 10
MODEL_NAME = "catboost"            # catboost | xgboost | random_forest | gradient_boosting | extra_trees
RESULTS_DIR = "results"
VERBOSE = True

# ---------- Outlier removal (LOF) ----------
LOF_NEIGHBORS = [5, 10, 15, 20, 25, 30]
LOF_CONTAMINATION = 0.05

# ---------- Oversampling (SVMSMOTE) ----------
SMOTE_PARAMS = {
    "sampling_strategy": "auto",
    "k_neighbors": 2,
    "m_neighbors": 4,
    "out_step": 0.5,
}

# ---------- Feature selection ----------
# Note: "MutualInfo", "SelectKBest" and "MutualInfo_classif" are identical.
FS_METHODS = [
    "RFECV", "SelectFromModel", "MutualInfo", "RFE",
    "VarianceThreshold", "SelectKBest", "SelectPercentile", "MutualInfo_classif",
]
FS_CV_RANGE = range(2, 11)
FS_RFE_N_FEATURES = 5
FS_PERCENTILE = 50
FS_RF_PARAM_GRID = {
    "n_estimators": [10, 20, 30, 40, 50],
    "max_depth": [2, 3, 4, 5],
    "min_samples_split": [2, 3, 4, 5, 6],
    "min_samples_leaf": [2, 3, 4],
    "max_features": ["sqrt", "log2"],
}
