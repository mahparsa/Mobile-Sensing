"""Data preparation from outside.ipynb (code unchanged, wrapped in a function)."""
import pandas as pd

import config


def load_raw(path=config.RAW_DATA_PATH):
    """Load the raw week-1 data (df_week1)."""
    return pd.read_csv(path)


def prepare_dataset(df_week1):
    df = df_week1
    Label_column_name = 'disorder_name___10'
    label_column = df[Label_column_name]
    print(label_column)

    week1_data = df_week1
    week1_ios = week1_data[week1_data['phone'] == 'iOS']
    week1_android = week1_data[week1_data['phone'] == 'Android']

    df_ios = week1_ios
    df_android = week1_android

    ID = ['participantid']

    less_mobile_sensing_features=[
       'total_haversine', 'number_of_stay_points', 'location_entropy', 'normalized_entropy',
        'total_call_duration', 'total_incoming_calls', 'total_outgoing_calls', 'total_screen_time',
        'total_unlocks', 'total_evening_screen_time', 'total_evening_unlocks', 'total_daily_screen_time',
        'total_daily_unlocks', 'total_nightly_screen_time', 'total_nightly_unlocks',
        'total_screen_estimated_sleep_time_in_secs', 'activity_entropy', 'normalized_activity_entropy',
        'percent_time_spent_stationary', 'percent_time_spent_stationary_evening', 'percent_time_spent_stationary_night',
    ]
    ema_time_features= ['ema_3b_time_vigorous_activity', 'ema_4b_time_moderate_activity', 'ema_5b_time_walk']
    dm_features =['age']
    ms_features = less_mobile_sensing_features
    selected_columns_1 = ID+ms_features+ema_time_features+dm_features

    df = df_ios
    feature_names = [col for col in selected_columns_1 if col in df.columns]
    feature_columns_ms_ios = df_ios[feature_names]
    feature_columns_ms_android = df_android[feature_names]

    label_column_ios = df_ios[Label_column_name]
    label_column_android = df_android[Label_column_name]

    new_df_lessfeatures_ms_ios = pd.concat([feature_columns_ms_ios, label_column_ios], axis=1)
    new_df_lessfeatures_ms_android = pd.concat([feature_columns_ms_android, label_column_android], axis=1)

    week1_df_cleaned_ios = new_df_lessfeatures_ms_ios.dropna()
    week1_df_cleaned_android = new_df_lessfeatures_ms_android.dropna()

    df_ms_ios = week1_df_cleaned_ios
    df_ms_android = week1_df_cleaned_android

    df1_filtered_ios = df_ms_ios
    df1_filtered_android = df_ms_android

    participant_ids_ios = df1_filtered_ios['participantid'].tolist()
    participant_ids_android = df1_filtered_android['participantid'].tolist()

    ema = [
        'ema_2_hangout_with_friends',
        'ema_3a_days_vigorous_activity',
        'ema_4a_days_moderate_activity',
        'ema_5a_days_walk',
        'ema_8_sleep_disruption'
    ]
    dm_features =[ 'sex', 'education', 'sexual_orientation', 'ethnicity' ]

    selected_columns_2 = ['participantid'] + ema
    selected_columns_3 = ['participantid'] + dm_features

    df_filtered_ios_2 = df_ios[df_ios['participantid'].isin(participant_ids_ios)][selected_columns_2]
    df_filtered_android_2 = df_android[df_android['participantid'].isin(participant_ids_android)][selected_columns_2]

    df_filtered_ios_3 = df_ios[df_ios['participantid'].isin(participant_ids_ios)][selected_columns_3]
    df_filtered_android_3 = df_android[df_android['participantid'].isin(participant_ids_android)][selected_columns_3]

    df_filtered_ios_1 = df1_filtered_ios
    df_filtered_android_1 = df1_filtered_android

    columns_to_encode = [
        'ema_2_hangout_with_friends',
        'ema_3a_days_vigorous_activity',
        'ema_4a_days_moderate_activity',
        'ema_5a_days_walk',
        'ema_8_sleep_disruption'
    ]
    expected_features = [
        'ema_2_hangout_with_friends_0.0', 'ema_2_hangout_with_friends_1.0', 'ema_2_hangout_with_friends_2.0', 'ema_2_hangout_with_friends_3.0',
        'ema_2_hangout_with_friends_4.0', 'ema_2_hangout_with_friends_5.0', 'ema_3a_days_vigorous_activity_0.0',
        'ema_3a_days_vigorous_activity_1.0', 'ema_3a_days_vigorous_activity_2.0', 'ema_3a_days_vigorous_activity_3.0',
        'ema_3a_days_vigorous_activity_4.0', 'ema_3a_days_vigorous_activity_5.0', 'ema_3a_days_vigorous_activity_6.0',
        'ema_4a_days_moderate_activity_0.0', 'ema_4a_days_moderate_activity_1.0', 'ema_4a_days_moderate_activity_2.0',
        'ema_4a_days_moderate_activity_3.0', 'ema_4a_days_moderate_activity_4.0', 'ema_4a_days_moderate_activity_5.0',
        'ema_5a_days_walk_0.0', 'ema_5a_days_walk_1.0', 'ema_5a_days_walk_2.0', 'ema_5a_days_walk_3.0',
        'ema_5a_days_walk_4.0', 'ema_5a_days_walk_5.0', 'ema_5a_days_walk_6.0', 'ema_5a_days_walk_7.0',
        'ema_8_sleep_disruption_0.0', 'ema_8_sleep_disruption_1.0', 'ema_8_sleep_disruption_2.0',
        'ema_8_sleep_disruption_3.0', 'ema_8_sleep_disruption_4.0', 'ema_8_sleep_disruption_5.0'
    ]

    # One-hot EMA (iOS)
    df = df_filtered_ios_2
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode, dtype=int)
    df_encoded['participantid'] = df['participantid']
    df_encoded = df_encoded.reindex(columns=['participantid'] + expected_features, fill_value=0)
    df_filtered_ios_2 = df_encoded

    # One-hot EMA (Android)
    df = df_filtered_android_2
    df_encoded = pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode, dtype=int)
    df_encoded['participantid'] = df['participantid']
    df_encoded = df_encoded.reindex(columns=['participantid'] + expected_features, fill_value=0)
    df_filtered_android_2 = df_encoded

    df_combined_ios = pd.concat(
        [
            df_filtered_ios_1.set_index('participantid'),
            df_filtered_ios_2.set_index('participantid'),
            df_filtered_ios_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()

    df_combined_android = pd.concat(
        [
            df_filtered_android_1.set_index('participantid'),
            df_filtered_android_2.set_index('participantid'),
            df_filtered_android_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()

    df_selected_others_ios = df_combined_ios
    df_selected_others_android = df_combined_android

    df_selected_others_cleaned_ios = df_selected_others_ios.dropna()
    df_selected_others_cleaned_android = df_selected_others_android.dropna()

    # Numeric columns iOS (NOT scaled here; scaling is inside the CV loop)
    columns_to_normalize = [col for col in selected_columns_1 if col != 'participantid']
    df_numeric = df_selected_others_cleaned_ios[columns_to_normalize].apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.dropna()
    df_normalized = df_numeric.reset_index(drop=True)
    df_normalized_with_participantid = pd.concat([df_selected_others_cleaned_ios[['participantid']].reset_index(drop=True),
                                                  df_normalized.reset_index(drop=True)], axis=1)
    df_filtered_ios_1 = df_normalized_with_participantid

    # Numeric columns Android (NOT scaled here; scaling is inside the CV loop)
    df_numeric = df_selected_others_cleaned_android[columns_to_normalize].apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.dropna()
    df_normalized = df_numeric.reset_index(drop=True)
    df_normalized_with_participantid = pd.concat([df_selected_others_cleaned_android[['participantid']].reset_index(drop=True),
                                                  df_normalized.reset_index(drop=True)], axis=1)
    df_filtered_android_1 = df_normalized_with_participantid

    mappings = {
        'sex': {'Female': 1, 'Male': 0},
        'education': {
            'Finished university': 4,
            'Further education': 3,
            'Finished high school': 2,
            'Student at university': 1,
            'Currently in high school': 0
        },
        'sexual_orientation': {'Straight': 1, 'Non-Straight': 0},
        'ethnicity': {'White': 1, 'Non-White': 0}
    }
    columns_to_encode_dm = ['sex', 'education', 'sexual_orientation', 'ethnicity']

    # Demographics iOS
    df = df_filtered_ios_3.copy()
    participant_id = df[['participantid']]
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
            df[column] = df[column].apply(lambda x: mapping.get(x, -1))
    df_selected_others_cleaned_ios = pd.concat([participant_id, df[columns_to_encode_dm]], axis=1)
    df_filtered_ios_3 = df_selected_others_cleaned_ios

    df_combined_ios = pd.concat(
        [
            df_filtered_ios_1.set_index('participantid'),
            df_filtered_ios_2.set_index('participantid'),
            df_filtered_ios_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()

    # Demographics Android
    df = df_filtered_android_3.copy()
    participant_id = df[['participantid']]
    for column, mapping in mappings.items():
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
            df[column] = df[column].apply(lambda x: mapping.get(x, -1))
    df_selected_others_cleaned_android = pd.concat([participant_id, df[columns_to_encode_dm]], axis=1)
    df_filtered_android_3 = df_selected_others_cleaned_android

    df_combined_android = pd.concat(
        [
            df_filtered_android_1.set_index('participantid'),
            df_filtered_android_2.set_index('participantid'),
            df_filtered_android_3.set_index('participantid')
        ],
        axis=1
    ).reset_index()

    # Labels iOS
    result_df_ios = new_df_lessfeatures_ms_ios[new_df_lessfeatures_ms_ios['participantid'].isin(df_combined_ios['participantid'])]
    result_df_ios = result_df_ios[['participantid', 'disorder_name___10']]
    df_combined_ios = pd.concat(
        [
            df_filtered_ios_1.set_index('participantid'),
            df_filtered_ios_2.set_index('participantid'),
            df_filtered_ios_3.set_index('participantid'),
            result_df_ios.set_index('participantid')
        ],
        axis=1
    ).reset_index()

    # Labels Android
    result_df_android = df1_filtered_android[df1_filtered_android['participantid'].isin(df_combined_android['participantid'])]
    result_df_android = result_df_android[['participantid', 'disorder_name___10']]
    df_combined_android = pd.concat(
        [
            df_filtered_android_1.set_index('participantid'),
            df_filtered_android_2.set_index('participantid'),
            df_filtered_android_3.set_index('participantid'),
            result_df_android.set_index('participantid')
        ],
        axis=1
    ).reset_index()

    # Platform column (used for per-platform scaling inside the CV loop)
    df_combined_ios['platform'] = 1
    df_combined_android['platform'] = 0

    df_combined_all = pd.concat([df_combined_ios, df_combined_android], ignore_index=True)

    df_combined_all['disorder_name___10_encoded'] = df_combined_all['disorder_name___10'].apply(lambda x: 1 if x == 'Anxiety disorder' else 0)

    y=df_combined_all['disorder_name___10_encoded']
    X=df_combined_all.drop(['participantid','disorder_name___10',  'disorder_name___10_encoded'], axis=1)
    class_counts = pd.Series(y).value_counts()
    print(class_counts)
    print(X.shape)

    return X, y
