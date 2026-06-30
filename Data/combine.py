import numpy as np
import pandas as pd

from config import NUM_PEOPLE, DEMOGRAPHIC_COLS
from gps_features import GPS_COLS, simulate_gps_features
from sleep_features import SLEEP_COLS, simulate_sleep_features
from activity_features import ACTIVITY_COLS, simulate_activity_features
from screen_features import SCREEN_COLS, simulate_screen_features

# the full behavioral feature set, assembled from each domain's own columns
BEHAVIOR_COLS = GPS_COLS + SLEEP_COLS + ACTIVITY_COLS + SCREEN_COLS

FINAL_COLS = (
    ["participantid", "date_local"]
    + BEHAVIOR_COLS
    + DEMOGRAPHIC_COLS
    + ["daily_sum", "daily_average_score"]
)


def simulate_one_day(persona, mood_deficit, chronotype_offset):
    """Simulate every feature domain for one day, then combine them into a
    single per-participant row."""
    sleep = simulate_sleep_features(persona, mood_deficit)
    awake_minutes = 1440 - sleep["sleep_total_sleep_minutes"]

    gps_rows = [
        simulate_gps_features(
            persona["home_lat"][i], persona["home_lon"][i],
            persona["mobility_level"][i], awake_minutes[i],
        )
        for i in range(NUM_PEOPLE)
    ]
    gps = pd.DataFrame(gps_rows).to_dict(orient="list")

    activity = simulate_activity_features(persona, mood_deficit, awake_minutes, chronotype_offset)
    screen = simulate_screen_features(persona, mood_deficit, chronotype_offset)

    day_df = pd.DataFrame({**gps, **sleep, **activity, **screen})

    # composite scores, derived from the combined day rather than sampled directly
    day_df["daily_sum"] = np.clip(
        day_df["acc_total_active_minutes"]
        + day_df["screen_total_screen_time_in_seconds"] / 60
        + day_df["sleep_total_sleep_minutes"] / 2,
        0, 2000,
    )
    day_df["daily_average_score"] = np.clip(day_df["daily_sum"] / 20, 0, 100)

    return day_df[BEHAVIOR_COLS + ["daily_sum", "daily_average_score"]]


def combine_with_demographics(behavioral_df, demographics_df, num_days):
    """Repeat each participant's fixed demographics across their daily rows
    and stitch everything into the final, ordered dataset."""
    demo_repeated = demographics_df.loc[demographics_df.index.repeat(num_days)].reset_index(drop=True)
    dataset = pd.concat([behavioral_df, demo_repeated], axis=1)
    return dataset[FINAL_COLS]
