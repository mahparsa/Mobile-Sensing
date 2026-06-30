import numpy as np
import pandas as pd

from .config import DEMOGRAPHIC_COLS
from .gps_features import GPS_COLS, simulate_gps_features
from .sleep_features import SLEEP_COLS, simulate_sleep_features
from .activity_features import ACTIVITY_COLS, simulate_activity_features
from .screen_features import SCREEN_COLS, simulate_screen_features

DOMAIN_COLS = {
    "gps": GPS_COLS,
    "sleep": SLEEP_COLS,
    "activity": ACTIVITY_COLS,
    "screen": SCREEN_COLS,
}


def behavior_cols_for(domains):
    """Column list for just the requested domains, in a stable order."""
    return [col for domain in ["gps", "sleep", "activity", "screen"] if domain in domains
            for col in DOMAIN_COLS[domain]]


def final_cols_for(domains):
    return (
        ["participantid", "date_local"]
        + behavior_cols_for(domains)
        + DEMOGRAPHIC_COLS
        + ["daily_sum", "daily_average_score"]
    )


def simulate_one_day(persona, mood_deficit, chronotype_offset, num_people, rng, domains):
    """Simulate every feature domain for one day, then keep only the
    requested domains' columns in the output.

    Sleep is always simulated internally (even if "sleep" isn't in
    `domains`) because awake-time, which other domains depend on, comes
    from it. It's just dropped from the final columns if not requested.
    """
    sleep = simulate_sleep_features(persona, mood_deficit, num_people, rng)
    awake_minutes = 1440 - sleep["sleep_total_sleep_minutes"]

    pieces = {**sleep}

    if "gps" in domains:
        gps_rows = [
            simulate_gps_features(
                persona["home_lat"][i], persona["home_lon"][i],
                persona["mobility_level"][i], awake_minutes[i], rng,
            )
            for i in range(num_people)
        ]
        pieces.update(pd.DataFrame(gps_rows).to_dict(orient="list"))

    if "activity" in domains:
        pieces.update(
            simulate_activity_features(persona, mood_deficit, awake_minutes, chronotype_offset, num_people, rng)
        )

    if "screen" in domains:
        pieces.update(
            simulate_screen_features(persona, mood_deficit, chronotype_offset, num_people, rng)
        )

    day_df = pd.DataFrame(pieces)

    # composite scores -- computed from whatever was simulated, even if some
    # of those columns aren't in the final requested output
    active_minutes = day_df["acc_total_active_minutes"] if "acc_total_active_minutes" in day_df else 0
    screen_seconds = day_df["screen_total_screen_time_in_seconds"] if "screen_total_screen_time_in_seconds" in day_df else 0
    day_df["daily_sum"] = np.clip(
        active_minutes + screen_seconds / 60 + day_df["sleep_total_sleep_minutes"] / 2,
        0, 2000,
    )
    day_df["daily_average_score"] = np.clip(day_df["daily_sum"] / 20, 0, 100)

    keep_cols = [c for c in behavior_cols_for(domains) if c in day_df.columns] + ["daily_sum", "daily_average_score"]
    return day_df[keep_cols]


def combine_with_demographics(behavioral_df, demographics_df, num_days, domains):
    """Repeat each participant's fixed demographics across their daily rows
    and stitch everything into the final, ordered dataset."""
    demo_repeated = demographics_df.loc[demographics_df.index.repeat(num_days)].reset_index(drop=True)
    dataset = pd.concat([behavioral_df, demo_repeated], axis=1)
    return dataset[final_cols_for(domains)]
