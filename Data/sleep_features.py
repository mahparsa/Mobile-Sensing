import numpy as np

from config import NUM_PEOPLE, RNG

SLEEP_COLS = [
    "sleep_total_sleep_minutes",
    "sleep_avg_sleep_hours",
    "sleep_total_night_screen_minutes",
    "sleep_total_screen_interruptions",
]


def simulate_sleep_features(persona, mood_deficit):
    duration_hours = np.clip(
        7.5 - 1.5 * mood_deficit + RNG.normal(0, 0.7, NUM_PEOPLE), 4, 10
    )
    total_sleep_minutes = duration_hours * 60

    night_screen_minutes = np.clip(
        15 + 70 * mood_deficit * persona["screen_habit"] + RNG.normal(0, 10, NUM_PEOPLE),
        0, 300,
    )
    interruptions = np.clip(
        RNG.poisson(1 + 4 * mood_deficit * persona["screen_habit"]), 0, 10
    )

    return {
        "sleep_total_sleep_minutes": total_sleep_minutes,
        "sleep_avg_sleep_hours": duration_hours,
        "sleep_total_night_screen_minutes": night_screen_minutes,
        "sleep_total_screen_interruptions": interruptions,
    }
