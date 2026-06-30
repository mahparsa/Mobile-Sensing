import numpy as np

SLEEP_COLS = [
    "sleep_total_sleep_minutes",
    "sleep_avg_sleep_hours",
    "sleep_total_night_screen_minutes",
    "sleep_total_screen_interruptions",
]


def simulate_sleep_features(persona, mood_deficit, num_people, rng):
    duration_hours = np.clip(
        7.5 - 1.5 * mood_deficit + rng.normal(0, 0.7, num_people), 4, 10
    )
    total_sleep_minutes = duration_hours * 60

    night_screen_minutes = np.clip(
        15 + 70 * mood_deficit * persona["screen_habit"] + rng.normal(0, 10, num_people),
        0, 300,
    )
    interruptions = np.clip(
        rng.poisson(1 + 4 * mood_deficit * persona["screen_habit"]), 0, 10
    )

    return {
        "sleep_total_sleep_minutes": total_sleep_minutes,
        "sleep_avg_sleep_hours": duration_hours,
        "sleep_total_night_screen_minutes": night_screen_minutes,
        "sleep_total_screen_interruptions": interruptions,
    }
