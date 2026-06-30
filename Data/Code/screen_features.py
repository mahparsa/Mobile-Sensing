import numpy as np

SCREEN_COLS = [
    "screen_total_screen_time_in_seconds",
    "screen_num_of_events_total",
    "screen_morning_screen_time_in_seconds",
    "screen_num_of_events_morning",
    "screen_afternoon_screen_time_in_seconds",
    "screen_num_of_events_afternoon",
    "screen_evening_screen_time_in_seconds",
    "screen_num_of_events_evening",
    "screen_nighttime_screen_time_in_seconds",
    "screen_num_of_events_nighttime",
]


def simulate_screen_features(persona, mood_deficit, chronotype_offset, num_people, rng):
    screen_habit = persona["screen_habit"].to_numpy()

    total_seconds = np.clip(
        3600 * (1 + 4 * screen_habit + 2.5 * mood_deficit) + rng.normal(0, 600, num_people),
        0, 20000,
    )
    total_events = np.clip(rng.poisson(20 + 150 * screen_habit), 0, 300)

    time_split, event_split = _split_screen_by_time_of_day(
        total_seconds, total_events, mood_deficit, chronotype_offset, num_people
    )

    morning_t, afternoon_t, evening_t, night_t = time_split
    morning_e, afternoon_e, evening_e, night_e = event_split

    return {
        "screen_total_screen_time_in_seconds": total_seconds,
        "screen_num_of_events_total": total_events,
        "screen_morning_screen_time_in_seconds": morning_t,
        "screen_num_of_events_morning": morning_e,
        "screen_afternoon_screen_time_in_seconds": afternoon_t,
        "screen_num_of_events_afternoon": afternoon_e,
        "screen_evening_screen_time_in_seconds": evening_t,
        "screen_num_of_events_evening": evening_e,
        "screen_nighttime_screen_time_in_seconds": night_t,
        "screen_num_of_events_nighttime": night_e,
    }


def _split_screen_by_time_of_day(total_seconds, total_events, mood_deficit, chronotype_offset, num_people):
    night_bias = np.clip(0.10 + 0.20 * mood_deficit, 0.05, 0.5)
    weights = np.stack([
        np.clip(0.25 - 0.04 * chronotype_offset, 0.05, 0.5),
        np.full(num_people, 0.30),
        np.full(num_people, 0.25),
        night_bias,
    ], axis=1)
    weights = weights / weights.sum(axis=1, keepdims=True)

    time_split = np.clip(weights * total_seconds[:, None], 0, 8000).T
    event_split = np.clip(weights * total_events[:, None], 0, 120).T

    return time_split, event_split
