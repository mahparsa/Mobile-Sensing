import numpy as np

ACTIVITY_COLS = [
    "acc_vigorous_pa_minutes",
    "acc_non_vigorous_pa_minutes",
    "acc_sedentary_minutes",
    "acc_total_active_minutes",
    "acc_mean_activity_intensity",
    "acc_max_activity_intensity",
    "acc_activity_intensity_std",
    "acc_activity_bout_count",
    "acc_longest_sedentary_bout_minutes",
    "acc_activity_fragmentation_index",
    "acc_morning_activity_minutes",
    "acc_afternoon_activity_minutes",
    "acc_evening_activity_minutes",
    "acc_night_activity_minutes",
    "acc_accelerometer_sample_count",
    "acc_data_coverage_hours",
]


def simulate_activity_features(persona, mood_deficit, awake_minutes, chronotype_offset, num_people, rng):
    activity_level = persona["activity_level"].to_numpy()

    vigorous = np.clip(activity_level * 35 + rng.normal(0, 8, num_people), 0, 180)
    non_vigorous = np.clip(
        activity_level * 70 + (1 - mood_deficit) * 30 + rng.normal(0, 15, num_people), 0, 200
    )
    total_active = vigorous + non_vigorous
    sedentary = np.clip(awake_minutes - total_active, 0, 1200)

    mean_intensity = np.clip(0.5 + 2.2 * activity_level + rng.normal(0, 0.3, num_people), 0, 5)
    max_intensity = np.clip(mean_intensity * 1.6 + rng.normal(0, 0.4, num_people), 0, 10)
    intensity_std = np.clip(0.3 + 1.4 * (1 - activity_level) + rng.normal(0, 0.2, num_people), 0, 5)

    bout_count = np.clip(rng.poisson(4 + 14 * activity_level), 0, 100)
    longest_sedentary_bout = np.clip(
        30 + (1 - activity_level) * 280 + mood_deficit * 150 + rng.normal(0, 30, num_people), 0, 600
    )
    fragmentation_index = np.clip(bout_count / (total_active + 1), 0, 1)

    morning, afternoon, evening, night = _split_activity_by_time_of_day(
        total_active, mood_deficit, chronotype_offset, num_people
    )

    sample_count = np.clip((awake_minutes / 60) * rng.uniform(700, 1300, num_people), 5000, 30000)
    coverage_hours = np.clip((awake_minutes / 60) * rng.uniform(0.75, 1.0, num_people), 1, 24)

    return {
        "acc_vigorous_pa_minutes": vigorous,
        "acc_non_vigorous_pa_minutes": non_vigorous,
        "acc_sedentary_minutes": sedentary,
        "acc_total_active_minutes": total_active,
        "acc_mean_activity_intensity": mean_intensity,
        "acc_max_activity_intensity": max_intensity,
        "acc_activity_intensity_std": intensity_std,
        "acc_activity_bout_count": bout_count,
        "acc_longest_sedentary_bout_minutes": longest_sedentary_bout,
        "acc_activity_fragmentation_index": fragmentation_index,
        "acc_morning_activity_minutes": morning,
        "acc_afternoon_activity_minutes": afternoon,
        "acc_evening_activity_minutes": evening,
        "acc_night_activity_minutes": night,
        "acc_accelerometer_sample_count": sample_count,
        "acc_data_coverage_hours": coverage_hours,
    }


def _split_activity_by_time_of_day(total_active, mood_deficit, chronotype_offset, num_people):
    """Spread total active minutes across morning/afternoon/evening/night,
    shifted earlier or later depending on each participant's chronotype."""
    morning_bias = np.clip(0.30 - 0.05 * chronotype_offset, 0.05, 0.6)
    weights = np.stack([
        morning_bias,
        np.full(num_people, 0.30),
        np.full(num_people, 0.25),
        np.clip(0.15 + mood_deficit * 0.1, 0.05, 0.4),
    ], axis=1)
    weights = weights / weights.sum(axis=1, keepdims=True)

    split = weights * total_active[:, None]
    return np.clip(split, 0, 120).T
