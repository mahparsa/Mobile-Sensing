import pandas as pd


def make_personas(num_people, rng):
    """Fixed per-participant traits that drive their simulated daily routine."""
    return pd.DataFrame({
        "chronotype_offset": rng.normal(0, 1.0, num_people),     # hrs, neg = early bird
        "activity_level": rng.beta(2, 2, num_people),            # 0-1
        "mobility_level": rng.beta(2, 2, num_people),            # 0-1, homebody -> roamer
        "screen_habit": rng.beta(2, 2, num_people),              # 0-1
        "home_lat": 40.0 + rng.uniform(-0.15, 0.15, num_people),
        "home_lon": -74.0 + rng.uniform(-0.15, 0.15, num_people),
    })


def make_demographics(num_people, rng):
    completed_8_weeks = rng.choice([0, 1], size=num_people, p=[0.2, 0.8])
    age = rng.integers(18, 66, size=num_people)
    biological_sex = rng.choice(["Male", "Female"], size=num_people)
    identifying_gender = rng.choice(
        ["Man", "Woman", "Non-binary", "Prefer not to say"],
        size=num_people, p=[0.46, 0.46, 0.05, 0.03],
    )
    phone = rng.choice(["iOS", "Android"], size=num_people, p=[0.55, 0.45])
    mental_health_diagnosis = rng.choice([0, 1], size=num_people, p=[0.6, 0.4])

    disorder_name___1 = (mental_health_diagnosis == 1) & (rng.random(num_people) < 0.5)
    disorder_name___10 = (mental_health_diagnosis == 1) & (rng.random(num_people) < 0.3)

    # 1-60 scale; diagnosed participants skew higher
    depression_score = pd.Series(
        rng.integers(1, 25, size=num_people)
    ).where(mental_health_diagnosis == 0, rng.integers(20, 61, size=num_people))

    return pd.DataFrame({
        "completed_8_weeks": completed_8_weeks,
        "age": age,
        "biological_sex": biological_sex,
        "identifying_gender": identifying_gender,
        "phone": phone,
        "mental_health_diagnosis": mental_health_diagnosis,
        "disorder_name___10": disorder_name___10.astype(int),
        "disorder_name___1": disorder_name___1.astype(int),
        "depression_score": depression_score.to_numpy(),
    })
