import pandas as pd

from config import NUM_PEOPLE, RNG


def make_personas():
    """Fixed per-participant traits that drive their simulated daily routine."""
    return pd.DataFrame({
        "chronotype_offset": RNG.normal(0, 1.0, NUM_PEOPLE),     # hrs, neg = early bird
        "activity_level": RNG.beta(2, 2, NUM_PEOPLE),            # 0-1
        "mobility_level": RNG.beta(2, 2, NUM_PEOPLE),            # 0-1, homebody -> roamer
        "screen_habit": RNG.beta(2, 2, NUM_PEOPLE),              # 0-1
        "home_lat": 40.0 + RNG.uniform(-0.15, 0.15, NUM_PEOPLE),
        "home_lon": -74.0 + RNG.uniform(-0.15, 0.15, NUM_PEOPLE),
    })


def make_demographics():
    completed_8_weeks = RNG.choice([0, 1], size=NUM_PEOPLE, p=[0.2, 0.8])
    age = RNG.integers(18, 66, size=NUM_PEOPLE)
    biological_sex = RNG.choice(["Male", "Female"], size=NUM_PEOPLE)
    identifying_gender = RNG.choice(
        ["Man", "Woman", "Non-binary", "Prefer not to say"],
        size=NUM_PEOPLE, p=[0.46, 0.46, 0.05, 0.03],
    )
    phone = RNG.choice(["iOS", "Android"], size=NUM_PEOPLE, p=[0.55, 0.45])
    mental_health_diagnosis = RNG.choice([0, 1], size=NUM_PEOPLE, p=[0.6, 0.4])

    disorder_name___1 = (mental_health_diagnosis == 1) & (RNG.random(NUM_PEOPLE) < 0.5)
    disorder_name___10 = (mental_health_diagnosis == 1) & (RNG.random(NUM_PEOPLE) < 0.3)

    # 1-60 scale; diagnosed participants skew higher
    depression_score = pd.Series(
        RNG.integers(1, 25, size=NUM_PEOPLE)
    ).where(mental_health_diagnosis == 0, RNG.integers(20, 61, size=NUM_PEOPLE))

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
