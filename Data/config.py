from datetime import date
import numpy as np

NUM_PEOPLE = 2000
NUM_DAYS = 7
START_DATE = date(2024, 1, 1)
RNG = np.random.default_rng(7)

DEMOGRAPHIC_COLS = [
    "completed_8_weeks",
    "age",
    "biological_sex",
    "identifying_gender",
    "phone",
    "mental_health_diagnosis",
    "disorder_name___10",
    "disorder_name___1",
    "depression_score",
]
