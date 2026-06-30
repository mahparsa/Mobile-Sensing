from datetime import timedelta

import numpy as np
import pandas as pd

from config import NUM_PEOPLE, NUM_DAYS, START_DATE
from personas import make_personas, make_demographics
from combine import simulate_one_day, combine_with_demographics


def main():
    persona = make_personas()
    demographics = make_demographics()

    mood_deficit = demographics["depression_score"].to_numpy() / 60.0
    chronotype_offset = persona["chronotype_offset"].to_numpy()

    dates = [START_DATE + timedelta(days=t) for t in range(NUM_DAYS)]

    all_days = []
    for t in range(NUM_DAYS):
        day_df = simulate_one_day(persona, mood_deficit, chronotype_offset)
        day_df.insert(0, "date_local", dates[t].isoformat())
        day_df.insert(0, "participantid", np.arange(NUM_PEOPLE))
        all_days.append(day_df)

    behavioral = (
        pd.concat(all_days, ignore_index=True)
        .sort_values(["participantid", "date_local"])
        .reset_index(drop=True)
    )

    dataset = combine_with_demographics(behavioral, demographics, NUM_DAYS)

    dataset.to_csv("synthetic_diffusion_dataset.csv", index=False)

    print(dataset.shape)
    print(dataset.head())


if __name__ == "__main__":
    main()
