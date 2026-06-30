from datetime import timedelta

import numpy as np
import pandas as pd

from .config import DEFAULT_NUM_PEOPLE, DEFAULT_NUM_DAYS, DEFAULT_SEED, START_DATE, ALL_DOMAINS
from .personas import make_personas, make_demographics
from .combine import simulate_one_day, combine_with_demographics


def generate(num_people=DEFAULT_NUM_PEOPLE, num_days=DEFAULT_NUM_DAYS, domains=None, seed=DEFAULT_SEED):
    """Generate a synthetic mobile sensing dataset.

    Parameters
    ----------
    num_people : int
        Number of simulated participants.
    num_days : int
        Number of days of data per participant.
    domains : list[str] or None
        Which behavioral feature domains to include. Any subset of
        ["gps", "sleep", "activity", "screen"]. Defaults to all of them.
        Demographic/clinical columns and the daily summary scores are
        always included.
    seed : int
        Random seed, for reproducible output.

    Returns
    -------
    pandas.DataFrame
    """
    domains = list(domains) if domains is not None else list(ALL_DOMAINS)
    unknown = set(domains) - set(ALL_DOMAINS)
    if unknown:
        raise ValueError(f"Unknown domain(s): {sorted(unknown)}. Choose from {ALL_DOMAINS}.")

    rng = np.random.default_rng(seed)

    persona = make_personas(num_people, rng)
    demographics = make_demographics(num_people, rng)

    mood_deficit = demographics["depression_score"].to_numpy() / 60.0
    chronotype_offset = persona["chronotype_offset"].to_numpy()

    dates = [START_DATE + timedelta(days=t) for t in range(num_days)]

    all_days = []
    for t in range(num_days):
        day_df = simulate_one_day(persona, mood_deficit, chronotype_offset, num_people, rng, domains)
        day_df.insert(0, "date_local", dates[t].isoformat())
        day_df.insert(0, "participantid", np.arange(num_people))
        all_days.append(day_df)

    behavioral = (
        pd.concat(all_days, ignore_index=True)
        .sort_values(["participantid", "date_local"])
        .reset_index(drop=True)
    )

    return combine_with_demographics(behavioral, demographics, num_days, domains)
