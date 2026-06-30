# Mobile Sensing Data Simulator

This project looks at how AI-based mobile sensing could help study anxiety.
We collected data from participants through a mobile sensing app and
extracted behavioral features from it (GPS, sleep, activity, screen time).

Since the real data is private, this repo has a script that generates a
**synthetic** version with the same structure, so the pipeline/models can
be built and tested without touching real participant data.

## Files

- `config.py` – shared settings (number of participants/days, column names)
- `geo.py` – distance calculation helper
- `personas.py` – creates each fake participant + their demographics
- `gps_features.py` – simulates location/mobility features
- `sleep_features.py` – simulates sleep features
- `activity_features.py` – simulates accelerometer/activity features
- `screen_features.py` – simulates screen time features
- `combine.py` – puts all the features together into one dataset
- `generate_synthetic_data.py` – run this one, it does everything

## How to run

```bash
pip install numpy pandas
python generate_synthetic_data.py
```

This creates `synthetic_diffusion_dataset.csv` with 2,000 fake participants
over 7 days each.

## Notes

- All data is fake, not from real people.
- Each fake participant has a "mood" level behind the scenes that affects
  their sleep, screen time, and activity — so people with a simulated
  mental health diagnosis tend to sleep less and use their phone more at
  night, similar to patterns seen in real research.
