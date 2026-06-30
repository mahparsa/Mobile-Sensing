# mobile-sensing-sim

Synthetic mobile sensing data generator for my anxiety / mental health
sensing project. Real data comes from a sensing app participants had on
their phones, but since I can't share that, this generates a fake version
with the same structure so people can build/test pipelines without needing
real participant data.

## Install

Straight from GitHub, no need to clone anything:

```bash
pip install "git+https://github.com/mahparsa/Mobile-Sensing.git#subdirectory=Data/Code"

```


## Use it as a library

```python
import mobile_sensing_sim as mss

# default: 2000 participants, 7 days each, all feature domains
df = mss.generate()

# pick your own size
df = mss.generate(num_people=500, num_days=14)

# only want certain feature domains? pick from "gps", "sleep", "activity", "screen"
df = mss.generate(num_people=500, num_days=7, domains=["sleep", "screen"])

df.to_csv("my_dataset.csv", index=False)
```

Demographic/clinical columns (age, sex, diagnosis, depression score, etc.)
and the daily summary scores are always included — `domains` just controls
which of the four behavioral feature groups (GPS, sleep, activity, screen)
show up.

## Use it from the command line

```bash
generate-mobile-data --num-people 500 --num-days 7 --domains gps,sleep --output my_dataset.csv
```

Options:
- `--num-people` (default 2000)
- `--num-days` (default 7)
- `--domains` comma-separated, any of `gps,sleep,activity,screen` (default: all)
- `--seed` (default 7, for reproducibility)
- `--output` (default `synthetic_diffusion_dataset.csv`)

## How the data is generated

Each fake participant gets a "persona" (chronotype, activity level,
mobility level, screen habit, home location, plus demographics including
a simulated mental health diagnosis and depression score). Each day's
features are then *derived* from a simulated routine for that persona —
sleep first (which sets awake time), then GPS itinerary, activity, and
screen use, each shaped by the persona and a hidden "mood" factor so
diagnosed participants tend to sleep less and use their phone more at
night, similar to patterns seen in real research.

All data is fully synthetic, not from real people.
