import argparse

from .config import DEFAULT_NUM_PEOPLE, DEFAULT_NUM_DAYS, DEFAULT_SEED, ALL_DOMAINS
from .generator import generate


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic mobile sensing dataset."
    )
    parser.add_argument("--num-people", type=int, default=DEFAULT_NUM_PEOPLE,
                         help=f"Number of participants (default: {DEFAULT_NUM_PEOPLE})")
    parser.add_argument("--num-days", type=int, default=DEFAULT_NUM_DAYS,
                         help=f"Number of days per participant (default: {DEFAULT_NUM_DAYS})")
    parser.add_argument("--domains", type=str, default=",".join(ALL_DOMAINS),
                         help=f"Comma-separated list of domains to include, any of "
                              f"{ALL_DOMAINS} (default: all)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                         help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--output", type=str, default="synthetic_diffusion_dataset.csv",
                         help="Output CSV path (default: synthetic_diffusion_dataset.csv)")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    df = generate(
        num_people=args.num_people,
        num_days=args.num_days,
        domains=domains,
        seed=args.seed,
    )
    df.to_csv(args.output, index=False)

    print(f"wrote {args.output}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())


if __name__ == "__main__":
    main()
