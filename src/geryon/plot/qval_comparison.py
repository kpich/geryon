"""Plot train vs val BH-corrected q-values (output of geryon.eval.batch)."""

import argparse
from pathlib import Path

import pandas as pd

from geryon.eval.plot import plot_qval_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="CSV from geryon.eval.batch"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    plot_qval_comparison(df, args.output)


if __name__ == "__main__":
    main()
