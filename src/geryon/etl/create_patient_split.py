"""
Create a stable exploration/validation patient split from clinical patient parquet.

Writes patient_split.parquet with columns (PATIENT_ID, split) where split is
'explore' (80%) or 'validation' (20%), assigned randomly with a fixed seed.
"""

import argparse

import numpy as np
import pandas as pd  # type: ignore


def create_split(input_path: str, output_path: str, seed: int) -> None:
    patient_ids = pd.read_parquet(input_path, columns=["PATIENT_ID"])[
        "PATIENT_ID"
    ].unique()

    rng = np.random.default_rng(seed)
    n_holdout = round(len(patient_ids) * 0.2)
    holdout_idx = rng.choice(len(patient_ids), size=n_holdout, replace=False)

    splits = np.full(len(patient_ids), "explore", dtype=object)
    splits[holdout_idx] = "validation"

    pd.DataFrame({"PATIENT_ID": patient_ids, "split": splits}).to_parquet(
        output_path, index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate patient exploration/validation split parquet"
    )
    parser.add_argument(
        "--input", required=True, help="Path to data_clinical_patient.parquet"
    )
    parser.add_argument(
        "--output", required=True, help="Output path for patient_split.parquet"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    create_split(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
