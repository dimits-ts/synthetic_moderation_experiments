import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tasks.constants
import tasks.graphs


def get_toxicity_df(
    main_df_path: Path, toxicity_df_path: Path
) -> pd.DataFrame:
    df = pd.read_csv(main_df_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    toxicity_df = pd.read_csv(toxicity_df_path)
    toxicity_df = toxicity_df.loc[toxicity_df.error.isna()]
    toxicity_df = toxicity_df.loc[
        :, ~toxicity_df.columns.str.contains("^Unnamed")
    ]
    full_df = df.merge(right=toxicity_df, how="inner", on="message_id")
    print(full_df.columns)
    full_df = full_df.loc[
        (full_df.model != "hardcoded") & (~full_df.is_moderator),
        [
            "conv_id",
            "message_id",
            "toxicity",
            "special_instructions",
        ],
    ]
    return full_df


def main(main_output_dir: Path, toxicity_ratings_dir: Path):
    tasks.graphs.seaborn_setup()
    df = get_toxicity_df(
        main_df_path=main_output_dir / "vmd.csv",
        toxicity_df_path=toxicity_ratings_dir / "vmd.csv",
    )
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Perspective API scoring and save results to CSV."
    )
    parser.add_argument(
        "--main-output-dir",
        type=str,
        help="Directory holding the VMD and ablation datasets",
    )
    parser.add_argument(
        "--toxicity-rating-dir",
        type=str,
        help="Directory holding the VMD and ablation toxicity ratings",
    )
    parser.add_argument(
        "--graph-output-dir",
        type=str,
        help="Graph output directory",
    )
    args = parser.parse_args()
    main(
        main_output_dir=Path(args.main_output_dir),
        toxicity_ratings_dir=Path(args.toxicity_rating_dir),
    )
