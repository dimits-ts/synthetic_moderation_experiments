import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tasks.constants
import tasks.graphs


def get_toxicity_df(main_df_path: Path, toxicity_df_path: Path) -> pd.DataFrame:
    df = pd.read_csv(main_df_path)
    df: pd.DataFrame = df.loc[
        (df.intent != "Moderator") & (df.model != "hardcoded")
    ]
    variant_name_dict = {
        "constructive": "Facilitation",
        "erulemaking.txt": "Moderation",
        "vanilla.txt": "No instructions",
    }
    df.tag_2 = df.tag_2.replace(variant_name_dict)
    
    toxicity_df = pd.read_csv(toxicity_df_path)
    full_df = df.merge(right=toxicity_df, how="left", )

def main(input_csv_path: Path, output_dir: Path):
    tasks.graphs.seaborn_setup()

    
    tasks.stats.mean_comp_test(
        df=analysis_df, feature_col="conv_variant", score_col="Toxicity"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Perspective API scoring and save results to CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Graph output directory",
    )
    args = parser.parse_args()
    main(input_csv_path=Path(args.input_csv), output_dir=Path(args.output_dir))
