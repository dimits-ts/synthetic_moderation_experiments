import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm

import tasks.graphs


def plot_dataset_length(df: pd.DataFrame, y_col: str) -> None:
    len_df = df.loc[:, ["message", y_col]]
    len_df["comment_length"] = len_df.message.apply(lambda x: len(x.split()))
    sns.histplot(
        data=len_df,
        x="comment_length",
        hue=y_col,
        common_norm=False,
        stat="density",
    )
    plt.xlim(0, 400)
    plt.xlabel(r"Comment length (\# words)")


def plot_dataset_diversity(df: pd.DataFrame, y_col: str):
    similarity_df = (
        df.groupby(["conv_id", y_col])["message"].apply(list).reset_index()
    )

    similarity_df["rougel_similarity"] = similarity_df[
        "message"
    ].progress_apply(tasks.stats.rougel_similarity)

    similarity_df = similarity_df.dropna(subset=["rougel_similarity"])

    sns.histplot(
        data=similarity_df,
        x="rougel_similarity",
        hue=y_col,
        stat="density",
        common_norm=False,
    )
    plt.xlabel("Diversity")


def main(input_csv_path: Path, output_dir: Path):
    tasks.graphs.seaborn_setup()
    tqdm.pandas()
    df = pd.read_csv(input_csv_path)

    plot_dataset_length(df=df, y_col="model")
    tasks.graphs.save_plot(output_dir / "comment_len_model.png")
    plt.close()

    plot_dataset_diversity(df=df, y_col="model")
    tasks.graphs.save_plot(output_dir / "comment_diversity_model.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Get statistics about length and "
            "linguistic diversity of the dataset."
        )
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
