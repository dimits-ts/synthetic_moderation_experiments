import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import tasks.graphs
import tasks.stats


def main(
    main_csv_path: Path,
    ablation_csv_path: Path,
    human_csv_path: Path,
    graph_output_dir: Path,
    cache_dir: Path,
):
    tasks.graphs.seaborn_setup()
    tqdm.pandas()

    main_df = pd.read_csv(main_csv_path)

    human_df = pd.read_csv(human_csv_path)
    human_df = human_df.rename(columns={"text": "message"})
    human_df["model"] = "Human"
    human_df["variant"] = "Human"
    human_df["user_prompts"] = "Human"
    human_df["turn_taking"] = "Human"

    combined_df = pd.concat([main_df, human_df], ignore_index=True)

    plot_dataset_length(
        df=combined_df,
        y_col="model",
        graph_output_dir=graph_output_dir,
    )

    plot_dataset_diversity(
        df=combined_df,
        y_col="model",
        graph_output_path=graph_output_dir / "diversity_main_model.png",
        cache_path=cache_dir / "diversity_main_model.csv",
    )
    plot_dataset_diversity(
        df=combined_df,
        y_col="variant",
        graph_output_path=graph_output_dir / "diversity_main_variant.png",
        cache_path=cache_dir / "diversity_main_variant.csv",
    )

    dataset_stats(main_df, main_csv_path)

    ablation_df = pd.read_csv(ablation_csv_path)
    dataset_stats(ablation_df, ablation_csv_path)
    full_df = pd.concat([main_df, ablation_df, human_df], ignore_index=True)

    plot_dataset_diversity(
        df=full_df,
        y_col="user_prompts",
        graph_output_path=graph_output_dir / "diversity_full_userprompts.png",
        cache_path=cache_dir / "diversity_full_userprompts.csv",
    )
    plot_dataset_diversity(
        df=full_df,
        y_col="turn_taking",
        graph_output_path=graph_output_dir / "diversity_full_turntaking.png",
        cache_path=cache_dir / "diversity_full_turntaking.csv",
    )

    optimal_model_df = full_df.loc[full_df.model.isin(["qwen7b", "Human"])]
    plot_dataset_diversity(
        df=optimal_model_df,
        y_col="user_prompts",
        graph_output_path=graph_output_dir
        / "diversity_optimal_userprompts.png",
        cache_path=cache_dir / "diversity_optimal_userprompts.csv",
    )
    plot_dataset_diversity(
        df=optimal_model_df,
        y_col="turn_taking",
        graph_output_path=graph_output_dir
        / "diversity_optimal_turntaking.png",
        cache_path=cache_dir / "diversity_optimal_turntaking.csv",
    )


def plot_dataset_length(
    df: pd.DataFrame, y_col: str, graph_output_dir: Path
) -> None:
    len_df = df.loc[df.model != "hardcoded", ["message", y_col]]
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
    tasks.graphs.save_plot(graph_output_dir / "comment_len_model.png")
    plt.close()


def plot_dataset_diversity(
    df: pd.DataFrame,
    y_col: str,
    graph_output_path: Path,
    cache_path: Path,
):
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cache if available
    if cache_path.exists():
        print(f"Loading cached similarities from {cache_path}")
        similarity_df = pd.read_csv(cache_path)

    else:
        print("Computing similarities (cache miss)")
        working_df = df.loc[df.model != "hardcoded"]

        similarity_df = (
            working_df.groupby(["conv_id", y_col])["message"]
            .apply(list)
            .reset_index()
        )

        similarity_df["rougel_similarity"] = similarity_df[
            "message"
        ].progress_apply(tasks.stats.rougel_similarity)

        similarity_df = similarity_df.dropna(subset=["rougel_similarity"])

        similarity_df.to_csv(cache_path, index=False)
        print(f"Saved cache → {cache_path}")

    sns.histplot(
        data=similarity_df,
        x="rougel_similarity",
        hue=y_col,
        stat="density",
        common_norm=False,
        multiple="dodge",
        kde=True,
        bins=20,
    )
    plt.xlim(0.6, 1)
    plt.xlabel("Diversity")
    tasks.graphs.save_plot(graph_output_path)
    plt.close()


def dataset_stats(df: pd.DataFrame, csv_path: Path):
    print("*" * 25)
    print("Comments per discussion:")
    print(df.groupby("conv_id").size().describe())

    print("#Comments:", len(df))

    print("#Discussions:", df["conv_id"].nunique())

    print("Word count per comment:")
    print(
        df.message.astype(str)
        .apply(lambda x: x.split())
        .apply(len)
        .astype(int)
        .describe()
    )
    print(f"Dataset total size: {_convert_bytes(csv_path.stat().st_size)}")
    print("*" * 25)


def _convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Get statistics about length and "
            "linguistic diversity of the dataset."
        )
    )
    parser.add_argument(
        "--main-output-dir",
        type=str,
        help="Directory holding the VMD and ablation datasets",
    )
    parser.add_argument(
        "--human-csv",
        type=str,
        help="CSV file containing human responses",
    )
    parser.add_argument(
        "--graph-output-dir",
        type=str,
        help="Graph output directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory to store similarity caches",
    )
    args = parser.parse_args()
    main(
        main_csv_path=Path(args.main_output_dir) / "vmd.csv",
        ablation_csv_path=Path(args.main_output_dir) / "ablation.csv",
        human_csv_path=Path(args.human_csv),
        graph_output_dir=Path(args.graph_output_dir),
        cache_dir=Path(args.cache_dir),
    )
