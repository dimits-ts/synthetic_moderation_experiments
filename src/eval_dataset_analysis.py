import argparse
from pathlib import Path

import pandas as pd
import numpy as np
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
    human_df["initialization"] = "Human"

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
    for dimension in ["user_prompts", "turn_taking", "initialization"]:
        plot_dataset_diversity(
            df=full_df,
            y_col=dimension,
            graph_output_path=graph_output_dir
            / f"diversity_full_{dimension}.png",
            cache_path=cache_dir / f"diversity_full_{dimension}.csv",
        )

        optimal_model_df = full_df.loc[
            full_df.model.isin(["qwen7b", "Human", "mistral24b", "llama70b"])
        ]
        plot_dataset_diversity(
            df=optimal_model_df,
            y_col=dimension,
            graph_output_path=graph_output_dir
            / f"diversity_optimal_{dimension}.png",
            cache_path=cache_dir / f"diversity_optimal_{dimension}.csv",
        )

        kl_df = compute_kl_divergence_to_human(
            df=full_df,
            dimensions=["user_prompts", "turn_taking", "initialization"],
            cache_dir=cache_dir
        )

        output_path = eval_output / "kl_divergence.csv"
        kl_df.to_csv(output_path)


def compute_kl_divergence_to_human(
    df: pd.DataFrame,
    dimensions: list[str],
    cache_dir: Path,
    bins: int = 50,
):
    """
    Produces a hierarchical CSV:
        level 1 → dimension
        level 2 → value within dimension
        rows    → model names
        cells   → KL(model || Human)
    """

    eval_output.mkdir(parents=True, exist_ok=True)
    results = {}

    working_df = df.loc[df.model != "hardcoded"]

    for dimension in dimensions:
        print(f"Computing KL for dimension: {dimension}")

        cache_path = cache_dir / f"diversity_full_{dimension}.csv"

        if cache_path.exists():
            similarity_df = pd.read_csv(cache_path)
        else:
            grouped = (
                working_df.groupby(["conv_id", "model", dimension])["message"]
                .apply(list)
                .reset_index()
            )
            grouped["rougel_similarity"] = grouped["message"].progress_apply(
                tasks.stats.rougel_similarity
            )
            similarity_df = grouped.dropna(subset=["rougel_similarity"])
            similarity_df.to_csv(cache_path, index=False)

        dim_results = {}

        for value in similarity_df[dimension].unique():
            subset = similarity_df[similarity_df[dimension] == value]

            human_vals = subset.loc[
                subset.model == "Human", "rougel_similarity"
            ].values

            if len(human_vals) == 0:
                continue

            hist_range = (0.6, 1.0)
            human_hist, edges = np.histogram(
                human_vals, bins=bins, range=hist_range, density=True
            )
            human_hist += 1e-10  # smoothing

            value_results = {}

            for model in subset.model.unique():
                if model == "Human":
                    continue

                model_vals = subset.loc[
                    subset.model == model, "rougel_similarity"
                ].values

                if len(model_vals) == 0:
                    continue

                model_hist, _ = np.histogram(
                    model_vals, bins=bins, range=hist_range, density=True
                )
                model_hist += 1e-10

                kl = np.sum(model_hist * np.log(model_hist / human_hist))
                value_results[model] = kl

            dim_results[value] = value_results

        results[dimension] = dim_results

    # Convert to hierarchical dataframe
    records = []
    for dimension, values in results.items():
        for value, model_scores in values.items():
            row = {
                "dimension": dimension,
                "value": value,
                **model_scores,
            }
            records.append(row)

    out_df = pd.DataFrame(records)
    out_df = out_df.set_index(["dimension", "value"]).sort_index()
    return out_df


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

    sns.kdeplot(
        data=similarity_df,
        x="rougel_similarity",
        hue=y_col,
        fill=True,
        common_norm=False,
        multiple="layer",
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
