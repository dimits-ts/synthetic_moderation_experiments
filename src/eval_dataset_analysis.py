
# Intervention Detection in Discussions
# Copyright (C) 2026 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

import tasks.graphs
import tasks.stats


def main(
    main_csv_path: Path,
    ablation_csv_path: Path,
    human_csv_path: Path,
    graph_output_dir: Path,
    stats_output_dir: Path, 
    cache_dir: Path,
):
    tasks.graphs.seaborn_setup()
    tqdm.pandas()

    main_df = pd.read_csv(main_csv_path)
    main_df = main_df[main_df.model != "hardcoded"]

    human_df = pd.read_csv(human_csv_path)
    human_df = human_df.rename(columns={"text": "message"})
    human_df["model"] = "Human"
    human_df["variant"] = "Human"
    human_df["user_prompts"] = "Human"
    human_df["turn_taking"] = "Human"
    human_df["initialization"] = "Human"
    human_df["sdbs"] = "Human"

    combined_df = pd.concat([main_df, human_df])

    MODEL_ORDER = tasks.graphs.get_sorted_labels(combined_df, "model")

    plot_dataset_length(
        df=combined_df,
        y_col="model",
        graph_output_dir=graph_output_dir,
        label_order=MODEL_ORDER,
    )

    plot_dataset_diversity(
        df=combined_df,
        y_col="model",
        graph_output_path=graph_output_dir / "diversity_main_model.png",
        cache_path=cache_dir / "diversity_main_model.csv",
        label_order=MODEL_ORDER,
    )

    ablation_df = pd.read_csv(ablation_csv_path)
    ablation_df = ablation_df[ablation_df.model != "hardcoded"]

    results_df = compute_js_divergence_to_human(
        full_df=pd.concat([main_df, ablation_df, human_df], ignore_index=True),
        dimensions=["user_prompts", "turn_taking", "initialization", "sdbs"],
        cache_dir=cache_dir,
    )
    output_divergence_results(df=results_df, output_dir=stats_output_dir)

    main_stats = dataset_stats(main_df, "Main")
    ablation_stats = dataset_stats(ablation_df, "Ablation")
    human_stats = dataset_stats(human_df, "Human")

    all_stats_df = (
        main_stats
        .merge(ablation_stats, on="Metric", how="outer")
        .merge(human_stats, on="Metric", how="outer")
    )

    caption = "Dataset statistics for all datasets."
    latex_path = stats_output_dir / "dataset_statistics.tex"

    all_stats_df.to_latex(
        buf=latex_path,
        index=False,
        caption=caption,
        label="tab:dataset-stats",
        float_format="%.2f",
        position="ht",
    )

    print(f"\nSaved dataset statistics → {latex_path}")


def output_divergence_results(df: pd.DataFrame, output_dir: Path) -> None:
    for dimension in df.dimension.unique():
        dim_df = df[df.dimension == dimension]
        dim_df = dim_df.rename(
            columns={
                "js_divergence": "Divergence",
                "model": "Model",
                "value": "Ablation",
            }
        )  # type: ignore
        dim_df = dim_df.pivot(
            index="Ablation", columns="Model", values="Divergence"
        )
        dim_df = dim_df.squeeze()

        caption = (
            "Jensen-Shannon divergence between the diversity "
            "distributions  of human and synthetic discussions by "
            f"{dimension.replace("_", " ")}. "
            "Smaller is better."
        )
        dim_df.to_latex(
            buf=output_dir / f"divergence_{dimension}.tex",
            caption=caption,
            label=f"tab:divergence-{dimension}",
            position="ht",
            float_format="%.3f",
        )

    print(f"\nSaved JS divergence results → {output_dir}")


def compute_js_divergence(
    dist1: np.ndarray, dist2: np.ndarray, bins: int = 50
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions of
    diversity scores.
    Uses KDE to estimate continuous distributions,
     then evaluates on a shared grid.
    """
    # Filter NaNs
    dist1 = dist1[~np.isnan(dist1)]
    dist2 = dist2[~np.isnan(dist2)]

    if len(dist1) < 2 or len(dist2) < 2:
        return np.nan

    # Shared evaluation grid over [0, 1]
    grid = np.linspace(0, 1, bins)

    kde1 = gaussian_kde(dist1)
    kde2 = gaussian_kde(dist2)

    p = kde1(grid)
    q = kde2(grid)

    # Normalize to sum to 1 (make proper probability mass vectors)
    p = p / p.sum()
    q = q / q.sum()

    # jensenshannon returns the square root of JS divergence; square it
    # for the divergence itself
    return jensenshannon(p, q) ** 2


def compute_rougel_similarities_for_group(
    df: pd.DataFrame,
    y_col: str,
    group_value: str,
    cache_dir: Path,
) -> np.ndarray:
    """
    Compute ROUGE-L intra-conversation similarities for a specific group
    (e.g. one model).
    Returns a flat array of similarity scores.
    """
    cache_path = (
        cache_dir / f"rougel_{y_col}_{group_value.replace('/', '_')}.csv"
    )

    if cache_path.exists():
        similarity_df = pd.read_csv(cache_path)
    else:
        working_df = df.loc[df[y_col] == group_value]
        similarity_df = (
            working_df.groupby("conv_id")["message"].apply(list).reset_index()
        )
        similarity_df["rougel_similarity"] = similarity_df[
            "message"
        ].progress_apply(tasks.stats.rougel_similarity)
        similarity_df = similarity_df.dropna(subset=["rougel_similarity"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        similarity_df.to_csv(cache_path, index=False)

    return similarity_df["rougel_similarity"].dropna().values  # type: ignore


def compute_js_divergence_to_human(
    full_df: pd.DataFrame,
    dimensions: list[str],
    cache_dir: Path,
    bins: int = 50,
) -> pd.DataFrame:
    """
    For each (model, dimension) pair, compute the JS divergence between the
    model's ROUGE-L similarity distribution and the Human distribution.

    Outputs a CSV to stats_output_path with columns:
        model | dimension | js_divergence
    """
    records = []
    human_similarities = {}

    models = set(full_df.model.unique())
    for model in models:
        if model == "Human":
            continue
        else:
            df = full_df.loc[(full_df.model.isin([model, "Human"]))]
            # filter out messages that are just a number of quotemarks
            df = df[
                ~df.message.fillna("").astype(str).str.fullmatch(r'[\s"]*')
            ]

        print("Model: ", model)
        for dimension in dimensions:
            print(f"\n[JS Divergence] Dimension: {dimension}")
            model_cache_dir = cache_dir / "js_divergence" / dimension / model

            # Pre-compute Human distribution for this dimension
            human_sim = compute_rougel_similarities_for_group(
                df=df,
                y_col=dimension,
                group_value="Human",
                cache_dir=model_cache_dir,
            )
            human_similarities[dimension] = human_sim
            print(f"  Human samples: {len(human_sim)}")

            non_human_values = df.loc[
                (df[dimension] != "Human"), dimension
            ].unique()

            for value in non_human_values:
                model_sim = compute_rougel_similarities_for_group(
                    df=df,
                    y_col=dimension,
                    group_value=value,
                    cache_dir=model_cache_dir,
                )

                js_div = compute_js_divergence(model_sim, human_sim, bins=bins)
                print(
                    f"{model} {value}: JS divergence = "
                    f"{js_div:.4f} (n={len(model_sim)})"
                )

                records.append(
                    {
                        "dimension": dimension,
                        "model": model,
                        "value": value,
                        "js_divergence": js_div,
                    }
                )
    results_df = pd.DataFrame(records)
    return results_df


def plot_dataset_length(
    df: pd.DataFrame,
    y_col: str,
    graph_output_dir: Path,
    label_order: list[str],
) -> None:
    len_df = df.loc[:, ["message", y_col]]
    len_df["comment_length"] = len_df.message.apply(lambda x: len(x.split()))
    sns.histplot(
        data=len_df,
        x="comment_length",
        hue=y_col,
        hue_order=label_order,
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
    label_order: list[str],
):
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load cache if available
    if cache_path.exists():
        print(f"Loading cached similarities from {cache_path}")
        similarity_df = pd.read_csv(cache_path)

    else:
        print("Computing similarities (cache miss)")
        similarity_df = (
            df.groupby(["conv_id", y_col])["message"].apply(list).reset_index()
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
        hue_order=label_order,
        fill=True,
        common_norm=False,
        multiple="layer",
    )
    plt.xlim(0.6, 1)
    plt.xlabel("Diversity")
    tasks.graphs.save_plot(graph_output_path)
    plt.close()


def dataset_stats(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    comments_per_discussion = df.groupby("conv_id").size().describe()

    word_counts = (
        df.message.astype(str)
        .apply(lambda x: x.split())
        .apply(len)
        .astype(int)
        .describe()
    )

    stats = {
        r"\# Comments": len(df),
        r"\# Discussions": df["conv_id"].nunique(),
        r"Mean Comments per Discussion": comments_per_discussion["mean"],
        r"Std. Comments per Discussion": comments_per_discussion["std"],
        r"Mean Words per Comment": word_counts["mean"],
        r"Std. Words per Comment": word_counts["std"],
    }

    stats_df = pd.DataFrame.from_dict(
        stats, orient="index", columns=[dataset_name]
    )
    stats_df.index.name = "Metric"

    return stats_df.reset_index()


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
        "--stats-output-dir",
        type=str,
        help="KL stats output directory",
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
        stats_output_dir=Path(args.stats_output_dir),
        cache_dir=Path(args.cache_dir),
    )
