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
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf

import tasks.graphs


def main(
    vmd_path: Path,
    ablation_path: Path,
    human_path: Path,
    toxicity_ratings_dir: Path,
    graph_dir: Path,
    latex_output_dir: Path,
):
    tasks.graphs.seaborn_setup()
    df = get_toxicity_df(
        main_df_path=vmd_path,
        toxicity_df_path=toxicity_ratings_dir / "vmd.csv",
    )

    df["role"] = np.where(
        df.is_moderator,
        "Facilitator",
        np.where(df.is_troll, "Troll", "Non-troll"),
    )

    MODEL_ORDER = tasks.graphs.get_sorted_labels(df, "model")
    STRATEGY_ORDER = tasks.graphs.get_sorted_labels(df, "strategy")

    toxicity_by_dimension(df, graph_dir, "role")
    toxicity_by_dimension(df, graph_dir, "model")
    participant_toxicity_regression(
        df[~df.is_moderator], latex_output_dir=latex_output_dir
    )
    moderator_toxicity_regression(df[df.is_moderator])
    facilitation_response_regression(df)

    palette = tasks.graphs.COLORBLIND_PALETTE
    # shift colors left by 1 so first color is skipped
    # given that humans dont exist in this subset
    offset_palette = palette[1:] + palette[:1]
    toxicity_through_time_plot(
        df=df,
        groupby_col="model",
        graph_output_path=graph_dir / "toxicity_through_time_model.png",
        label_order=MODEL_ORDER,
        palette=offset_palette,
    )

    toxicity_through_time_plot(
        df=df,
        groupby_col="strategy",
        graph_output_path=graph_dir / "toxicity_through_time_strategy.png",
        label_order=STRATEGY_ORDER,
        palette=palette,
    )

    ablation_df = get_toxicity_df(
        main_df_path=ablation_path,
        toxicity_df_path=toxicity_ratings_dir / "ablation.csv",
    )
    ablation_raw = pd.read_csv(ablation_path)

    # Keep only rows where prompt does NOT contain "strong opinions"
    # aka where instruction prompt is no_instructions.txt
    valid_ids = ablation_raw.loc[
        ~ablation_raw.prompt.str.contains(
            "strong opinions", case=False, na=False
        ),
        "message_id",
    ]

    ablation_df = ablation_df[ablation_df.message_id.isin(valid_ids)]
    ablation_df["instructions"] = "Default"
    df["instructions"] = "Respond-Provoke"

    synthetic_df = pd.concat([df, ablation_df], ignore_index=True)
    toxicity_vs_troll_count(df=synthetic_df, graph_dir=graph_dir)

    human_df = get_toxicity_df(
        main_df_path=human_path,
        toxicity_df_path=toxicity_ratings_dir / "cmv_awry2.csv",
    )
    human_synthetic_df = pd.concat([df, human_df], ignore_index=True)

    toxicity_distribution_comparison(
        df=human_synthetic_df[~human_synthetic_df.is_moderator],
        graph_output_path=graph_dir
        / "toxicity_distribution_human_vs_synthetic.png",
        label_order=["Human", "Synthetic"],
    )


def get_toxicity_df(
    main_df_path: Path, toxicity_df_path: Path
) -> pd.DataFrame:
    df = pd.read_csv(main_df_path)

    toxicity_df = pd.read_csv(toxicity_df_path)
    toxicity_df = toxicity_df.loc[toxicity_df.error.isna()]
    toxicity_df.toxicity = pd.to_numeric(toxicity_df.toxicity)

    full_df = df.merge(right=toxicity_df, how="inner", on="message_id")

    if "prompt" in full_df.columns:
        full_df["is_troll"] = full_df.prompt.str.contains("troll")
        full_df["dataset"] = "Synthetic"
    else:
        full_df["is_troll"] = False
        full_df["model"] = "Human"
        full_df["strategy"] = "Human"
        full_df["message_order"] = full_df.groupby("conv_id").cumcount() + 1
        full_df["dataset"] = "Human"

    full_df = full_df.loc[
        (full_df.model != "hardcoded"),
        [
            "conv_id",
            "message_id",
            "user",
            "is_moderator",
            "toxicity",
            "is_troll",
            "strategy",
            "message_order",
            "model",
            "dataset",
        ],
    ]

    return full_df


def toxicity_by_dimension(
    plot_df: pd.DataFrame, graph_dir: Path, dimension: str
) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=plot_df,
        x="toxicity",
        y=dimension,
        estimator=np.mean,
        errorbar=("ci", 95),
        order=["Facilitator", "Non-troll", "Troll"],
    )

    ax.set_ylabel("")
    ax.set_xlabel("Average toxicity")

    plt.tight_layout()
    tasks.graphs.save_plot(graph_dir / f"{dimension}_mean_toxicity.png")
    plt.close()


def participant_toxicity_regression(
    df: pd.DataFrame, latex_output_dir: Path
) -> None:
    df["message_order_c"] = df["message_order"] - df["message_order"].mean()
    df = df.rename(columns={"toxicity": "Toxicity"})
    model = smf.mixedlm(
        "Toxicity ~ C(strategy, Treatment(reference='No Facilitator')) * message_order_c",
        data=df,
        groups=df["conv_id"],
    )
    result = model.fit()
    print(result.summary())


def moderator_toxicity_regression(df: pd.DataFrame) -> None:
    df["message_order_c"] = df["message_order"] - df["message_order"].mean()
    df = df.rename(columns={"toxicity": "Toxicity"})
    model = smf.mixedlm(
        "Toxicity ~ C(strategy, Treatment(reference='No Instructions')) * message_order_c",
        data=df,
        groups=df["conv_id"],
    )
    result = model.fit()
    print(result.summary())


def facilitation_response_regression(df: pd.DataFrame) -> None:
    """
    For each facilitator comment, find the next comment by the same user
    (pattern: user A -> facilitator -> user A) and use that as the outcome.
    Runs a mixed-effects model of strategy + is_troll on post-facilitation toxicity.
    """
    df = df.sort_values(["conv_id", "message_order"]).reset_index(drop=True)

    records = []
    for conv_id, conv_df in df.groupby("conv_id"):
        conv_df = conv_df.reset_index(drop=True)
        moderator_mask = conv_df["is_moderator"]

        for mod_idx in conv_df.index[moderator_mask]:
            # Find the comment immediately before the facilitator
            if mod_idx == 0:
                continue
            pre_mod = conv_df.loc[mod_idx - 1]
            if pre_mod["is_moderator"]:
                continue

            target_user = pre_mod["user"]

            # Find the next comment by that same user after the facilitator
            after = conv_df.loc[mod_idx + 1:]
            same_user_after = after[after["user"] == target_user]
            if same_user_after.empty:
                continue

            post_mod = same_user_after.iloc[0]

            records.append({
                "conv_id": conv_id,
                "user": target_user,
                "is_troll": pre_mod["is_troll"],
                "strategy": conv_df.loc[mod_idx, "strategy"],
                "pre_toxicity": pre_mod["toxicity"],
                "post_toxicity": post_mod["toxicity"],
            })

    response_df = pd.DataFrame(records)
    print(f"\nFacilitation response pairs found: {len(response_df)}")
    print(f"Strategy breakdown:\n{response_df['strategy'].value_counts()}\n")

    model = smf.mixedlm(
        "post_toxicity ~ C(strategy, Treatment(reference='No Instructions')) * C(is_troll)",
        data=response_df,
        groups=response_df["conv_id"],
    )
    result = model.fit()
    print(result.summary())


# Significance stars
def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def toxicity_vs_troll_count(df: pd.DataFrame, graph_dir: Path) -> None:
    non_troll_df = df.loc[(~df.is_troll) & (~df.is_moderator)]

    avg_toxicity = (
        non_troll_df.groupby(["conv_id", "instructions"])["toxicity"]
        .mean()
        .rename("avg_non_troll_toxicity")
    )

    troll_counts = (
        df.loc[df.is_troll]
        .groupby(["conv_id", "instructions"])["user"]
        .nunique()
        .rename("n_distinct_trolls")
    )

    plot_df = (
        pd.concat([avg_toxicity, troll_counts], axis=1).fillna(0).reset_index()
    )

    plot_df["troll_bin"] = plot_df["n_distinct_trolls"].clip(upper=4)
    plot_df["troll_bin"] = (
        plot_df["troll_bin"].astype(int).astype(str).replace({"4": "4+"})
    )

    plot_toxicity_vs_trolls(plot_df, graph_dir)


def plot_toxicity_vs_trolls(plot_df: pd.DataFrame, graph_dir: Path) -> None:
    plt.figure(figsize=(7, 4))

    ax = sns.pointplot(
        data=plot_df,
        x="troll_bin",
        order=["0", "1", "2", "3", "4+"],
        y="avg_non_troll_toxicity",
        hue="instructions",
        estimator=np.mean,
        errorbar=("ci", 95),
        hue_order=["Respond-Provoke", "Default"],
        markers=["*", "^"],
    )

    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_title("Toxicity of non-troll users")
    ax.set_xlabel("#Active troll users")
    ax.set_ylabel("Avg. toxicity")
    ax.legend(title="")

    plt.tight_layout()
    tasks.graphs.save_plot(graph_dir / "toxicity_vs_troll_count.png")
    plt.close()


def toxicity_distribution_comparison(
    df: pd.DataFrame,
    graph_output_path: Path,
    label_order: list[str],
) -> None:
    """
    KDE distribution plot comparing toxicity scores across human vs synthetic
    datasets.
    Styled after plot_dataset_diversity with hatched fills and custom legend.
    """

    ax = sns.kdeplot(
        data=df,
        x="toxicity",
        hue="dataset",
        hue_order=label_order,
        fill=True,
        common_norm=False,
        multiple="layer",
    )

    palette_colors = sns.color_palette(n_colors=len(label_order))
    color_to_hatch = {
        tuple(color): hatch
        for color, hatch in zip(
            palette_colors, tasks.graphs.HATCHES[: len(label_order)]
        )
    }

    poly_collections = [
        c
        for c in ax.collections
        if isinstance(c, matplotlib.collections.PolyCollection)
    ]
    for poly in poly_collections:
        face_color = tuple(poly.get_facecolor()[0][:3])
        hatch = color_to_hatch.get(face_color)
        if hatch:
            poly.set_hatch(hatch)

    new_handles = [
        matplotlib.patches.Patch(
            facecolor=palette_colors[i],
            hatch=tasks.graphs.HATCHES[i],
            label=label,
        )
        for i, label in enumerate(label_order)
    ]
    ax.legend(handles=new_handles, title="")

    ax.set_xlabel("Toxicity")
    ax.set_ylabel("Density")

    plt.tight_layout()
    tasks.graphs.save_plot(graph_output_path)
    plt.close()


def toxicity_through_time_plot(
    df: pd.DataFrame,
    groupby_col: str,
    graph_output_path: Path,
    label_order: list[str],
    palette: list[str],
) -> None:
    plot_df = df[~df.is_moderator].copy()
    plot_df = plot_df.drop_duplicates(subset=["conv_id", "message_id"])
    plot_df = plot_df.sort_values(["conv_id", "message_order"])

    plot_df["turn_index"] = plot_df.groupby("conv_id").cumcount() + 1
    plot_df["cum_avg_toxicity"] = (
        plot_df.groupby("conv_id")["toxicity"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=plot_df,
        x="turn_index",
        y="cum_avg_toxicity",
        hue=groupby_col,
        hue_order=label_order,
        palette=palette,
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        style=groupby_col,  # required for markers to actually render
        style_order=label_order,
        markersize=10,
    )

    # the errorbar argument turns the x-axis into 0-index for some reason
    plt.xticks(sorted(plot_df["turn_index"].unique()))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    plt.xlabel("#Comments (start -> end)")
    plt.ylabel("Cumulative average toxicity")
    plt.legend(title="", loc="upper right")
    plt.tight_layout()

    tasks.graphs.save_plot(graph_output_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run toxicity analysis and generate plots."
    )

    parser.add_argument(
        "--vmd-path",
        type=str,
        required=True,
        help="Path to the main VMD dataset CSV",
    )

    parser.add_argument(
        "--ablation-path",
        type=str,
        required=True,
        help="Path to the ablation dataset CSV",
    )

    parser.add_argument(
        "--human-path",
        type=str,
        required=True,
        help="Path to the human CMV dataset CSV",
    )

    parser.add_argument(
        "--toxicity-rating-dir",
        type=str,
        required=True,
        help="Directory holding toxicity ratings",
    )

    parser.add_argument(
        "--graph-output-dir",
        type=str,
        required=True,
        help="Graph output directory",
    )

    parser.add_argument(
        "--stats-output-dir",
        type=str,
        required=True,
        help="Directory for LaTeX regression output",
    )

    args = parser.parse_args()

    main(
        vmd_path=Path(args.vmd_path),
        ablation_path=Path(args.ablation_path),
        human_path=Path(args.human_path),
        toxicity_ratings_dir=Path(args.toxicity_rating_dir),
        graph_dir=Path(args.graph_output_dir),
        latex_output_dir=Path(args.stats_output_dir),
    )
