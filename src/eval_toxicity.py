import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf

import tasks.graphs


def main(main_output_dir: Path, toxicity_ratings_dir: Path, graph_dir: Path):
    tasks.graphs.seaborn_setup()
    df = get_toxicity_df(
        main_df_path=main_output_dir / "vmd.csv",
        toxicity_df_path=toxicity_ratings_dir / "vmd.csv",
    )

    df["role"] = np.where(
        df.is_moderator,
        "Moderator",
        np.where(df.is_troll, "Troll", "Non-troll user"),
    )

    MODEL_ORDER = tasks.graphs.get_sorted_labels(df, "model")
    STRATEGY_ORDER = tasks.graphs.get_sorted_labels(df, "strategy")

    toxicity_overall(df[~df.is_moderator], graph_dir)
    toxicity_by_dimension(df, graph_dir, "role")
    toxicity_by_dimension(df, graph_dir, "strategy")
    toxicity_by_dimension(df, graph_dir, "model")
    toxicity_regression(df[~df.is_moderator], graph_dir=graph_dir)

    palette = tasks.graphs.COLORBLIND_PALETTE
    # shift colors left by 1 so first color is skipped
    # given that humans dont exist in this subset
    offset_palette = palette[1:] + palette[:1]
    toxicity_through_time_plot(
        df=df,
        groupby_col="model",
        graph_output_path=graph_dir / "toxicity_through_time_model.png",
        label_order=MODEL_ORDER,
        palette=offset_palette
    )

    toxicity_through_time_plot(
        df=df,
        groupby_col="strategy",
        graph_output_path=graph_dir / "toxicity_through_time_strategy.png",
        label_order=STRATEGY_ORDER,
        palette=palette
    )

    ablation_df = get_toxicity_df(
        main_df_path=main_output_dir / "ablation.csv",
        toxicity_df_path=toxicity_ratings_dir / "ablation.csv",
    )

    # Keep only rows where prompt does NOT contain "strong opinions"
    # aka where instruction prompt is no_instructions.txt
    ablation_raw = pd.read_csv(main_output_dir / "ablation.csv")
    valid_ids = ablation_raw.loc[
        ~ablation_raw.prompt.str.contains(
            "strong opinions", case=False, na=False
        ),
        "message_id",
    ]

    ablation_df = ablation_df[ablation_df.message_id.isin(valid_ids)]
    ablation_df["dataset"] = "Basic Prompt"
    df["dataset"] = "Provocation-Reactive Prompt"

    full_df = pd.concat([df, ablation_df], ignore_index=True)
    toxicity_vs_troll_count(df=full_df, graph_dir=graph_dir)


def get_toxicity_df(
    main_df_path: Path, toxicity_df_path: Path
) -> pd.DataFrame:
    df = pd.read_csv(main_df_path)

    toxicity_df = pd.read_csv(toxicity_df_path)
    toxicity_df = toxicity_df.loc[toxicity_df.error.isna()]
    toxicity_df.toxicity = pd.to_numeric(toxicity_df.toxicity)

    full_df = df.merge(right=toxicity_df, how="inner", on="message_id")
    full_df["is_troll"] = full_df.prompt.str.contains("troll")

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
        ],
    ]

    return full_df


def toxicity_overall(df: pd.DataFrame, graph_dir: Path) -> None:
    sns.histplot(df.toxicity)
    tasks.graphs.save_plot(graph_dir / "overall_toxicity.png")
    plt.close()


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
    )

    ax.set_ylabel("")
    ax.set_xlabel("Average toxicity")

    plt.tight_layout()
    tasks.graphs.save_plot(graph_dir / f"{dimension}_mean_toxicity.png")
    plt.close()


def toxicity_regression(df: pd.DataFrame, graph_dir: Path) -> None:
    df["message_order_c"] = df["message_order"] - df["message_order"].mean()

    model = smf.mixedlm(
        "toxicity ~ C(strategy, Treatment(reference='No Facilitator')) * message_order_c",
        data=df,
        groups=df["conv_id"],
    )
    result = model.fit()

    latex = result.summary().as_latex()

    replacements = {
        "C(strategy, Treatment(reference='No Facilitator'))[T.Constr. Comms]": "CC",
        "C(strategy, Treatment(reference='No Facilitator'))[T.E-Rulemaking]": "ER",
        "C(strategy, Treatment(reference='No Facilitator'))[T.No Instructions]": "NoMod",
        "C(strategy, Treatment(reference='No Facilitator'))[T.Constr. Comms]:message_order": "CC × order",
        "C(strategy, Treatment(reference='No Facilitator'))[T.E-Rulemaking]:message_order": "ER × order",
        "C(strategy, Treatment(reference='No Facilitator'))[T.No Instructions]:message_order": "NoMod × order",
    }

    for old, new in replacements.items():
        latex = latex.replace(old, new)
    with open(graph_dir / "toxicity_regression.tex", "w") as f:
        f.write(latex)


def toxicity_vs_troll_count(df: pd.DataFrame, graph_dir: Path) -> None:
    non_troll_df = df.loc[(~df.is_troll) & (~df.is_moderator)]

    avg_toxicity = (
        non_troll_df.groupby(["conv_id", "dataset"])["toxicity"]
        .mean()
        .rename("avg_non_troll_toxicity")
    )

    troll_counts = (
        df.loc[df.is_troll]
        .groupby(["conv_id", "dataset"])["user"]
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
        hue="dataset",
        estimator=np.mean,
        errorbar=("ci", 95),
    )

    ax.set_title("Toxicity of non-troll users by instruction prompt")
    ax.set_xlabel("#Active troll users")
    ax.set_ylabel("Avg. toxicity")
    ax.legend(title="")

    plt.tight_layout()
    tasks.graphs.save_plot(graph_dir / "toxicity_vs_troll_count.png")
    plt.close()


def toxicity_through_time_plot(
    df: pd.DataFrame,
    groupby_col: str,
    graph_output_path: Path,
    label_order: list[str],
    palette: list[str]
) -> None:
    # --- Step 1: copy and filter out moderators ---
    plot_df = df[~df.is_moderator].copy()

    # --- Step 2: remove duplicate messages per conversation ---
    plot_df = plot_df.drop_duplicates(subset=["conv_id", "message_id"])

    # --- Step 3: sort messages by conversation and message order ---
    plot_df = plot_df.sort_values(["conv_id", "message_order"])

    # --- Step 4: reconstruct turn index within each conversation ---
    plot_df["turn_index"] = plot_df.groupby("conv_id").cumcount() + 1

    # --- Step 5: compute cumulative average toxicity per conversation ---
    plot_df["cum_avg_toxicity"] = (
        plot_df.groupby("conv_id")["toxicity"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    # --- Step 6: seaborn lineplot with errorbar ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=plot_df,
        x="turn_index",
        y="cum_avg_toxicity",
        hue=groupby_col,
        hue_order=label_order,
        marker="o",
        palette=palette,
        errorbar=("ci", 95),
    )

    plt.xlabel("#User messages in conversation")
    plt.ylabel("Cumulative average toxicity")
    plt.legend(title="")
    plt.tight_layout()

    # --- Step 7: save the plot ---
    tasks.graphs.save_plot(graph_output_path)
    plt.close()


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
        graph_dir=Path(args.graph_output_dir),
    )
