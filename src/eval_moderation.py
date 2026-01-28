import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import tasks.constants
import tasks.graphs


def build_moderation_summary(
    df: pd.DataFrame, groupby_col: str
) -> pd.DataFrame:
    """
    Returns one row per conversation with:
    - conv_id
    - moderator model / strategy (groupby_col)
    - intervention percentage (empty moderator replies)
    """
    df = df.copy()

    # Keep only conversations with at least one moderator message
    mod_df = df[df.is_moderator]
    moderated_convs = mod_df.conv_id.unique()
    mod_df = mod_df[mod_df.conv_id.isin(moderated_convs)]

    # Drop hardcoded moderators if needed
    mod_df = mod_df[mod_df.model != "hardcoded"]

    # Total moderator messages per conversation
    mod_total = mod_df.groupby("conv_id").size()

    # Empty moderator messages per conversation
    empty_msg = mod_df.message.astype(str).str.strip() == '""'
    mod_empty = (
        mod_df[empty_msg]
        .groupby("conv_id")
        .size()
        .reindex(mod_total.index, fill_value=0)
    )

    intervention_pct = (mod_empty / mod_total) * 100

    # Moderator model / strategy (must be unique per conversation)
    moderator_group = (
        mod_df.groupby("conv_id")[groupby_col].first().reindex(mod_total.index)
    )

    return pd.DataFrame(
        {
            "conv_id": mod_total.index,
            groupby_col: moderator_group.values,
            "intervention_pct": intervention_pct.values,
        }
    ).reset_index(drop=True)


def intervention_plot(summary_df: pd.DataFrame, groupby_col: str) -> None:
    # Create bins 0–10, 10–20, ..., 90–100
    bins = np.arange(0, 110, 10)
    summary_df["intervention_bin"] = pd.cut(
        summary_df["intervention_pct"],
        bins=bins,
        right=False,
        include_lowest=True,
    )

    # Count discussions per (bin × group)
    grouped = (
        summary_df.groupby(["intervention_bin", groupby_col])
        .size()
        .unstack(fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(grouped))
    bottom = np.zeros(len(grouped))

    colors = tasks.graphs.COLORBLIND_PALETTE
    hatches = tasks.constants.HATCHES

    for i, group in enumerate(grouped.columns):
        counts = grouped[group].values
        ax.bar(
            x,
            counts,
            bottom=bottom,
            label=group,
            color=colors[i % len(colors)],
            edgecolor="black",
            hatch=hatches[i % len(hatches)],
        )
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(iv.left)}-{int(iv.right)}%" for iv in grouped.index],
        rotation=45,
    )

    ax.set_xlabel("Percentage of unmoderated comments")
    ax.set_ylabel("Number of discussions")
    ax.legend()

    plt.tight_layout()


def moderation_analysis(
    df: pd.DataFrame, groupby_col: str, graph_output_path: Path
) -> None:
    summary_df = build_moderation_summary(
        df=df,
        groupby_col=groupby_col,
    )

    intervention_plot(
        summary_df=summary_df,
        groupby_col=groupby_col,
    )

    tasks.graphs.save_plot(graph_output_path)
    plt.close()


def main(input_csv_path: Path, output_dir: Path):
    tasks.graphs.seaborn_setup()

    df = pd.read_csv(input_csv_path)
    variant_name_dict = {
        "constructive": "Facilitation",
        "erulemaking.txt": "Moderation",
        "vanilla.txt": "No instructions",
    }
    df.tag_2 = df.tag_2.replace(variant_name_dict)

    moderation_analysis(
        df=df,
        groupby_col="model",
        graph_output_path=output_dir / "intervention_count_model.png",
    )
    moderation_analysis(
        df=df,
        groupby_col="tag_2",
        graph_output_path=output_dir / "intervention_count_strategy.png",
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
