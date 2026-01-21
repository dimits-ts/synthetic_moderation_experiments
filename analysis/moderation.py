import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import tasks.constants
import tasks.graphs


def get_intervention_df(df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    df = df.copy()

    # Keep only conversations with at least one moderator message
    moderated_convs = df[df.is_moderator].conv_id.unique()
    df = df.loc[df.conv_id.isin(moderated_convs), :]
    df = df.loc[df.model != "hardcoded", :]
    mod_df = df[df.is_moderator]

    mod_total = mod_df.groupby("conv_id").size()
    empty_msg = mod_df.message.astype(str).str.strip() == '""'
    mod_empty = mod_df[empty_msg].groupby("conv_id").size()
    mod_empty = mod_empty.reindex(mod_total.index, fill_value=0)

    #   (empty moderator replies) / (all moderator replies) * 100
    intervention_pct = (mod_empty / mod_total) * 100
    groups = (
        df.groupby("conv_id")[groupby_col].first().reindex(mod_total.index)
    )

    return pd.DataFrame(
        {
            "conv_id": mod_total.index,
            groupby_col: groups.values,
            "intervention_pct": intervention_pct.values,
        }
    ).reset_index(drop=True)


def intervention_plot(intervention_df: pd.DataFrame, groupby_col: str) -> None:
    # Create bins 0–10, 10–20, ..., 90–100
    bins = np.arange(0, 110, 10)
    intervention_df["intervention_bin"] = pd.cut(
        intervention_df["intervention_pct"],
        bins=bins,
        right=False,
        include_lowest=True,
    )

    # Count rows in each (bin × variant)
    grouped = (
        intervention_df.groupby(["intervention_bin", groupby_col])
        .size()
        .unstack(fill_value=0)
    )
    print(grouped)

    # Prepare figure
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(grouped))
    bottom = np.zeros(len(grouped))

    colors = sns.color_palette("colorblind", n_colors=20)
    hatches = tasks.constants.HATCHES

    # Plot stacked bars
    for i, variant in enumerate(grouped.columns):
        counts = grouped[variant].values
        ax.bar(
            x,
            counts,
            bottom=bottom,
            label=variant,
            color=colors[i % len(colors)],
            edgecolor="black",
            hatch=hatches[i % len(hatches)],
        )
        bottom += counts

    # X-tick labels (e.g., "0–10%", "10–20%")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{int(iv.left)}-{int(iv.right)}%" for iv in grouped.index],
        rotation=45,
    )

    ax.set_xlabel("Percentage of unmoderated comments")
    ax.set_ylabel("Number of discussions")
    ax.set_title(
        "LLM facilitators almost always intervene in synthetic discussions"
    )
    ax.legend(title="Strategy")

    plt.tight_layout()


def moderation_analysis(
    df: pd.DataFrame, groupby_col: str, graph_output_path: Path
) -> None:
    mod_df = get_intervention_df(df, groupby_col=groupby_col)
    intervention_plot(
        intervention_df=mod_df,
        groupby_col=groupby_col,
    )
    tasks.graphs.save_plot(graph_output_path)
    plt.close()


def main(input_csv_path: Path, output_dir: Path):
    tasks.graphs.seaborn_setup()
    df = pd.read_csv(input_csv_path)
    moderation_analysis(
        df=df,
        groupby_col="model",
        graph_output_path=output_dir / "intervention_count.png",
    )
    moderation_analysis(
        df=df,
        groupby_col="tag_2",
        graph_output_path=output_dir / "intervention_count.png",
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
