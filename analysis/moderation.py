import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import tasks.constants


def get_intervention_df(df: pd.DataFrame) -> pd.DataFrame:
    moderated_convs = df[df.is_moderator].conv_id.unique()
    filtered_df = df[df.conv_id.isin(moderated_convs)]
    filtered_df = filtered_df[filtered_df.message.str.strip() != '""']

    # Count total messages per conversation
    total_msgs = filtered_df.groupby("conv_id").size()

    # Count moderator messages per conversation
    mod_msgs = (
        filtered_df[filtered_df["is_moderator"]].groupby("conv_id").size()
    )

    # Compute intervention percentage
    intervention_pct = (mod_msgs / total_msgs).fillna(0) * 2 * 100

    # Get conv_variant per conv_id (assuming it is unique per conv)
    conv_variants = filtered_df.groupby("conv_id")["conv_variant"].first()

    # Create intervention_df
    intervention_df = pd.DataFrame(
        {
            "conv_id": intervention_pct.index,
            "conv_variant": conv_variants,
            "intervention_pct": intervention_pct.values,
        }
    ).reset_index(drop=True)
    return intervention_df


def intervention_df(intervention_df: pd.DataFrame, colors, hatches) -> None:
    # (0-10%, 10-20%, ..., 90-100%)
    bins = np.arange(0, 110, 10)

    intervention_df["intervention_bin"] = pd.cut(
        intervention_df.intervention_pct,
        bins=bins,
        right=False,
        include_lowest=True,
    )

    variants = intervention_df.conv_variant.unique()
    colors = sns.color_palette("colorblind", 8).as_hex()

    # Group and count: rows = bins, columns = variants
    grouped = (
        intervention_df.groupby(["intervention_bin", "conv_variant"])
        .size()
        .unstack(fill_value=0)
    )
    grouped = grouped.reindex(
        index=pd.IntervalIndex.from_breaks(bins, closed="left"), fill_value=0
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(grouped))
    bottom = np.zeros(len(grouped))

    for i, variant in enumerate(variants):
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
        bottom += counts  # stack bars on top

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{int(interval.left)}-{int(interval.right)}%"
            for interval in grouped.index
        ],
        rotation=45,
    )
    ax.set_xlabel("Percentage of moderated comments")
    ax.set_ylabel("Number of Discussions")
    ax.set_title(
        "LLM facilitators almost always intervene in synthetic discussions"
    )
    ax.legend(title="Strategy")
    plt.tight_layout()


def main(input_csv_path: Path, output_dir: Path):
    sns.set(
        style="whitegrid",
        font_scale=1.5,
        font="Times New Roman",
        context="paper",
        palette="colorblind",
    )
    df = pd.read_csv(input_csv_path)
    mod_df = get_intervention_df(df)
    print(mod_df.intervention_pct.describe())
    print(mod_df)

    strategies = intervention_df.conv_variant.unique()
    colorblind_palette = sns.color_palette(
        "colorblind", n_colors=len(strategies)
    )
    intervention_df(
        intervention_df=mod_df,
        hatches=tasks.constants.HATCHES,
        colors=colorblind_palette.as_hex(),
    )
    tasks.graphs.save_plot(output_dir / "intervention_count.png")
    plt.close()


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
