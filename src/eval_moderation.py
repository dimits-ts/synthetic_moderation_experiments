import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import tasks.graphs


def main(input_csv_path: Path, output_dir: Path):
    tasks.graphs.seaborn_setup()

    df = pd.read_csv(input_csv_path)

    MODEL_ORDER = tasks.graphs.get_sorted_labels(df, "model")
    STRATEGY_ORDER = tasks.graphs.get_sorted_labels(df, "strategy")
    
    intervention_through_time_plot(
        df=df, groupby_col="model", label_order=MODEL_ORDER
    )
    tasks.graphs.save_plot(output_dir / "intervention_count_model.png")
    plt.close()

    intervention_through_time_plot(
        df=df, groupby_col="strategy", label_order=STRATEGY_ORDER
    )
    tasks.graphs.save_plot(output_dir / "intervention_count_strategy.png")
    plt.close()

    print(
        f"Participants responded {participant_participation(df) * 100:.2f}% "
        "of the time"
    )


def participant_participation(df: pd.DataFrame) -> float:
    non_mod_df = df[~df.is_moderator]
    empty_messages = non_mod_df.message.astype(str).str.strip() == '""'
    return len(empty_messages) / len(non_mod_df)


def build_moderation_summary(
    df: pd.DataFrame,
    groupby_col: str,
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


def intervention_through_time_plot(
    df: pd.DataFrame, groupby_col: str, label_order: list[str]
) -> None:
    df = df.copy()

    # Keep only moderator messages
    df = df[df.is_moderator]
    df = df[df.model != "hardcoded"]

    df["turn_index"] = df.groupby("conv_id").cumcount() + 1

    # Detect intervention (empty moderator reply)
    df["is_intervention"] = df.message.astype(str).str.strip() != '""'

    # Compute cumulative intervention rate per conversation
    df["cum_interventions"] = df.groupby("conv_id")["is_intervention"].cumsum()
    df["cum_messages"] = df.groupby("conv_id").cumcount() + 1
    df["cum_intervention_pct"] = (
        df["cum_interventions"] / df["cum_messages"] * 100
    )

    # Average across conversations for each model and turn index
    summary = (
        df.groupby([groupby_col, "turn_index"])["cum_intervention_pct"]
        .mean()
        .reset_index()
    )

    # build consistent color map from label_order
    base_colors = tasks.graphs.COLORBLIND_PALETTE
    palette = {
        label: base_colors[i % len(base_colors)]
        for i, label in enumerate(label_order)
    }

    markers = ["o", "s", "D", "^", "v", "P", "X"]

    fig, ax = plt.subplots(figsize=(12, 6))

    # iterate in sorted label order
    for i, label in enumerate(label_order):
        group_df = summary[summary[groupby_col] == label]
        if group_df.empty:
            continue  # skip labels not present in this plot

        ax.plot(
            group_df["turn_index"],
            group_df["cum_intervention_pct"],
            label=label,
            color=palette[label],
            marker=markers[i % len(markers)],
        )

    ax.set_xlabel("#Comments (start -> end)")
    ax.set_ylabel("% Interventions")

    # legend already sorted because plotting order is sorted
    ax.legend(title=groupby_col)

    plt.tight_layout()


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
