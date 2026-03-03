import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import tasks.graphs


def main(input_csv_path: Path, graph_output_dir: Path, latex_output_dir: Path):
    tasks.graphs.seaborn_setup()

    df = pd.read_csv(input_csv_path)
    df = df[df.model != "hardcoded"]

    MODEL_ORDER = tasks.graphs.get_sorted_labels(df, "model")
    STRATEGY_ORDER = tasks.graphs.get_sorted_labels(df, "strategy")

    intervention_through_time_plot(
        df=df, groupby_col="model", label_order=MODEL_ORDER
    )
    tasks.graphs.save_plot(graph_output_dir / "intervention_count_model.png")
    plt.close()

    intervention_through_time_plot(
        df=df, groupby_col="strategy", label_order=STRATEGY_ORDER
    )
    tasks.graphs.save_plot(
        graph_output_dir / "intervention_count_strategy.png"
    )
    plt.close()

    participation_df = participation_summary(df)
    participation_df = participation_df.rename(
        {"participation_rate": "Participation Rate"}
    )
    participation_df.to_latex(
        latex_output_dir / "participation.tex",
        caption="Participation rate by model in synthetic discussions.",
        label="tab:participation",
        position="ht",
        float_format="%.3f",
    )


def participation_summary(df: pd.DataFrame) -> pd.DataFrame:
    non_mod_df = df[~df.is_moderator].copy()

    # per-model participation
    per_model = non_mod_df.groupby("model")["message"].apply(
        _participation_rate
    )

    # overall participation
    overall = pd.Series({"Overall": _participation_rate(non_mod_df.message)})

    # combine → models + overall
    combined = pd.concat([per_model, overall])

    # return transposed single-row dataframe
    return combined.to_frame(name="participation_rate").T


def _participation_rate(messages: pd.Series) -> float:
    # messages where the only content is quotes, since some models
    # accidentally add more than two quotes when not intervening
    empty_mask = messages.fillna("").astype(str).str.fullmatch(r'[\s"]*')
    return 1 - (empty_mask.sum() / len(messages)) if len(messages) else 0.0


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
        "--graph-output-dir",
        type=str,
        help="Graph output directory",
    )
    parser.add_argument(
        "--stats-output-dir",
        type=str,
        help="Graph output directory",
    )
    args = parser.parse_args()
    main(
        input_csv_path=Path(args.input_csv),
        graph_output_dir=Path(args.graph_output_dir),
        latex_output_dir=Path(args.stats_output_dir),
    )
