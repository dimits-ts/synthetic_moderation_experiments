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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf

import tasks.graphs


def main(
    main_output_dir: Path,
    toxicity_ratings_dir: Path,
    graph_dir: Path,
    latex_output_dir: Path,
):
    tasks.graphs.seaborn_setup()
    df = get_toxicity_df(
        main_df_path=main_output_dir / "vmd.csv",
        toxicity_df_path=toxicity_ratings_dir / "vmd.csv",
    )

    df["role"] = np.where(
        df.is_moderator,
        "Facilitator",
        np.where(df.is_troll, "Troll", "Non-troll"),
    )

    MODEL_ORDER = tasks.graphs.get_sorted_labels(df, "model")
    STRATEGY_ORDER = tasks.graphs.get_sorted_labels(df, "strategy")

    toxicity_overall(df[~df.is_moderator], graph_dir)
    toxicity_by_dimension(df, graph_dir, "role")
    toxicity_by_dimension(df, graph_dir, "strategy")
    toxicity_by_dimension(df, graph_dir, "model")
    toxicity_regression(
        df[~df.is_moderator], latex_output_dir=latex_output_dir
    )

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
    ablation_df["dataset"] = "Minimal"
    df["dataset"] = "Responsive"

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
        order=["Facilitator", "Non-troll", "Troll"]
    )

    ax.set_ylabel("")
    ax.set_xlabel("Average toxicity")

    plt.tight_layout()
    tasks.graphs.save_plot(graph_dir / f"{dimension}_mean_toxicity.png")
    plt.close()


def toxicity_regression(df: pd.DataFrame, latex_output_dir: Path) -> None:
    df["message_order_c"] = df["message_order"] - df["message_order"].mean()
    df = df.rename(columns={"toxicity": "Toxicity"})
    model = smf.mixedlm(
        "Toxicity ~ C(strategy, Treatment(reference='No Facilitator')) * message_order_c",
        data=df,
        groups=df["conv_id"],
    )
    result = model.fit()

    # --- Extract coefficients, SEs, and p-values manually ---
    params = result.fe_params
    bse = result.bse_fe
    pvalues = result.pvalues[params.index]

    # Random effects variance
    re_var = (
        result.cov_re.iloc[0, 0] if result.cov_re is not None else float("nan")
    )

    # Significance stars
    def stars(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    # Human-readable label map
    label_map = {
        "Intercept": "Constant",
        "C(strategy, Treatment(reference='No Facilitator'))[T.Constr. Comms]": "Constructive Communications",
        "C(strategy, Treatment(reference='No Facilitator'))[T.E-Rulemaking]": "Moderation Guidelines",
        "C(strategy, Treatment(reference='No Facilitator'))[T.No Instructions]": "Minimal Instructions",
        "message_order_c": "Discussion Turn",
        "C(strategy, Treatment(reference='No Facilitator'))[T.Constr. Comms]:message_order_c": "Constructive Communications $\\times$ Turn",
        "C(strategy, Treatment(reference='No Facilitator'))[T.E-Rulemaking]:message_order_c": "Moderation Guidelines $\\times$ Turn",
        "C(strategy, Treatment(reference='No Facilitator'))[T.No Instructions]:message_order_c": "Minimal Instructions $\\times$ Turn",
    }

    # --- Build LaTeX table manually ---
    rows = []
    for raw_name, coef in params.items():
        se = bse[raw_name]
        p = pvalues[raw_name]
        label = label_map.get(raw_name, raw_name)
        s = stars(p)
        rows.append(
            f"    {label} & ${coef:8.3f}^{{{s}}}$ \\\\\n"
            f"    & $({se:.3f})$ \\\\"
        )

    body = "\n".join(rows)

    n_obs = int(result.nobs)
    n_groups = result.model.n_groups
    log_lik = (
        f"{result.llf:.3f}"
        if hasattr(result, "llf") and result.llf is not None
        else "---"
    )

    latex = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Mixed-Effects Model: Predictors of Message Toxicity}\n"
        "\\label{tab:toxicity_regression}\n"
        "\\begin{tabular}{lc}\n"
        "\\hline\\hline\n"
        " & Toxicity \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        f"    \\textit{{Random Effects}} & \\\\\n"
        f"    \\quad Group Variance & ${re_var:.4f}$ \\\\\n"
        "\\hline\n"
        f"    Observations & {n_obs} \\\\\n"
        f"    Groups & {n_groups} \\\\\n"
        f"    Log-Likelihood & {log_lik} \\\\\n"
        "\\hline\\hline\n"
        "\\multicolumn{2}{l}{\\footnotesize Standard errors in parentheses.} \\\\\n"
        "\\multicolumn{2}{l}{\\footnotesize $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$} \\\\\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    with open(latex_output_dir / "toxicity_regression.tex", "w") as f:
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
        hue_order=["Responsive", "Minimal"],
        markers=["*", "^"]
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
    sns.lineplot(
        data=plot_df,
        x="turn_index",
        y="cum_avg_toxicity",
        hue=groupby_col,
        hue_order=label_order,
        palette=palette,
        errorbar=("ci", 95),
        markers=True,
        dashes=False,
        style=groupby_col,   # required for markers to actually render
        style_order=label_order,
        markersize=10,
    )

    plt.xlabel("#Comments (start -> end)")
    plt.ylabel("Cumulative average toxicity")
    plt.legend(title="", loc="upper right")
    plt.tight_layout()

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
    parser.add_argument(
        "--stats-output-dir",
        type=str,
        help="Graph output directory",
    )
    args = parser.parse_args()
    main(
        main_output_dir=Path(args.main_output_dir),
        toxicity_ratings_dir=Path(args.toxicity_rating_dir),
        graph_dir=Path(args.graph_output_dir),
        latex_output_dir=Path(args.stats_output_dir),
    )
