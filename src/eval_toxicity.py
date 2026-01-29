import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf

import tasks.constants
import tasks.graphs


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
            "is_moderator",
            "toxicity",
            "is_troll",
            "strategy",
            "message_order"
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
    )

    ax.set_ylabel("Average toxicity")
    ax.set_xlabel("")
    ax.set_title(f"Average toxicity by {dimension}")

    plt.tight_layout()
    tasks.graphs.save_plot(graph_dir / f"{dimension}_mean_toxicity.png")
    plt.close()


def toxicity_regression(df: pd.DataFrame, graph_dir: Path) -> None:
    model = smf.mixedlm(
        "toxicity ~ C(strategy, Treatment(reference='No Instructions')) * message_order",
        data=df,
        groups=df["conv_id"],
    )
    result = model.fit()
    print(result.summary())

    latex = result.summary().as_latex()
    replacements = {
        "C(strategy, Treatment(reference='No Instructions'))[T.Constr. Comms]": "CC",
        "C(strategy, Treatment(reference='No Instructions'))[T.E-Rulemaking]": "ER",
        "C(strategy, Treatment(reference='No Instructions'))[T.Constr. Comms]:message_order": "CC × order",
        "C(strategy, Treatment(reference='No Instructions'))[T.E-Rulemaking]:message_order": "ER × order",
    }

    for old, new in replacements.items():
        latex = latex.replace(old, new)
    with open(graph_dir / "toxicity_regression.tex", "w") as f:
        f.write(latex)


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
    toxicity_by_dimension(df, graph_dir, "role")
    toxicity_by_dimension(df, graph_dir, "strategy")
    toxicity_regression(df[~df.is_moderator], graph_dir=graph_dir)


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
