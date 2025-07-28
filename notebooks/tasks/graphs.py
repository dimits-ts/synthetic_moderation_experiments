from pathlib import Path
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import scikit_posthocs as sp

from . import stats

hatches = ["/", "|", "\\", "-", "+", "x", "o", "O", ".", "*"]


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path: The full path (including filename)
        where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {path.resolve()}")


def comment_len_plot(
    df: pd.DataFrame, length_col: str, feature_col: str, hue_order: list
) -> None:
    ax = sns.displot(
        df,
        x=length_col,
        hue=feature_col,
        hue_order=hue_order,
        stat="density",
        kde=False,
        multiple="layer",
        common_norm=False,  # normalize observation counts by feature_col
    )
    plt.xlabel("Comment Length (#words)")
    # move legend inside plot
    sns.move_legend(ax, loc="center right", bbox_to_anchor=(0.7, 0.5))


def difference_histogram(df, feature="Toxicity", bins=20, figsize=(6, 5)):
    """
    Plots the difference in normalized histograms for a specified feature
    between trolls_exist=True and trolls_exist=False, grouped by instructions.
    """

    # Define histogram bins
    bin_edges = np.histogram_bin_edges(df[feature], bins=bins)

    # Sort instruction groups in reverse order (top will be plotted last)
    instruction_values = sorted(df["instructions"].unique())

    # Get colorblind palette in *same order* as instructions (top to bottom)
    palette = sns.color_palette("colorblind", n_colors=len(instruction_values))

    # Build legend patches in advance (top to bottom)
    legend_patches = [
        matplotlib.patches.Patch(
            facecolor=palette[i],
            edgecolor="black",
            label=instruction.capitalize(),
            hatch=hatches[i % len(hatches)],
        )
        for i, instruction in enumerate(instruction_values)
    ]

    plt.figure(figsize=figsize)

    # Plot groups so that last one is plotted on top (visually highest)
    for idx, (instruction, color) in enumerate(
        zip(instruction_values, palette)
    ):
        trolls_true = df[
            (df["trolls_exist"]) & (df["instructions"] == instruction)
        ][feature]
        trolls_false = df[
            (~df["trolls_exist"]) & (df["instructions"] == instruction)
        ][feature]

        hist_true, _ = np.histogram(trolls_true, bins=bin_edges, density=True)
        hist_false, _ = np.histogram(
            trolls_false, bins=bin_edges, density=True
        )
        hist_diff = hist_true - hist_false

        # Stack top-to-bottom by INVERTING vertical offset: top bar = higher y
        y_positions = (
            bin_edges[:-1] + (len(instruction_values) - idx - 1) * 0.15
        )

        plt.barh(
            y_positions,
            hist_diff,
            height=np.diff(bin_edges) * 2 / len(instruction_values),
            color=color,
            hatch=hatches[idx % len(hatches)],
            edgecolor="black",
        )

    # Vertical reference line
    plt.axvline(0, color="red", linestyle="--")

    # Labels and title
    plt.ylabel(f"{feature} level", fontsize=16)
    plt.xlabel("Normalized difference in #annotations", fontsize=16)
    plt.title(
        "Trolls affect discussions differently depending on\nparticipant instructions",
        fontsize=15,
        weight="bold",
    )

    # Legend: order now matches visual top-to-bottom
    plt.legend(
        handles=legend_patches,
        loc="upper left",
        title="Prompt",
        fontsize=13,
        title_fontsize=14,
        frameon=False,
    )

    # X-axis side labels
    xmin, xmax = plt.xlim()
    y_pos = bin_edges[0] - np.diff(bin_edges).max()
    plt.text(
        xmin, y_pos, "Fewer comments", ha="left", va="center", fontsize=13
    )
    plt.text(
        xmax, y_pos, "More comments", ha="right", va="center", fontsize=13
    )

    plt.tight_layout()


def rougel_plot(
    df: pd.DataFrame,
    rougel_col: str,
    feature_col: str,
    hue_order: list,
    palette: str = "colorblind",
    title: str = "Diversity Distribution by Group",
) -> None:
    plt.figure(figsize=(9, 4.5))
    
    colors = sns.color_palette(palette, n_colors=len(hue_order))

    bin_edges = np.histogram_bin_edges(df[rougel_col], bins=30, range=(0.6, 1.0))
    bin_width = bin_edges[1] - bin_edges[0]

    for i, group in enumerate(hue_order):
        subset = df[df[feature_col] == group][rougel_col]
        counts, _ = np.histogram(subset, bins=bin_edges, density=True)

        for j, count in enumerate(counts):
            plt.bar(
                bin_edges[j],
                count,
                width=bin_width * 0.9,
                align='edge',
                color=colors[i],
                edgecolor='black',
                alpha=0.6,
                label=group if j == 0 else "",  # only add legend once per group
                hatch=hatches[i % len(hatches)],
            )

    plt.xlabel("Diversity", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.xlim(0.6, 1.0)
    plt.title(title, fontsize=16)

    plt.legend(
        title=feature_col.replace("_", " ").title(),
        title_fontsize=14,
        fontsize=12,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )

    plt.tight_layout()


def posthoc_heatmap(
    df: pd.DataFrame,
    val_col: str,
    group_col: str,
    show_labels: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> None:
    """
    Generate a heatmap visualizing correlation (or other) values along with
    p-value significance.

    This function produces a heatmap where the lower triangle of the matrix
    contains correlation values from `value_df`.
    These values are annotated with asterisks based
    on the significance levels of corresponding p-values from `pvalue_df`.
    The heatmap can be saved to a file if a filename is specified.

    :param df: the data
    :type df: pd.DataFrame
    :param val_col: The column containing the quantity to be measured
    :type val_col: str
    :param group_col: The column containing the groups
    :type group_col: str
    :param show_labels: Whether to display axis labels on the heatmap,
        defaults to False.
    :type show_labels: bool, optional
    :param vmin: Minimum value for color mapping, if specified
    :type vmin: float | None, optional
    :param vmax: Maximum value for color mapping, if specified
    :type vmax: float | None, optional
    :param ax: The matplotlib axes object where the heatmap will be drawn
    :type ax: matplotlib.axes.Axes | None, optional
    """
    pvalues = sp.posthoc_ttest(
        df, val_col=val_col, group_col=group_col, p_adjust="holm"
    )
    diff_values = _pairwise_diffs(df, group_col=group_col, value_col=val_col)
    _pvalue_heatmap(
        value_df=diff_values,
        pvalue_df=pvalues,
        show_labels=show_labels,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    plt.ylabel("")
    plt.xlabel("")
    plt.title(f"{val_col} by facilitation strategy")


def plot_metric_barplot(
    df: pd.DataFrame,
    group_by_col: str,
    group_by_col_label: str,
    metric: str,
    yticks: np.array,
) -> None:
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=df,
        x=group_by_col,
        hue=group_by_col,
        y=metric,
        errorbar="sd",
        dodge=False,
    )
    plt.xticks(rotation=45)
    plt.ylabel(metric.capitalize())
    plt.xlabel("")
    plt.yticks(yticks)
    plt.title(f"Impact of {group_by_col_label} on {metric}")
    plt.tight_layout()


def plot_timeseries(
    df: pd.DataFrame, y_col: str, hue_col: str, hue_col_label: str
) -> None:
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=df,
        x="message_order",
        y=y_col,
        hue=hue_col,
        lw=1,
        alpha=0.6,
    )

    plt.title(
        "Average (all comments from all annotators)"
        f"{y_col.capitalize()} by {hue_col_label}"
    )
    plt.xlabel("Discussion Length (# messages)")
    plt.ylabel(f"Average {y_col.capitalize()}")
    plt.xticks(rotation=45)

    plt.legend(title=hue_col_label)
    plt.tight_layout()


def trolls_plot(df: pd.DataFrame, title: str, val_col: str) -> None:
    ax = sns.displot(
        data=df.rename(columns={"trolls_exist": "Trolls in Discussion"}),
        x=val_col,
        hue="Trolls in Discussion",
        common_norm=False,
        stat="density",
        multiple="dodge",
        bins=10,
    )
    plt.title(title, fontsize=18)
    sns.move_legend(ax, loc="center right", bbox_to_anchor=(0.7, 0.5))


def disagreement_plot(
    var_with_sdb: pd.Series, var_no_sdb: pd.Series, title: str, stat_col: str
) -> None:
    sdb_toxicity_var = var_with_sdb.reset_index()
    sdb_toxicity_var["annotator"] = "With SDB"
    no_sdb_toxicity_var = var_no_sdb.reset_index()
    no_sdb_toxicity_var["annotator"] = "No SDB"
    merged_df = pd.concat(
        [sdb_toxicity_var, no_sdb_toxicity_var], ignore_index=True
    )
    # I can not find how to remove legend title because displot returns
    # a facet grid for some reason
    merged_df = merged_df.rename(columns={"annotator": "Annotator SDB"})

    plot = sns.histplot(
        data=merged_df,
        x=stat_col,
        hue="Annotator SDB",
        common_norm=False,
        multiple="dodge",
        bins=10,
    )
    plt.title(title)
    plt.xlim(0, 1)
    plt.xlabel("nDFU")
    plot.get_legend().set_title(None)
    # sns.move_legend(plot, loc="center right", bbox_to_anchor=(0.68, 0.5))


def polarization_plot(df, metric_col: str):
    ndfu_df = stats.polarization_df(df, metric_col)
    ax = sns.boxplot(
        ndfu_df,
        y="polarization",
        x=metric_col,
        hue=metric_col,
        palette="flare",
    )
    ax.set_title(f"Annotator Polarization vs. {metric_col}")
    ax.set_xlabel(metric_col)
    ax.set_ylabel("nDFU")
    ax.legend(title=metric_col, loc="upper left")


# ======== posthoc_dunn_heatmap ========


def _pvalue_heatmap(
    value_df: pd.DataFrame,
    pvalue_df: pd.DataFrame,
    show_labels: bool,
    vmin: float | None,
    vmax: float | None,
    ax: matplotlib.axes.Axes | None,
) -> None:
    """
    Generate a heatmap visualizing correlation (or other) values along with
     p-value significance.

    This function produces a heatmap where the lower triangle of the matrix
    contains correlation values from `value_df`.
    These values are annotated with asterisks based
    on the significance levels of corresponding p-values from `pvalue_df`.
    The heatmap can be saved to a file if a filename is specified.

    :param value_df: DataFrame containing the correlation or other values to
        be visualized.
    :type value_df: pd.DataFrame
    :param pvalue_df: DataFrame containing p-values corresponding to the
        values in `value_df`.
    :type pvalue_df: pd.DataFrame
    :param show_labels: Whether to display axis labels on the heatmap,
        defaults to False.
    :type show_labels: bool, optional
    :param vmin: Minimum value for color mapping, if specified
    :type vmin: float | None, optional
    :param vmax: Maximum value for color mapping, if specified
    :type vmax: float | None, optional
    :param ax: The matplotlib axes object where the heatmap will be drawn
    :type ax: matplotlib.axes.Axes | None
    """

    # Format the value_df with asterisks based on pvalue_df
    formatted_values = _format_with_asterisks(value_df, pvalue_df)

    # Define tick labels
    ticklabels = value_df.columns if show_labels else "auto"

    # Create the heatmap
    ax = sns.heatmap(
        value_df,
        annot=formatted_values,
        fmt="",  # This allows us to use strings with asterisks
        cmap="RdBu_r",
        # mask=_upper_tri_masking(value_df),
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cbar_kws={"label": "Mean Difference"},
        annot_kws={"fontsize": 14},
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=12)


def _pairwise_diffs(
    df: pd.DataFrame, group_col: str, value_col: str
) -> pd.DataFrame:
    """
    Calculate pairwise differences in mean values between groups and pivot the
    result into an MxM matrix.

    :param df: The input DataFrame containing the data.
    :type df: pd.DataFrame
    :param group_col: The column to group by in order to calculate mean values.
    :type group_col: str
    :param value_col: The column name containing the values for which pairwise
        differences will be calculated.
    :type value_col: str
    :return: An MxM DataFrame with groups as rows and columns,
        and mean differences as values.
    :rtype: pd.DataFrame
    """
    # Calculate mean values per group
    group_means = df.groupby(group_col)[value_col].mean()

    # Prepare a dictionary to collect results
    results = {}

    # Generate all possible pairs of groups
    for (group1, mean1), (group2, mean2) in itertools.combinations(
        group_means.items(), 2
    ):
        diff = mean1 - mean2
        # Store differences in both directions for a symmetric matrix
        results[(group1, group2)] = diff
        results[(group2, group1)] = -diff

    # Create a DataFrame with multi-index from the results dictionary
    result_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["mean_diff"]
    )
    result_df.index = pd.MultiIndex.from_tuples(
        result_df.index, names=[f"{group_col}_1", f"{group_col}_2"]
    )

    # Pivot the DataFrame to create an MxM matrix
    matrix_df = result_df.unstack(level=1).fillna(0)
    matrix_df.columns = matrix_df.columns.droplevel(
        0
    )  # Remove the extra column level

    return matrix_df


# ======== toxicity bar plot ========


# code from https://stackoverflow.com/questions/47314754/
# how-to-get-triangle-upper-matrix-without-the-diagonal-using-numpy
def _upper_tri_masking(array: np.ndarray) -> np.ndarray:
    """
    Generate a mask for the upper triangular of a NxN matrix,
    without the main diagonal

    :param array: the NxN matrix
    :type array: np.array
    :return: the mask
    :rtype: np.array
    """
    m = array.shape[0]
    r = np.arange(m)
    mask = r[:, None] <= r
    return mask


def _format_with_asterisks(
    value_df: pd.DataFrame, pvalue_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Format the values in the value_df with asterisks based on p-value
    significance levels

    :param value_df: DataFrame containing the values to display
    :param pvalue_df: DataFrame containing the p-values
    :return: DataFrame with values formatted with asterisks
    """
    formatted_df = value_df.copy().astype(str)
    for i in range(value_df.shape[0]):
        for j in range(value_df.shape[1]):
            value = value_df.iloc[i, j]
            pvalue = pvalue_df.iloc[i, j]
            if pd.notnull(
                pvalue
            ):  # Only apply formatting if pvalue is not NaN
                if pvalue < 0.001:
                    num_asterisks = 3
                elif pvalue < 0.01:
                    num_asterisks = 2
                elif pvalue < 0.05:
                    num_asterisks = 1
                else:
                    num_asterisks = 0
            else:  # if NaN
                num_asterisks = 0

            formatted_df.iloc[i, j] = f"{value:.3f}\n{num_asterisks * '*'}"

    return formatted_df
