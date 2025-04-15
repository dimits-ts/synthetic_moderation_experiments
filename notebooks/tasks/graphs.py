from pathlib import Path
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import scikit_posthocs as sp


def save_plot(path: Path) -> None:
    """
    Saves a plot to the specified filepath.

    :param path: The full path (including filename)
        where the plot will be saved.
    :type path: pathlib.Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    print(f"Figure saved to {path.resolve()}")


def comment_len_plot(df: pd.DataFrame, feature_col: str) -> None:
    len_df = df.copy()
    len_df["comment_length"] = len_df.message.apply(lambda x: len(x.split()))
    len_df = len_df.loc[
        (len_df.comment_length > 0) & (len_df.comment_length < 600),
        ["message_id", "comment_length", feature_col],
    ]
    sns.displot(
        len_df,
        x="comment_length",
        hue=feature_col,
        stat="density",
        kde=True,
        common_norm=False,  # normalize observation counts by feature_col
    )
    plt.xlabel("Comment Length (#words)")


def toxicity_barplot(df: pd.DataFrame, ax: matplotlib.axes.Axes):
    """
    Create a bar plot displaying the mean toxicity scores for different
    conversation variants, grouped by annotator prompts.

    This function generates a horizontal bar plot where the x-axis
    represents toxicity
    scores, and the y-axis represents different conversation variants.
    The bars are colored by annotator demographic.
    An additional vertical red line is plotted at a
    toxicity score of 3 to mark a threshold.

    :param df: The input DataFrame containing the toxicity scores,
        conversation variants, and annotator prompts.
    :type df: pd.DataFrame
    :param ax: The matplotlib axes object where the bar plot will be drawn.
    :type ax: matplotlib.axes.Axes
    :return: None

    :example:
        >>> fig, example_ax = plt.subplots()
        >>> toxicity_barplot(df, example_ax)
        >>> plt.show()
    """

    sns.barplot(
        data=df,
        y="conv_variant",
        x="toxicity",
        hue="annotator_prompt",
        estimator=np.mean,
        ax=ax,
    )
    ax.axvline(x=3, color="r")
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim(0, 5)
    ax.legend(
        title="Annotator Demographic",
        fontsize="6",
        title_fontsize="6.5",
        loc="upper right",
    )


def rougel_plot(rougel_sim: pd.Series, feature: pd.Series):
    sns.displot(
        x=rougel_sim,
        hue=feature,
        stat="density",
        kde=True,
        common_norm=False,  # normalize observation counts by feature_col
    )
    plt.xlabel("ROUGE Similarity")
    plt.ylabel("Density")


def posthoc_dunn_heatmap(
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


def plot_metrics_barplots(
    df: pd.DataFrame,
    group_by_col: str,
    group_by_col_label: str,
    metrics: list[str],
    yticks_list: list[np.array],
) -> None:
    fig, axes = plt.subplots(1, len(metrics))
    fig.set_size_inches(12, 6)

    for ax, metric, yticks in zip(axes, metrics, yticks_list):

        sns.barplot(
            data=df,
            x=group_by_col,
            hue=group_by_col,
            y=metric,
            errorbar="sd",
            legend=False,
            ax=ax,
        )
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel(metric.capitalize())
        ax.set_yticks(yticks)

    fig.suptitle(f"Impact of {group_by_col_label} on Discussions")
    fig.supxlabel(group_by_col_label)
    fig.supylabel("Annotation Scores")
    fig.tight_layout()


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
    sns.heatmap(
        np.tril(value_df),
        annot=np.tril(formatted_values),
        fmt="",  # This allows us to use strings with asterisks
        cmap="icefire",
        mask=_upper_tri_masking(value_df),
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cbar_kws={"label": "Mean Difference"},
        annot_kws={"fontsize": 8},
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )


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

            formatted_df.iloc[i, j] = f"{value:.3f}{num_asterisks * '*'}"

    return formatted_df
