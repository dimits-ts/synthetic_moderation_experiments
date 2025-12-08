import itertools
import multiprocessing
from typing import Iterable, Callable

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import scipy.stats
from rouge_score import rouge_scorer


# code adapted from John Pavlopoulos
# https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def ndfu(input_data: Iterable[float], num_bins: int = 5) -> float:
    """The normalized Distance From Unimodality measure
    :param: input_data: a list of annotations, not necessarily discrete
    :raises ValueError: if input_data is empty
    :return: the nDFU score
    """
    # compute DFU
    hist = _to_hist(input_data, num_bins=num_bins)
    max_value = max(hist)
    pos_max = np.where(hist == max_value)[0][0]
    # right search
    max_diff = 0
    for i in range(pos_max, len(hist) - 1):
        diff = hist[i + 1] - hist[i]
        if diff > max_diff:
            max_diff = diff
    for i in range(pos_max, 0, -1):
        diff = hist[i - 1] - hist[i]
        if diff > max_diff:
            max_diff = diff

    # return normalized dfu
    return max_diff / max_value


def rougel_similarity(comments: list[str]) -> list[float]:
    """
    Return the average of the pairwise ROUGE-L similarity for all
    comments in a discussion.
    :param: comments: the list of comments to compute ROUGE-L
     similarities on
    :return: a similarity score from 0 (no similarities) to 1 (identical)
    """

    # 1 thread if no cpu_count found, else leave 1 core for system
    with multiprocessing.Pool(multiprocessing.cpu_count() - 1 or 1) as pool:
        rougel_similarities = list(
            tqdm(
                pool.imap(_compute_pairwise_rougel, comments),
                total=len(comments),
                desc="Computing ROUGE-L similarities",
            )
        )
    return rougel_similarities


def discussion_var(
    df: pd.DataFrame,
    discussion_key_col: str,
    comment_key_col: str,
    val_col: str,
    var_func: Callable[[Iterable[float]], float] = ndfu,
) -> pd.DataFrame:
    comment_var_df = (
        df.groupby([discussion_key_col, comment_key_col])[val_col]
        .agg(ndfu)
        .reset_index()
    )
    return comment_var_df


def polarization_df(df: pd.DataFrame, metric_col: str):
    ndfu_df = df
    ndfu_df["polarization"] = (
        ndfu_df.groupby(["conv_id", "message"])[metric_col]
        .transform(lambda x: ndfu(x))
        .astype(float)
    )
    return ndfu_df


def mean_comp_test(
    df: pd.DataFrame, feature_col: str, score_col: str
) -> float:
    """
    Return the p-value of a means comparison test comparing a
    scores across a given dimension.
    :param: df: the dataframe containing the comments and model information
    :param: feature_col: the column containing the dimension across which the
        difference in means is investigated
    :param score_col: the column containing the scores to be compared
    :return: the p-value given by the test
    """
    groups = [
        df.loc[df[feature_col] == factor, [score_col]]
        for factor in df[feature_col].unique()
    ]

    return scipy.stats.f_oneway(*groups).pvalue[0]


def _compute_pairwise_rougel(comments: list[str]) -> float:
    """
    Return the average of the pairwise ROUGE-L similarity for all comments
    in a discussion.
    :param: comments: the comments of the discussion
    :return: a similarity score from 0 (no similarities) to 1 (identical)
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    scores = []
    for c1, c2 in itertools.combinations(comments, 2):
        scores.append(scorer.score(c1.lower(), c2.lower())["rougeL"].fmeasure)
    return (1 - float(np.mean(scores))) if scores else np.nan


def _to_hist(
    scores: Iterable[float], num_bins: int, normed: bool = True
) -> np.ndarray:
    """Creating a normalised histogram
    :param: scores: the ratings (not necessarily discrete)
    :param: num_bins: the number of bins to create
    :param: normed: whether to normalise the counts or not, by default true
    :return: the histogram
    """
    scores_array = np.array(scores)
    if len(scores_array) == 0:
        raise ValueError("Annotation list can not be empty.")

    # not keeping the values order when bins are not created
    counts, bins = np.histogram(a=scores_array, bins=num_bins)
    counts_normed = counts / counts.sum()
    return counts_normed if normed else counts
