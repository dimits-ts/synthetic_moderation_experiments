import itertools
import multiprocessing

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer


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
    return float(np.mean(scores)) if scores else np.nan


def discussion_var(
    df: pd.DataFrame,
    discussion_key_col: str,
    comment_key_col: str,
    val_col: str,
) -> pd.Series:
    comment_var_df = (
        df.groupby([discussion_key_col, comment_key_col])[val_col]
        .agg("std")
        .reset_index()
    )
    discussion_var_df = comment_var_df.groupby(discussion_key_col)[
        val_col
    ].agg("mean")
    return discussion_var_df
