import itertools
import numpy as np
from rouge_score import rouge_scorer


def pairwise_rougel_similarity(comments: list[str]) -> float:
    """
    Return the average of the pairwise ROUGE-L similarity for all comments in a discussion.
    :param: comments: the comments of the discussion
    :return: a similarity score from 0 (no similarities) to 1 (identical)
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    scores = []
    for c1, c2 in itertools.combinations(comments, 2):
        scores.append(scorer.score(c1.lower(), c2.lower())["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else np.nan



# code from John Pavlopoulos https://github.com/ipavlopoulos/ndfu/blob/main/src/__init__.py
def ndfu(input_data, histogram_input=True, normalised=True):
    """
    The normalized Distance From Unimodality measure
    :param: input_data: the data, by default the relative frequencies of ratings
    :param: histogram_input: False to compute rel. frequencies (ratings as input)
    :return: the DFU score
    """
    hist = input_data if histogram_input else _to_hist(input_data, bins_num=5)
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
    if normalised:
        return max_diff / max_value
    return max_diff


def _to_hist(scores, bins_num=3, normed=True):
    """
    Creating a normalised histogram
    :param: scores: the ratings (not necessarily discrete)
    :param: bins_num: the number of bins to create
    :param: normed: whether to normalise or not, by default true
    :return: the histogram
    """
    # not keeping the values order when bins are not created
    counts, bins = np.histogram(a=scores, bins=bins_num)
    counts_normed = counts / counts.sum()
    return counts_normed if normed else counts

