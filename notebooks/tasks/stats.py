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
    
