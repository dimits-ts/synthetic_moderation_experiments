# Synthetic discussion generation experiments
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

import itertools
from typing import Iterable, Callable

import numpy as np
import pandas as pd
import scipy.stats
from rouge_score import rouge_scorer


def rougel_similarity(comments: list[str]) -> list[float]:
    """
    Return the average of the pairwise ROUGE-L similarity for all
    comments in a discussion.
    :param: comments: the list of comments to compute ROUGE-L
     similarities on
    :return: a similarity score from 0 (no similarities) to 1 (identical)
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"])
    scores = []
    for c1, c2 in itertools.combinations(comments, 2):
        scores.append(scorer.score(c1.lower(), c2.lower())["rougeL"].fmeasure)
    return (1 - float(np.mean(scores))) if scores else np.nan

