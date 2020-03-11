# -*- coding: utf-8 -*-
# Natural Language Toolkit: Distance Metrics
#
# Copyright (C) 2001-2019 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Tom Lippincott <tom@cs.columbia.edu>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Distance Metrics.

Compute the distance between two items (usually strings).
As metrics, they must satisfy the following three requirements:

1. d(a, a) = 0
2. d(a, b) >= 0
3. d(a, c) <= d(a, b) + d(b, c)
"""

from collections import Counter

import numpy as np

EDIT_SKIP_S1 = 0
EDIT_ADD_S2 = 1
EDIT_SUBSTITUTE = 2
EDIT_TRANSPOSE = 3


def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _edit_type_init(len1, len2):
    edits = []
    for i in range(len1):
        edits.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        edits[i][0] = EDIT_SKIP_S1
    for j in range(len2):
        edits[0][j] = EDIT_ADD_S2
    return edits


def _edit_dist_step(lev, edits, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    possible_edits = [a, b, c, d]
    argmin = np.argmin(possible_edits)
    lev[i][j] = possible_edits[argmin]
    edits[i][j] = argmin


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to
    substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)
    edits = _edit_type_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(
                lev,
                edits,
                i + 1,
                j + 1,
                s1,
                s2,
                substitution_cost=substitution_cost,
                transpositions=transpositions,
            )

    # backtrack and get edit path
    i = len1
    j = len2
    edits_d = Counter()
    while i >= 1 or j >= 1:
        edit = edits[i][j]
        if edit == EDIT_SUBSTITUTE:
            if lev[i][j] != lev[i - 1][j - 1]:
                edits_d[f'substitute_{s1[i - 1]}_{s2[j - 1]}'] += 1
            i = i - 1
            j = j - 1
        elif edit == EDIT_SKIP_S1:
            edits_d[f'remove_{s1[i - 1]}'] += 1
            i = i - 1
        elif edit == EDIT_ADD_S2:
            edits_d[f'insert_{s2[j - 1]}'] += 1
            j = j - 1
        elif edit == EDIT_TRANSPOSE:
            edits_d[f'transpose_{s1[i - 1]}_{s2[j - 1]}'] += 1
            i = i - 2
            j = j - 2

    return lev[len1][len2], edits_d
