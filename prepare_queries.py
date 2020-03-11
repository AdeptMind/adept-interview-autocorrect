import json
import random
import re
from collections import defaultdict

import numpy as np
from misspell import misspell


def sample_np(items, key, n):
    w = [key(i) for i in items]
    w = np.array(w)
    w = w / np.sum(w)
    rand = np.random.RandomState(0)
    idx = rand.choice(np.arange(len(items)), n, replace=False, p=w)
    return [items[i] for i in idx]


def get_negative_candidates(q1, corrections=[]):
    good = {re.sub(' +', ' ', q) for q in [q1, *corrections]}
    rand = random.Random(0)
    candidates = [misspell(q1, rand) for _ in range(20)]
    return {c for c in candidates if re.sub(' +', ' ', c) not in good}


def get_typo_queries():
    validated_query_pairs = json.load(open('data/typo_queries.json'))
    return {q1: set(q2s.keys()) for q1, q2s, in validated_query_pairs.items()}


def get_no_typo_queries():
    no_typo_q = json.load(open('data/no_typo_queries.json'))
    return no_typo_q.keys()


def assemble_queries():
    training_qs = defaultdict(dict)

    n = 0
    no_typo_q = get_no_typo_queries()
    for q1 in no_typo_q:
        n += 1
        neg = get_negative_candidates(q1)
        training_qs[q1][q1] = 1
        for q2 in neg:
            training_qs[q1][q2] = 0
    print(f'{n} queries from no typo')
    print(f'examples: {list(no_typo_q)[:5]}')

    typo_queries = get_typo_queries()
    n = 0
    for q1, correct_qs in typo_queries.items():
        n += 1
        neg = get_negative_candidates(q1, correct_qs)
        for q2 in correct_qs:
            training_qs[q1][q2] = 1
        for q2 in neg:
            training_qs[q1][q2] = 0
    print(f'{n} queries from typo')
    print(f'examples: {list(typo_queries.keys())[:5]}')

    with open('data/training_queries.json', 'w') as fout:
        json.dump(training_qs, fout)


if __name__ == '__main__':
    assemble_queries()
