import itertools
import json
import pickle
import sys

import numpy as np
from features import extract_features
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRanker

MODEL_PATH = 'model'
QUERIES_PATH = 'data/training_queries.json'


def get_raw_data(query_path):
    data = json.load(open(query_path))

    output = []
    for q, candidates in data.items():
        has_correct = False
        features = extract_features(q, candidates)
        for (candidate, label), feature in zip(data[q].items(), features):
            if label:
                has_correct = True
            output.append({'q1': q, 'q2': candidate, 'x': feature, 'y': 1 if label else 0})
        if not has_correct:
            print(f'{q} has no correct label {data[q]}')
    return output


def vectorize_data(data, vectorizer=None):
    typo_weight = 10
    data = sorted(data, key=lambda item: item['q1'])
    group_sizes = [
        len(list(group)) for key, group in
        itertools.groupby(data, key=lambda item: item['q1'])]
    group_w = [
        1 if any(row['y'] == 1 and row['q1'] == row['q2'] for row in group) else typo_weight for key, group in
        itertools.groupby(data, key=lambda item: item['q1'])]
    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=False)
        x = vectorizer.fit_transform(item['x'] for item in data)
    else:
        x = vectorizer.transform(item['x'] for item in data)
    y = np.array([item['y'] for item in data])
    return group_sizes, x, y, group_w, vectorizer


def save_model(model_path, model, vectorizer):
    with open(model_path, mode='wb') as writer:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer
        }, writer)


def load_model(model_path):
    with open(model_path, mode='rb') as reader:
        return pickle.load(reader)


def train(train_path: str = QUERIES_PATH, model_path: str = MODEL_PATH):
    data = get_raw_data(train_path)
    g_train, x_train, y_train, w_train, vectorizer = vectorize_data(data)
    params = {'objective': 'rank:pairwise', 'n_estimators': 100, 'silent': False, 'verbose_eval': True, 'missing': np.nan}
    print(f'model params {params}')
    model = XGBRanker(**params)
    # model.fit(x_train, y_train, g_train, sample_weight=w_train, verbose=True)
    model.fit(x_train, y_train, g_train, sample_weight=w_train, eval_set=[(x_train, y_train)], eval_group=[g_train], sample_weight_eval_set=[w_train], eval_metric='ndcg', verbose=True)
    save_model(model_path, model, vectorizer)
    print('model trained and saved')


def evaluate(test_path: str = QUERIES_PATH, model_path: str = MODEL_PATH):
    # TODO: implement and print out the following evaluation metrics
    # accuracy of the top ranked candidate
    # accuracy of the typo queries subset and the non-typo queries subset
    # For example, you have 2 queries as follows:
    # {"Q1": {"Q11": 0, "Q12": 1}, "Q2": {"Q2": 1, "Q21": 0}]
    # your ranking is as follows
    # {"Q1": ["Q12", "Q11", "Q1"], "Q2": ["Q21", "Q2"]}
    # and the accuracy is 0.5


    item = load_model(model_path)
    model = item['model']  # this is your autocorrect model
    vectorizer = item['vectorizer']  # this is your feature vectorizer
    queries = json.load(open(test_path))  # this is your test query set

    for query, candidate_labels in queries.items():
        # eg. query = "addidas"
        # candidates = {"adidas": 1, "addidid": 0, etc...}
        candidates = list(set(candidate_labels.keys()) | {query})
        features = extract_features(query, candidates)
        x = vectorizer.transform(features)
        h = model.predict(x)  # np.array with shape (len(candidates), )

def predict(model_path: str= MODEL_PATH):
    item = load_model(model_path)
    model = item['model']
    vectorizer = item['vectorizer']
    while True:
        print('query=', end='')
        query = input()
        print('candidate=', end='')
        candidate = input()
        features = extract_features(query, [candidate])
        pred = model.predict(vectorizer.transform(features))
        print(f'pred={pred}')
        print()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'predict':
        predict()
    elif sys.argv[1] == 'evaluate':
        evaluate()
