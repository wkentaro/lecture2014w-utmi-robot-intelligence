#!/usr/bin/env python
#-*- coding: utf-8 -*-
# test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        confusion_matrix
        )

import nn


def load_data(standardize=True):
    # load train data
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    y = mnist.target
    target_names = np.unique(y)
    # standardize
    if standardize is True:
        X = X.astype(np.float64)
        X /= X.max()
    return X, y, target_names


def test_mnist(corruption_level=0.0,
               noise_level=0.0,
               learning_rate=0.2,
               inertia_rate=0.0,
               nh=0.1,
               epochs=40000,
               verbose=False,
               ):
    # load data
    X, y, target_names = load_data(standardize=True)

    # get classifier
    clf = nn.NN(ni=X.shape[1],
                nh=int(nh*X.shape[1]),
                no=len(target_names),
                learning_rate=learning_rate,
                inertia_rate=inertia_rate,
                corruption_level=corruption_level,
                epochs=epochs)

    # cross validation
    skf = StratifiedKFold(y, n_folds=3)
    scores = np.zeros(len(skf))
    for i, (train_index, test_index) in enumerate(skf):
        clf.fit(X[train_index], y[train_index])
        score = clf.score(X[test_index], y[test_index])
        scores[i] = score

    return scores.mean(), clf


if __name__ == '__main__':
    test_mnist(verbose=True)