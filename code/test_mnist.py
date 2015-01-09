#!/usr/bin/env python
#-*- coding: utf-8 -*-
# test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
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
                corruption_level=corruption_level)

    # split data to train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # convert train data to 1-of-k expression
    label_train = LabelBinarizer().fit_transform(y_train)
    label_test = LabelBinarizer().fit_transform(y_test)

    # train the model
    clf.fit(X_train,
            label_train,
            epochs=epochs,
            learning_rate=learning_rate,
            inertia_rate=inertia_rate)

    # predict with the trained model
    y_pred = np.zeros(len(X_test))
    for i, xt in enumerate(X_test):
        # add noise to the x
        p = np.random.binomial(n=1, p=1-noise_level, size=len(xt))
        xt[p==0] = np.random.random(len(xt))[p==0]
        # get the prediction
        o = clf.predict(xt)
        y_pred[i] = np.argmax(o)
    # get score
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    # output report
    if verbose is True:
        print classification_report(y_true=y_test, y_pred=y_pred)
        print confusion_matrix(y_true=y_test, y_pred=y_pred)
        print score

    return score, clf


if __name__ == '__main__':
    test_mnist(verbose=True)