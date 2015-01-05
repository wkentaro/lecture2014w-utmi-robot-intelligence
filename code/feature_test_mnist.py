#!/usr/bin/env python
# -*- coding: utf-8 -*-
# feature_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np
import cv2

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        confusion_matrix
        )

import nn

from dA import dA


def get_dA_hidden(X, corruption_level):
    # load train data
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    # construct dA
    ni = X.shape[1]
    nh = int(0.06*ni)
    da = dA(input=X, n_visible=ni, n_hidden=nh)
    # train dA
    for epoch in xrange(training_epochs):
        da.train(lr=learning_rate, corruption_level=corruption_level)
        cost = da.negative_log_likelihood(corruption_level=corruption_level)
        print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.95
    # get hidden layer
    X_hidden = da.get_hidden_values(input=X)
    return X_hidden


def feature_test_mnist(
        corruption_level=0.0,
        nh=100,
        epochs=10000,
        verbose=False,
        ):
    # load train data
    mnist = fetch_mldata('MNIST original')
    X_origin = mnist.data
    y = mnist.target
    target_names = np.unique(y)

    # get feature & create input
    X = get_dA_hidden(X=X_origin, corruption_level=0.0)

    # standardize
    X = X.astype(np.float64)
    X /= X.max()

    # get classifier
    if nh <= 0:
        raise ValueError('nh should be >0')
    elif nh > 1:
        pass
    else:
        nh = int(nh * X.shape[1])
    clf = nn.NN(ni=X.shape[1],
                nh=nh,
                no=len(target_names),
                corruption_level=corruption_level)

    # split data to train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # convert train data to 1-of-k expression
    label_train = LabelBinarizer().fit_transform(y_train)
    label_test = LabelBinarizer().fit_transform(y_test)

    clf.fit(X_train, label_train, epochs=epochs)

    y_pred = np.zeros(len(X_test))
    for i, xt in enumerate(X_test):
        o = clf.predict(xt)
        y_pred[i] = np.argmax(o)
    # print y_pred

    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    if verbose is True:
        print classification_report(y_true=y_test, y_pred=y_pred)
        print confusion_matrix(y_true=y_test, y_pred=y_pred)
        print score

    return score, clf


if __name__ == '__main__':
    feature_test_mnist()