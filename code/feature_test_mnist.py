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

np.seterr(all='ignore')


def get_dA_hidden(X,
                  corruption_level,
                  learning_rate=0.1,
                  training_epochs=50):
    # # load train data
    # mnist = fetch_mldata('MNIST original')
    # X = mnist.data
    # construct dA
    ni = X.shape[1]
    # nh = int(0.06*ni)
    nh = 100
    da = dA(input=X, n_visible=ni, n_hidden=nh)
    # train dA
    for epoch in xrange(training_epochs):
        da.train(lr=learning_rate, corruption_level=corruption_level)
        # cost = da.negative_log_likelihood(corruption_level=corruption_level)
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        # learning_rate *= 0.95
    # get hidden layer
    X_hidden = da.get_hidden_values(input=X)
    return X_hidden


def feature_test_mnist(verbose=True):
    # load train data
    mnist = fetch_mldata('MNIST original')
    X_origin = mnist.data
    y = mnist.target
    target_names = np.unique(y)

    p = np.random.randint(0, len(X_origin), 10000)
    X_origin = X_origin[p][:10000]
    y = y[p][:10000]
    # standardize
    X_origin = X_origin.astype(np.float64)
    X_origin /= X_origin.max()
    # print X_origin.min(), X_origin.mean(), X_origin.max(), X_origin.shape

    # get feature & create input
    X = get_dA_hidden(X=X_origin, corruption_level=0.1)

    # get classifier
    nh = int(0.16 * X.shape[1])
    clf = nn.NN(ni=X.shape[1],
                nh=nh,
                no=len(target_names),
                corruption_level=0.04)

    # split data to train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # convert train data to 1-of-k expression
    label_train = LabelBinarizer().fit_transform(y_train)
    label_test = LabelBinarizer().fit_transform(y_test)

    clf.fit(X=X_train,
            y_train=label_train,
            learning_rate=0.3,
            inertia_rate=0.24,
            epochs=70000)

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
    feature_test_mnist(verbose=True)
