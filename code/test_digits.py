#!/usr/bin/env python
#-*- coding: utf-8 -*-
# test_digits.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np

from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import nn


def test_digits():
    # load train data
    digits = load_digits()
    X = digits.data
    y = digits.target
    target_names = digits.target_names

    # standardize
    X /= X.max()

    clf = nn.NN(ni=X.shape[1], nh=2*X.shape[1], no=len(target_names), corruption_level=0.25)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # convert train data to 1-of-k expression
    label_train = LabelBinarizer().fit_transform(y_train)
    label_test = LabelBinarizer().fit_transform(y_test)

    clf.fit(X_train, label_train, epochs=10000, learning_rate=0.4, inertia_rate=0.3)

    y_pred = np.zeros(len(X_test))
    for i, xt in enumerate(X_test):
        o = clf.predict(xt)
        y_pred[i] = np.argmax(o)
    # print y_pred

    print classification_report(y_true=y_test, y_pred=y_pred)
    print accuracy_score(y_true=y_test, y_pred=y_pred)
    print confusion_matrix(y_true=y_test, y_pred=y_pred)


if __name__ == '__main__':
    test_digits()