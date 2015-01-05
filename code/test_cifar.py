#!/usr/bin/env python
#-*- coding: utf-8 -*-
# test_cifar.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import os
import numpy as np
import collections
import cPickle

from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        confusion_matrix
        )

import nn


def load_pkl(filename):
    with open(filename, 'rb') as f:
        return cPickle.load(f)


def load_cifar(data_dir_path='../data/cifar-10-batches-py'):
    X = np.zeros((10000*5, 3072))
    y = np.zeros(10000*5)
    for i in xrange(5):
        filename = os.path.join(data_dir_path,
                'data_batch_{0}'.format(i+1))
        data = load_pkl(filename)
        X[10000*i:10000*(i+1)] = data['data']
        y[10000*i:10000*(i+1)] = data['labels']

    MLdata = collections.namedtuple('MLdata', 'data target')
    return MLdata(X, y)


def test_cifar(
        corruption_level=0.0,
        epochs=10000,
        verbose=False
        ):
    # load train data
    cifar = load_cifar()
    X = cifar.data
    y = cifar.target
    target_names = np.unique(y)

    # standardize
    X = X.astype(np.float64)
    X /= X.max()

    if verbose is True:
        print("Layer size: first: {0}, second: {1}, final: {2}".format(X.shape[1], 100, len(target_names)))
    clf = nn.NN(ni=X.shape[1], nh=100, no=len(target_names), corruption_level=corruption_level)

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

    return score


if __name__ == '__main__':
    test_cifar()