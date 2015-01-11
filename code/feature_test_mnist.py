#!/usr/bin/env python
# -*- coding: utf-8 -*-
# feature_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

from __future__ import print_function
import sys

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

from autoencoder import AutoEncoder



def feature_test_mnist(verbose=True):
    print("... loading date")
    # load train data
    mnist = fetch_mldata('MNIST original')
    X_origin = mnist.data
    y = mnist.target
    target_names = np.unique(y)
    print("--- done")

    print("... random sampling & standardizing")
    p = np.random.randint(0, len(X_origin), 10000)
    X_origin = X_origin[p]
    y = y[p]
    # standardize
    X_origin = X_origin.astype(np.float64)
    X_origin /= X_origin.max()
    # print X_origin.min(), X_origin.mean(), X_origin.max(), X_origin.shape
    print("--- done")

    print("... encoding with denoising auto-encoder")
    # get feature & create input
    import theano.tensor as T
    ae = AutoEncoder(X=X_origin,
                     hidden_size=500,
                     activation_function=T.nnet.sigmoid,
                     output_function=T.nnet.sigmoid)
    ae.train(n_epochs=20, mini_batch_size=20)
    X = ae.get_hidden(data=X_origin)[0]
    print("--- done")

    # get classifier
    clf = nn.NN(ni=X.shape[1],
                nh=int(0.1*X.shape[1]),
                no=len(target_names),
                learning_rate=0.3,
                inertia_rate=0.0,
                corruption_level=0.0,
                noise_level=0.0,
                epochs=10000)

    # split data to train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # convert train data to 1-of-k expression
    label_train = LabelBinarizer().fit_transform(y_train)
    label_test = LabelBinarizer().fit_transform(y_test)

    clf.fit(X=X_train,
            y_train=label_train,
            )

    y_pred = clf.predict(X_test)

    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    if verbose is True:
        print(classification_report(y_true=y_test, y_pred=y_pred))
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))
        print(score)

    return score, clf


if __name__ == '__main__':
    feature_test_mnist(verbose=True)
