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


def feature_test_mnist(corruption_level=0.0, epochs=10000):
    # load train data
    mnist = fetch_mldata('MNIST original', data_home='.')
    y = mnist.target
    target_names = np.unique(y)
    # get feature & create input
    X = []
    for data in mnist.data:
        img = data.reshape((28, 28))
        # fd, f_img = hog(img, orientations=8, pixels_per_cell=(16, 16),
        #                     cells_per_block=(1, 1), visualise=True)
        # f_img = cv2.Canny(img, 50, 200)
        img[img < 122] = 0
        img[img > 255] = 255
        f_img = img
        X.append(f_img.reshape(-1,))
    X = np.array(X)

    # standardize
    X = X.astype(np.float64)
    X /= X.max()

    clf = nn.NN(ni=X.shape[1], nh=100, no=len(target_names), corruption_level=0.0)

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

    print classification_report(y_true=y_test, y_pred=y_pred)
    print confusion_matrix(y_true=y_test, y_pred=y_pred)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print score

    return score


if __name__ == '__main__':
    feature_test_mnist()