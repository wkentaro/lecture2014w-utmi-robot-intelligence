#!/usr/bin/env python
# -*- coding: utf-8 -*-
# nn.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dsigmoid(y):
    return y * (1. - y)


class NN(object):
    def __init__(
            self,
            ni,
            nh,
            no,
            learning_rate=0.3,
            inertia_rate=0.0,
            corruption_level=0.0,
            epochs=10000
            ):
        # params
        self.learning_rate = learning_rate
        self.inertia_rate = inertia_rate
        self.corruption_level = corruption_level
        self.epochs = epochs

        # activations for nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh + 1  # +1 for bias node
        self.no = no

        # create weights
        # input layer  -> hidden layer
        self.wi = np.random.uniform(-1., 1., (self.nh, self.ni))
        self.dwi_old = 0.
        # hidden layer -> output layer
        self.wo = np.random.uniform(-1., 1., (self.no, self.nh))
        self.dwo_old = 0.

    def fit(self, X, y_train):
        """Update the weights in nn using training datasets"""
        # add bias unit to input data
        X = np.hstack([np.ones((len(X), 1)), X])
        y_train = np.array(y_train)
        y_train_binarized = LabelBinarizer().fit_transform(y_train)

        # training
        for k in xrange(self.epochs):
            # randomly select training data
            i = np.random.randint(len(X))
            x = X[i]

            # add noise to the input
            p = np.random.binomial(n=1, p=1 - self.corruption_level,
                                   size=len(x))
            x[p==0] = np.random.random(len(x))[p==0]

            # forward propagation
            z = sigmoid(np.dot(self.wi, x))
            y = sigmoid(np.dot(self.wo, z))

            # compute error in output layer
            delta2 = dsigmoid(y) * (y - y_train_binarized[i])

            # update weight in hidden layer
            # using the error in output layer
            delta1 = dsigmoid(z) * np.dot(self.wo.T, delta2)

            # update hidden layer weight using
            # error in hidden layer
            x = np.atleast_2d(x)
            delta1 = np.atleast_2d(delta1)
            dwi = - self.learning_rate * np.dot(delta1.T, x) + self.inertia_rate * self.dwi_old
            self.wi += dwi
            self.dwi_old = dwi

            # update output layer weight using
            # error in output layer
            z = np.atleast_2d(z)
            delta2 = np.atleast_2d(delta2)
            dwo = - self.learning_rate * np.dot(delta2.T, z) + self.inertia_rate * self.dwo_old
            self.wo += dwo
            self.dwo_old = dwo

    def predict(self, X_test):
        """Predict the solution for test data"""
        # predict with the trained model
        X_test = np.array(X_test)
        y_pred = np.zeros(len(X_test))
        for i, xt in enumerate(X_test):
            # get the model output
            xt = np.insert(xt, 0, 1)  # for bias
            # forward propagation
            z = sigmoid(np.dot(self.wi, xt))
            o = sigmoid(np.dot(self.wo, z))

            if len(o) > 1:
                y_pred[i] = np.argmax(o)
            else:
                y_pred[i] = round(o)

        return y_pred

    def score(self, X_test, y_true):
        y_pred = self.predict(X_test)
        score = accuracy_score(y_true=y_true, y_pred=y_pred)
        return score


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    ni = len(X[0])
    nn = NN(ni=ni, nh=ni, no=1)

    nn.fit(X, y_train)

    y_pred = nn.predict(X)
    print y_train
    print y_pred


if __name__ == '__main__':
    main()
