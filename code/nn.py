#!/usr/bin/env python
# -*- coding: utf-8 -*-
# nn.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dsigmoid(y):
    return y * (1. - y)


class NN(object):
    def __init__(self, ni, nh, no, corruption_level=0.0):
        self.corruption_level = corruption_level

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

    def fit(self, X, y_train, learning_rate=0.4, inertia_rate=0.1, epochs=10000):
        """Update the weights in nn using training datasets"""
        # add bias unit to input data
        X = np.hstack([np.ones((len(X), 1)), X])
        y_train = np.array(y_train)

        # training
        for k in xrange(epochs):
            # print("[#{0:4}]: training".format(k))

            # randomly select training data
            i = np.random.randint(len(X))
            x = X[i]

            # add noise to the input
            p = np.random.binomial(n=1, p=1 - self.corruption_level,
                                   size=len(x))
            rnd_samples = np.where(p == 0)
            for rs in rnd_samples:
                x[rs] = np.random.random()

            # forward propagation
            z = sigmoid(np.dot(self.wi, x))
            y = sigmoid(np.dot(self.wo, z))

            # compute error in output layer
            delta2 = dsigmoid(y) * (y - y_train[i])

            # update weight in hidden layer
            # using the error in output layer
            delta1 = dsigmoid(z) * np.dot(self.wo.T, delta2)

            # update hidden layer weight using
            # error in hidden layer
            x = np.atleast_2d(x)
            delta1 = np.atleast_2d(delta1)
            dwi = - learning_rate * np.dot(delta1.T, x) + inertia_rate * self.dwi_old
            self.wi += dwi
            self.dwi_old = dwi

            # update output layer weight using
            # error in output layer
            z = np.atleast_2d(z)
            delta2 = np.atleast_2d(delta2)
            dwo = - learning_rate * np.dot(delta2.T, z) + inertia_rate * self.dwo_old
            self.wo += dwo
            self.dwo_old = dwo

    def predict(self, x):
        """Predict the solution for test data"""
        x = np.array(x)
        x = np.insert(x, 0, 1)  # for bias
        # forward propagation
        z = sigmoid(np.dot(self.wi, x))
        y = sigmoid(np.dot(self.wo, z))
        return y


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 1, 1, 0])

    ni = len(X[0])
    nn = NN(ni=ni, nh=ni, no=1)

    nn.fit(X, y_train)

    for x in X:
        print x, nn.predict(x)


if __name__ == '__main__':
    main()
