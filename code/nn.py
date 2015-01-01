#!/usr/bin/env python
# -*- coding: utf-8 -*-
# nn.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))


def dsigmoid2(y):
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
        # hidden layer -> output layer
        self.wo = np.random.uniform(-1., 1., (self.no, self.nh))

    def fit(self, X, y_train, learning_rate=0.4, epochs=10000):
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
                x[rs] *= np.random.random()

            # forward propagation
            u_z = np.dot(self.wi, x)
            z = sigmoid(u_z)
            u_y = np.dot(self.wo, z)
            y = sigmoid(u_y)

            # compute error in output layer
            delta2 = dsigmoid(u_y) * (y - y_train[i])
            # OR delta2 = dsigmoid2(y) * (y - y_train[i])

            # update weight in hidden layer
            # using the error in output layer
            delta1 = dsigmoid(u_z) * np.dot(self.wo.T, delta2)
            # OR delta1 = dsigmoid2(z) * np.dot(self.wo.T, delta2)

            # update hidden layer weight using
            # error in hidden layer
            x = np.atleast_2d(x)
            delta1 = np.atleast_2d(delta1)
            self.wi -= learning_rate * np.dot(delta1.T, x)

            # update output layer weight using
            # error in output layer
            z = np.atleast_2d(z)
            delta2 = np.atleast_2d(delta2)
            self.wo -= learning_rate * np.dot(delta2.T, z)

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
