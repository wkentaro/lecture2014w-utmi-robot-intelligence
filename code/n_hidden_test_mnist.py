#!/usr/bin/env python
# -*- coding: utf-8 -*-
# n_hidden_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

from __future__ import print_function
import time
import cPickle

import numpy as np
import matplotlib.pyplot as plt

from test_mnist import test_mnist


def n_hidden_test_mnist():
    print("... doing n_hidden test")
    # least hidden layer n_neuron analysis
    scores, x, nns = [], [], []
    n_samples = 70000
    for nh in np.arange(1, 40) * 0.02:
        print("...... n_hidden: {0}".format(nh), end='')
        score, nn = test_mnist(nh=nh,
                               corruption_level=0.04,
                               learning_rate=0.3,
                               inertia_rate=0.24,
                               epochs=n_samples)
        scores.append(score)
        x.append(nh)
        nns.append(nn)
        print("score: {0}".format(score))
    scores, x = np.array(scores), np.array(x)
    print("--- done")

    print("... plotting test result")
    print("...... plotting scores")
    # plot
    ax = plt.subplot()
    ax.plot(x, scores)
    ax.set_title('n_hidden and score with {0}'.format(n_samples))
    ax.set_xlabel('number of hidden layer neurons')
    ax.set_ylabel('score')
    plt.savefig('../output/n_hidden_test_mnist_score_{0}.png'.format(n_samples))
    print("...... showing hidden layer weights as images")
    # about minimal hidden layer neurons
    for index in [np.argmax(scores), np.argmin(scores)]:
        W = nns[index].wi
        n_hidden = x[index]
        p = np.arange(0, 25)
        for i, w in enumerate(W[p]):
            ax = plt.subplot(5, 5, i+1)
            ax.axis('off')
            w = np.delete(w, 0)
            ax.imshow(w.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.savefig('../output/n_hidden_test_mnist_image_nsmpl{0}_nh{1}.png'.format(n_samples, n_hidden))
    print("--- done")


if __name__ == '__main__':
    n_hidden_test_mnist()