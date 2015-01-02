#!/usr/bin/env python
# -*- coding: utf-8 -*-
# hidden_layer_analyze.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import time
import cPickle

import numpy as np
import matplotlib.pyplot as plt

from test_mnist import test_mnist


def hidden_layer_analyze():
    # least hidden layer n_neuron analysis
    scores, x, nns = [], [], []
    n_samples = 70000
    for nh in np.arange(1, 40) * 0.02:
        score, nn = test_mnist(nh=nh, epochs=n_samples)
        scores.append(score)
        x.append(nh)
        nns.append(nn)
    scores, x = np.array(scores), np.array(x)
    # plot
    ax = plt.subplot()
    ax.plot(x, scores)
    ax.set_title('n_hidden and score with {0}'.format(n_samples))
    ax.set_xlabel('number of hidden layer neurons')
    ax.set_ylabel('score')
    plt.savefig('../output/hidden_layer_analyze_mnist_score_{0}.png'.format(n_samples))
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
        print 'n_hidden is {0}'.format(n_hidden)
        plt.savefig('../output/hidden_layer_analyze_mnist_image_nsamp{0}_nh{1}.png'.format(n_samples, n_hidden))


if __name__ == '__main__':
    hidden_layer_analyze()