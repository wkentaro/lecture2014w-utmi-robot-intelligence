#!/usr/bin/env python
# -*- coding: utf-8 -*-
# n_hidden_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

from __future__ import print_function
import time
import cPickle
import gzip

import numpy as np
import matplotlib.pyplot as plt

from test_mnist import test_mnist
from utils import tile_raster_images


def n_hidden_test_mnist():
    print("... doing n_hidden test")
    # least hidden layer n_neuron analysis
    scores, x, nns = [], [], []
    epochs = 150000
    for nh in np.arange(1, 20) * 0.04:
        print("...... n_hidden: {0}".format(nh), end='')
        score, nn = test_mnist(nh=nh,
                               corruption_level=0.0,
                               noise_level=0.0,
                               learning_rate=0.3,
                               inertia_rate=0.12,
                               epochs=epochs)
        scores.append(score)
        x.append(nh)
        nns.append(nn)
        print(" score: {0}".format(score))
    scores, x = np.array(scores), np.array(x)
    print("--- done")

    print("... plotting test result")
    print("...... plotting scores")
    # plot
    ax = plt.subplot()
    ax.plot(x, scores)
    ax.set_title('n_hidden and score with {0}'.format(epochs))
    ax.set_xlabel('number of hidden layer neurons')
    ax.set_ylabel('score')
    plt.savefig('../output/n_hidden_test_mnist_score_{0}.png'.format(epochs))
    print("...... showing hidden layer weights as images")
    # about minimal hidden layer neurons
    for index in [np.argmax(scores), np.argmin(scores)]:
        W = nns[index].wi
        W = np.delete(W, 0, axis=1)
        n_hidden = x[index]
        p = np.random.randint(0, len(W), 400)
        tile = tile_raster_images(W[p], (28,28), (20,20), scale_rows_to_unit_interval=True, output_pixel_vals=True, tile_spacing=(1,1))
        # save tiled data's image
        plt.axis('off')
        plt.imshow(tile, cmap=plt.cm.gray_r)
        plt.savefig('../output/n_hidden_test_mnist_image_nsmpl{0}_nh{1}.png'.format(epochs, n_hidden))
    print("--- done")

    print("... saving the results")
    dump_data = {'n_hidden': x,
                 'score': scores,
                 'nn': nns}
    with gzip.open('../output/n_hidden_test_mnist.pkl.gz', 'wb') as f:
        cPickle.dump(dump_data, f)
    print("--- done")


if __name__ == '__main__':
    n_hidden_test_mnist()