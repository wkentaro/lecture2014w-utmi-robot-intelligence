#!/usr/bin/env python
# -*- coding: utf-8 -*-
# feature_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

from __future__ import print_function
import sys
import cPickle
import gzip

import numpy as np
import theano.tensor as T

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import (
        classification_report,
        accuracy_score,
        confusion_matrix
        )
import matplotlib.pyplot as plt

import nn
from utils import tile_raster_images

from autoencoder import AutoEncoder


def feature_test_mnist(verbose=True):
    print("... loading date")
    # load train data
    mnist = fetch_mldata('MNIST original')
    X_origin = mnist.data
    y = mnist.target
    target_names = np.unique(y)
    # standardize
    X_origin = X_origin.astype(np.float64)
    X_origin /= X_origin.max()
    print("--- done")

    print("... encoding with denoising auto-encoder")
    # get feature & create input
    ae = AutoEncoder(X=X_origin,
                     hidden_size=22*22,
                     activation_function=T.nnet.sigmoid,
                     output_function=T.nnet.sigmoid)
    ae.train(n_epochs=20, mini_batch_size=20)
    X = ae.get_hidden(data=X_origin)[0]
    print("--- done")

    # get classifier
    clf = nn.NN(ni=X.shape[1],
                nh=int(0.16*X.shape[1]),
                no=len(target_names),
                learning_rate=0.3,
                inertia_rate=0.12,
                corruption_level=0.0,
                epochs=150000)

    # cross validation
    skf = StratifiedKFold(y, n_folds=3)
    scores = np.zeros(len(skf))
    for i, (train_index, test_index) in enumerate(skf):
        # train the model
        clf.fit(X[train_index], y[train_index])
        # get score
        score = clf.score(X[test_index], y[test_index])
        scores[i] = score

    # stdout of the score
    if verbose is True:
        print(scores)

    print("... plotting the autoencoder hidden layer")
    # get tiled image
    p = np.random.randint(0, len(X), 400)
    tile = tile_raster_images(X[p], (22,22), (20,20), scale_rows_to_unit_interval=True, output_pixel_vals=True)
    # save tiled data's image
    plt.axis('off')
    plt.title('MNIST dataset')
    plt.imshow(tile, cmap=plt.cm.gray_r)
    plt.savefig('../output/tiled_autoencoder_hidden_mnist.png')
    print("--- done")

    print("... saving the results")
    data = {'scores': scores,
            'hidden layer': X,}
    with gzip.open('../output/feature_test_mnist.pkl.gz', 'wb') as f:
        cPickle.dump(data, f)
    print("--- done")


if __name__ == '__main__':
    feature_test_mnist(verbose=True)
