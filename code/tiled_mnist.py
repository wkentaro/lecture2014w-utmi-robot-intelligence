#!/usr/bin/env python
#-*- coding: utf-8 -*-
# tiled_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

from utils import tile_raster_images


def tiled_mnist(noise_level=0.0):
    # load data
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    X = X.astype(np.float64)
    X /= X.max()

    # add noise to the image
    p = np.random.binomial(n=1, p=1-noise_level, size=X.shape)
    X[p==0] = np.random.random(X.shape)[p==0]

    # get tiled image
    p = np.random.randint(0, len(X), 400)
    # tile = tile_raster_images(X[p], (28,28), (20,20), scale_rows_to_unit_interval=True, output_pixel_vals=True)
    tile = tile_raster_images(X[p], (28,28), (20,20), scale_rows_to_unit_interval=True, output_pixel_vals=False)

    # save tiled data's image
    plt.axis('off')
    plt.title('MNIST dataset')
    plt.imshow(tile, cmap=plt.cm.gray_r)
    plt.savefig('../output/tiled_mnist_nl{0}.png'.format(noise_level))


if __name__ == '__main__':
    for nl in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        tiled_mnist(noise_level=nl)
