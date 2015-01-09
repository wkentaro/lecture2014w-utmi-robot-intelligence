#!/usr/bin/env python
#-*- coding: utf-8 -*-
# plot_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

from utils import tile_raster_images
import cv2


def plot_mnist():
    # load data
    mnist = fetch_mldata('MNIST original')

    # get tiled image
    p = np.random.randint(0, len(mnist.data), 400)
    tile = tile_raster_images(mnist.data[p], (28,28), (20,20), scale_rows_to_unit_interval=False)
    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2RGB)

    # save tiled data's image
    plt.imshow(tile)
    plt.savefig('../output/tiled_mnist.png')


if __name__ == '__main__':
    plot_mnist()
