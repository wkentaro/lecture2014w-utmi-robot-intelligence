#!/usr/bin/env python
#-*- coding: utf-8 -*-
# plot_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata


def plot_mnist():
    mnist = fetch_mldata('MNIST original', data_home='.')

    p = np.random.randint(0, len(mnist.data), 25)
    for i, (data, label) in enumerate(zip(mnist.data[p], mnist.target[p])):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(data.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('{0}'.format(label))
    plt.savefig('../output/mnist.png')


if __name__ == '__main__':
    plot_mnist()
