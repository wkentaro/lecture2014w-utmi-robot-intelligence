#!/usr/bin/env python
#-*- coding: utf-8 -*-
# plot_cifar.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

from test_cifar import load_cifar


def plot_cifar():
    cifar = load_cifar()

    p = np.random.randint(0, len(cifar.data), 25)
    for i, (data, label) in enumerate(zip(cifar.data[p], cifar.target[p])):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        data = data.reshape((3,32,32))
        data = np.rollaxis(data, 2)
        data = np.rollaxis(data, 2)
        plt.imshow(data, interpolation='nearest')
        plt.title('{0}'.format(label))
    plt.savefig('../output/cifar.png')


if __name__ == '__main__':
    plot_cifar()
