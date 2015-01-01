#!/usr/bin/env python
#-*- coding: utf-8 -*-
# plot_digits.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def plot_digits():
    digits = load_digits()
    p = np.random.randint(0, len(digits.images), 25)
    for i, (data, label) in enumerate(zip(digits.images[p], digits.target[p])):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(data, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('{0}'.format(label))
    plt.savefig('../output/digits.png')


if __name__ == '__main__':
    plot_digits()
