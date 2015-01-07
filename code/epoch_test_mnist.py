#!/usr/bin/env python
# -*- coding: utf-8 -*-
# learning_rate_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import time

import numpy as np
import matplotlib.pyplot as plt

from test_mnist import test_mnist


def learning_rate_test_mnist():
    scores, x = [], []
    for nsmpl in np.arange(1, 400) * 5000:
        score, _ = test_mnist(corruption_level=0.0,
                              learning_rate=0.3,
                              inertia_rate=0.24,
                              epochs=nsmpl,
                              verbose=False)
        scores.append(score)
        x.append(nsmpl)
    scores = np.array(scores)
    x = np.array(x)

    ax1 = plt.subplot()
    ax1.plot(x, scores)
    ax1.set_title('Relation between number of samples and score')
    ax1.set_xlabel('number of samples')
    ax1.set_ylabel('score')

    ax2 = plt.subplot()
    label_text = r"""
        $\mu = %.5f$
        $\sigma = %.5f$
        """ % (scores.mean(), scores.std())
    label = ax2.text(0.05, 0.05, label_text,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax2.transAxes)
    plt.savefig('../output/epoch_test_mnist.png')


if __name__ == '__main__':
    learning_rate_test_mnist()