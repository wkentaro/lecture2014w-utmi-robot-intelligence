#!/usr/bin/env python
# -*- coding: utf-8 -*-
# inertia_rate_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

from __future__ import print_function
import time

import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint

from test_mnist import test_mnist


def inertia_rate_test_mnist():
    print("... trying inertia_rate test")
    scores, x = [], []
    n_samples = 70000
    for ir in np.arange(0, 20) * 0.02:
        print("...... inertia_rate: {0} ".format(ir), end='')
        score, _ = test_mnist(corruption_level=0.0,
                              learning_rate=0.3,
                              inertia_rate=ir,
                              epochs=n_samples,
                              verbose=False)
        scores.append(score)
        x.append(ir)
        print("score: {0}".format(score))
    scores = np.array(scores)
    x = np.array(x)
    print("--- done")

    print("... ploting points")
    # for graph
    ax1 = plt.subplot()
    ax1.plot(x, scores)
    ax1.set_title('inertia_rate and score with {0}'.format(n_samples))
    ax1.set_xlabel('inertia rate level')
    ax1.set_ylabel('score')
    # for label
    ax2 = plt.subplot()
    label_text = r"""
        $\mu = %.5f$
        $\sigma = %.5f$
        """ % (scores.mean(), scores.std())
    label = ax2.text(0.05, 0.05, label_text,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax2.transAxes)
    plt.savefig('../output/inertia_rate_test_mnist_{0}.png'.format(n_samples))
    print("--- done")


if __name__ == '__main__':
    inertia_rate_test_mnist()