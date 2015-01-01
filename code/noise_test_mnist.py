#!/usr/bin/env python
# -*- coding: utf-8 -*-
# noise_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

import time

import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint

from test_mnist import test_mnist


def noise_test():
    scores = []
    x = []
    n_samples = 10000
    for cl in np.arange(0, 26) * 0.01:
        score = test_mnist(corruption_level=cl, epochs=n_samples, verbose=False)
        scores.append(score)
        x.append(cl)
    scores = np.array(score)
    x = np.array(x)

    with open('noise_test_{0}.txt'.format(time.time()), 'w') as f:
        f.write('Result\n')
        f.write('mean: {0}\n'.format(scores.mean()))
        f.write('var: {0}\n'.format(scores.var()))

    ax = plt.subplot()
    ax.plot(x, scores)
    ax.set_title('noise and score with {0}'.format(n_samples))
    ax.set_xlabel('noise')
    ax.set_ylabel('score')
    plt.savefig('../output/noise_test_mnist_{0}.png'.format(n_samples))


if __name__ == '__main__':
    noise_test()