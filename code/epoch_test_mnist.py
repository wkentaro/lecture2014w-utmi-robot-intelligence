#!/usr/bin/env python
# -*- coding: utf-8 -*-
# epoch_test_mnist.py
# author: Kentaro Wada <www.kentaro.wada@gmail.com>

from __future__ import print_function
import time
import cPickle
import gzip

import numpy as np
import matplotlib.pyplot as plt

from test_mnist import test_mnist


def epoch_test_mnist():
    print("... trying epoch test")
    scores, x = [], []
    for nsmpl in np.arange(1, 300) * 5000:
        try:
            print("...... epoch: {0} ".format(nsmpl), end='')
            score, _ = test_mnist(corruption_level=0.0,
                                learning_rate=0.3,
                                inertia_rate=0.0,
                                nh=0.1,
                                epochs=nsmpl,
                                verbose=False)
            scores.append(score)
            x.append(nsmpl)
            print("score: {0}".format(score))
        except KeyboardInterrupt:
            break
    scores = np.array(scores)
    x = np.array(x)
    print("--- done")

    print("... ploting points")
    # for graph
    ax1 = plt.subplot()
    ax1.plot(x, scores)
    ax1.set_title('Relation between epoch and score')
    ax1.set_xlabel('epoch')
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
    plt.savefig('../output/epoch_test_mnist.png')
    print("--- done")

    print("... saving the results")
    dump_data = {'epoch': x,
            'score': scores}
    with gzip.open('../output/epoch_test_mnist.pkl.gz', 'wb') as f:
        cPickle.dump(dump_data, f)
    print("--- done")


if __name__ == '__main__':
    epoch_test_mnist()