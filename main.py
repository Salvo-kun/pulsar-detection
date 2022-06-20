# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:33:02 2022

@author: Salvo
"""

import numpy
import utils
import matplotlib.pyplot as plt
import scipy.stats

features = {
    0: 'Mean of the integrated profile',
    1: 'Standard deviation of the integrated profile',
    2: 'Excess kurtosis of the integrated profile',
    3: 'Skewness of the integrated profile',
    4: 'Mean of the DM-SNR curve',
    5: 'Standard deviation of the DM-SNR curve',
    6: 'Excess kurtosis of the DM-SNR curve',
    7: 'Skewness of the DM-SNR curve'
}

labels = {
    0: 'Not Pulsar',
    1: 'Pulsar'
}

def gaussianize(DTR, DTE):
    rankTR = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rankTR[i][j] += (DTR[i] < DTR[i][j]).sum() + 1
    rankTR /= DTR.shape[1] + 2
    
    rankTE = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rankTE[i][j] += (DTR[i] < DTE[i][j]).sum() + 1
    rankTE /= DTR.shape[1] + 2
    
    return scipy.stats.norm.ppf(rankTR), scipy.stats.norm.ppf(rankTE)

def plot_hist(D, L, prefix) -> None:
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for idx in range(len(features)):
        plt.figure()
        plt.xlabel(features[idx])
        plt.hist(D0[idx, :], bins = 10, density = True, alpha = 0.4, label = labels[0])
        plt.hist(D1[idx, :], bins = 10, density = True, alpha = 0.4, label = labels[1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{prefix}_{idx}.png')
    plt.show()

def load_dataset(filename):

    with open(filename, 'r') as f:
        vects = []
        labels = []
        for line in f:
            row = line.strip().split(',')
            vects.append(utils.mcol(numpy.array(row[:-1], dtype=float)))
            labels.append(row[-1:])
        return numpy.hstack(vects), numpy.array(labels, dtype=numpy.int32).ravel()
    
    
if __name__ == '__main__':
    DTR, LTR = load_dataset('data/Train.txt')
    DTE, LTE = load_dataset('data/Test.txt')
    plot_hist(DTR, LTR, 'plots/features/raw/hist')
    DTR_g, DTE_g = gaussianize(DTR, DTE)
    plot_hist(DTR_g, LTR, 'plots/features/gaussianized/hist_')
