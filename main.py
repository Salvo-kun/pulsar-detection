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

def z_normalize(DTR, DTE):
    return scipy.stats.zscore(DTR, axis=1), scipy.stats.zscore(DTE, axis=1)

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

def pearsonCorrelation(DTR):
    pearsonCorrelationMatrix = numpy.zeros((DTR.shape[0], DTR.shape[0]))
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[0]):
            pearsonCorrelationMatrix[i][j] += abs(utils.compute_cov(DTR[i], DTR[j])/(utils.compute_cov(DTR[i]) ** 0.5 * utils.compute_cov(DTR[j]) ** 0.5))
        
    return pearsonCorrelationMatrix

def plot_hist(D, L, prefix) -> None:
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for idx in range(len(features)):
        plt.figure()
        plt.xlabel(features[idx])
        plt.hist(D0[idx, :], bins = 50, density = True, alpha = 0.4, label = labels[0])
        plt.hist(D1[idx, :], bins = 50, density = True, alpha = 0.4, label = labels[1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{prefix}_{idx}.png')
    plt.show()
    
def plot_heatmap(corrMatrix, name):
    plt.imshow(corrMatrix, cmap='binary', interpolation='nearest')
    plt.savefig(f'{name}.png')

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
    DEV, LEV = load_dataset('data/Test.txt')
    
    # Useless, better to use the z-normalized
    # plot_hist(DTR, LTR, 'plots/features/hist_raw') 
    
    DTR_n, DEV_n = z_normalize(DTR, DEV)
    # plot_hist(DTR_n, LTR, 'plots/features/hist_znorm')
    
    DTR_g, DEV_g = gaussianize(DTR_n, DEV_n)
    # plot_hist(DTR_g, LTR, 'plots/features/hist_gauss')
    
    # plot_heatmap(pearsonCorrelation(DTR_g), 'plots/features/corrHeatmap_all_gauss')
    # plot_heatmap(pearsonCorrelation(DTR_g[:, LTR == 1]), 'plots/features/corrHeatmap_pulsar_gauss')
    # plot_heatmap(pearsonCorrelation(DTR_g[:, LTR == 0]), 'plots/features/corrHeatmap_notPulsar_gauss')
    
    splits = utils.K_folds_split(DTR_g, LTR, 5)
    
