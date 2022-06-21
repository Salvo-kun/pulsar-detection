# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:33:02 2022

@author: Salvo
"""

import numpy
import utils
import models
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
    
def GaussianClassifiers(DT, DTg, L):
    print('MVG Classifiers — min DCF on the validation set using 5-fold')
    
    PCAs = [8, 7, 6, 5]
    priors = [0.500, 0.100, 0.900]
    
    for dims in PCAs:
            
        print(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}')
            
        llrFull = []
        llrNaive = []
        llrTied = []
        llrNaiveTied = []
        minDcf = numpy.zeros((4, len(priors)))
        
        D = DT

        if dims != len(features):
            D = utils.apply_PCA(DT, L, dims)
            
        splits = utils.K_folds_split(D, L, 5)
        LTE = numpy.hstack([s[1] for s in splits])

        for i in range(len(splits)):
            trainSplits = splits[0:i] + splits[i+1:]
            LTR = numpy.hstack([trS[1] for trS in trainSplits])
            DTE = splits[i][0]
            DTR = numpy.hstack([trS[0] for trS in trainSplits])
                        
            llrFull = numpy.hstack([llrFull, models.compute_full_llr(DTR, LTR, DTE)])
            llrNaive = numpy.hstack([llrNaive, models.compute_naive_llr(DTR, LTR, DTE)])
            llrTied = numpy.hstack([llrTied, models.compute_tied_llr(DTR, LTR, DTE)])
            llrNaiveTied = numpy.hstack([llrNaiveTied, models.compute_tied_naive_bayes_llr(DTR, LTR, DTE)])
        
        for p in range(len(priors)):
            prior = priors[p]
                
            minDcf[0][p] += utils.computeMinDCF(llrFull, prior, 1, 1, LTE)
            minDcf[1][p] += utils.computeMinDCF(llrNaive, prior, 1, 1, LTE)
            minDcf[2][p] += utils.computeMinDCF(llrTied, prior, 1, 1, LTE)
            minDcf[3][p] += utils.computeMinDCF(llrNaiveTied, prior, 1, 1, LTE)
    
        minDcf = numpy.around(minDcf, 3)
        print(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}')
        print(f'MVG Full:\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}')
        print(f'MVG Diag:\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}')
        print(f'MVG Tied:\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}')
        print(f'MVG Tied Diag:\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}')
        
    print('\n')
    
    for dims in PCAs:
            
        print(f'Gaussianized features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}')
            
        llrFull = []
        llrNaive = []
        llrTied = []
        llrNaiveTied = []
        minDcf = numpy.zeros((4, len(priors)))

        Dg = DTg
        if dims != len(features):
            Dg = utils.apply_PCA(Dg, L, dims)
            
        splits = utils.K_folds_split(Dg, L, 5)
        LTE = numpy.hstack([s[1] for s in splits])

        for i in range(len(splits)):
            trainSplits = splits[0:i] + splits[i+1:]
            LTR = numpy.hstack([trS[1] for trS in trainSplits])
            DTE = splits[i][0]
            DTR = numpy.hstack([trS[0] for trS in trainSplits])
            
            llrFull = numpy.hstack([llrFull, models.compute_full_llr(DTR, LTR, DTE)])
            llrNaive = numpy.hstack([llrNaive, models.compute_naive_llr(DTR, LTR, DTE)])
            llrTied = numpy.hstack([llrTied, models.compute_tied_llr(DTR, LTR, DTE)])
            llrNaiveTied = numpy.hstack([llrNaiveTied, models.compute_tied_naive_bayes_llr(DTR, LTR, DTE)])
        
        for p in range(len(priors)):
            prior = priors[p]
                
            minDcf[0][p] += utils.computeMinDCF(llrFull, prior, 1, 1, LTE)
            minDcf[1][p] += utils.computeMinDCF(llrNaive, prior, 1, 1, LTE)
            minDcf[2][p] += utils.computeMinDCF(llrTied, prior, 1, 1, LTE)
            minDcf[3][p] += utils.computeMinDCF(llrNaiveTied, prior, 1, 1, LTE)
    
        minDcf = numpy.around(minDcf, 3)
        print(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}')
        print(f'MVG Full:\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}')
        print(f'MVG Diag:\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}')
        print(f'MVG Tied:\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}')
        print(f'MVG Tied Diag:\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}')
        
def LogisticRegressionClassifiers(DT, DTg, L):
    print('Linear LR Classifiers — min DCF on the validation set using 5-fold')
    priors = [0.500, 0.100, 0.900]
    lambdas = numpy.logspace(-5, 5) 
    D = DT
        
    splits = utils.K_folds_split(D, L, 5)
    LTE = numpy.hstack([s[1] for s in splits])
    minDcfs = numpy.zeros((len(priors), len(lambdas)))
    
    for pidx in range(len(priors)):
        prior = priors[pidx]
                
        for lidx in range(len(lambdas)):
            l = lambdas[lidx]
            scoreLLR = []

            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l, prior)])
               
            minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)

    plt.figure()
    plt.xscale('log')
    plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
    plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
    plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
    plt.legend()
    plt.savefig('LinearLR_raw.png')
    
    Dg = DTg

    splits = utils.K_folds_split(Dg, L, 5)
    LTE = numpy.hstack([s[1] for s in splits])
    minDcfs = numpy.zeros((len(priors), len(lambdas)))
    
    for pidx in range(len(priors)):
        prior = priors[pidx]
                
        for lidx in range(len(lambdas)):
            l = lambdas[lidx]
            scoreLLR = []

            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l, prior)])
               
            minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)

    plt.figure()
    plt.xscale('log')
    plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
    plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
    plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
    plt.legend()
    plt.savefig('LinearLR_gauss.png')
            
    
    PCAs = [8, 7, 6, 5]
    l = 1e-5
    
    for dims in PCAs:
            
        print(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}')
        D = DT

        if dims != len(features):
            D = utils.apply_PCA(DT, L, dims)
            
        splits = utils.K_folds_split(D, L, 5)
        LTE = numpy.hstack([s[1] for s in splits])
        minDcfs = numpy.zeros((len(priors), len(priors)))
         
        for pidx in range(len(priors)):
            prior = priors[pidx]
            scoreLLR = []

            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l, prior)])
               
            minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
            minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
            minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)

    
        minDcfs = numpy.around(minDcfs, 3)
        print(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}')
        print(f'Linear LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}')
        
    for dims in PCAs:
            
        print(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}')
        Dg = DTg

        if dims != len(features):
            Dg = utils.apply_PCA(DTg, L, dims)
            
        splits = utils.K_folds_split(Dg, L, 5)
        LTE = numpy.hstack([s[1] for s in splits])
        minDcfs = numpy.zeros((len(priors), len(priors)))
         
        for pidx in range(len(priors)):
            prior = priors[pidx]
            scoreLLR = []

            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l, prior)])
               
            minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
            minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
            minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)

    
        minDcfs = numpy.around(minDcfs, 3)
        print(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}')
        print(f'Linear LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}')

    print('Quadratic LR Classifiers — min DCF on the validation set using 5-fold')
    priors = [0.500, 0.100, 0.900]
    lambdas = numpy.logspace(-5, 5) 
    D = DT
        
    splits = utils.K_folds_split(D, L, 5)
    LTE = numpy.hstack([s[1] for s in splits])
    minDcfs = numpy.zeros((len(priors), len(lambdas)))
    
    for pidx in range(len(priors)):
        prior = priors[pidx]
                
        for lidx in range(len(lambdas)):
            l = lambdas[lidx]
            scoreLLR = []

            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l, prior)])
               
            minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)

    plt.figure()
    plt.xscale('log')
    plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
    plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
    plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
    plt.legend()
    plt.savefig('QuadraticLR_raw.png')
    
    Dg = DTg

    splits = utils.K_folds_split(Dg, L, 5)
    LTE = numpy.hstack([s[1] for s in splits])
    minDcfs = numpy.zeros((len(priors), len(lambdas)))
    
    for pidx in range(len(priors)):
        prior = priors[pidx]
                
        for lidx in range(len(lambdas)):
            l = lambdas[lidx]
            scoreLLR = []

            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l, prior)])
               
            minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)

    plt.figure()
    plt.xscale('log')
    plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
    plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
    plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
    plt.legend()
    plt.savefig('QuadraticLR_gauss.png')
            
    
    PCAs = [8, 7, 6, 5]
    l = 1e-5
    
    # for dims in PCAs:
            
    #     print(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}')
    #     D = DT

    #     if dims != len(features):
    #         D = utils.apply_PCA(DT, L, dims)
            
    #     splits = utils.K_folds_split(D, L, 5)
    #     LTE = numpy.hstack([s[1] for s in splits])
    #     minDcfs = numpy.zeros((len(priors), len(priors)))
         
    #     for pidx in range(len(priors)):
    #         prior = priors[pidx]
    #         scoreLLR = []

    #         for i in range(len(splits)):
    #             trainSplits = splits[0:i] + splits[i+1:]
    #             LTR = numpy.hstack([trS[1] for trS in trainSplits])
    #             DTE = splits[i][0]
    #             DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
    #             scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l, prior)])
               
    #         minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
    #         minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
    #         minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)

    
    #     minDcfs = numpy.around(minDcfs, 3)
    #     print(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}')
    #     print(f'Quadratic LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}')
        
    # for dims in PCAs:
            
    #     print(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}')
    #     Dg = DTg

    #     if dims != len(features):
    #         Dg = utils.apply_PCA(DTg, L, dims)
            
    #     splits = utils.K_folds_split(Dg, L, 5)
    #     LTE = numpy.hstack([s[1] for s in splits])
    #     minDcfs = numpy.zeros((len(priors), len(priors)))
         
    #     for pidx in range(len(priors)):
    #         prior = priors[pidx]
    #         scoreLLR = []

    #         for i in range(len(splits)):
    #             trainSplits = splits[0:i] + splits[i+1:]
    #             LTR = numpy.hstack([trS[1] for trS in trainSplits])
    #             DTE = splits[i][0]
    #             DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
    #             scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l, prior)])
               
    #         minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
    #         minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
    #         minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)

    
    #     minDcfs = numpy.around(minDcfs, 3)
    #     print(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}')
    #     print(f'Quadratic LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}')

def SupportVectorMachineClassifiers(splits, gSplits):
    pass

def GaussianMixturesClassifiers(splits, gSplits):
    pass
    
if __name__ == '__main__':
    DTrain, LTrain = load_dataset('./data/Train.txt')
    DEval, LEval = load_dataset('./data/Test.txt')
    
    DTR_n, DEV_n = utils.z_normalize(DTrain, DEval)
    # plot_hist(DTR_n, LTrain, 'plots/features/hist_znorm')
    
    DTR_g, DEV_g = utils.gaussianize(DTR_n, DEV_n)
    # plot_hist(DTR_g, LTrain, 'plots/features/hist_gauss')
    
    # plot_heatmap(pearsonCorrelation(DTR_n), 'plots/features/corrHeatmap_all_raw')
    # plot_heatmap(pearsonCorrelation(DTR_n[:, LTrain == 1]), 'plots/features/corrHeatmap_pulsar_raw')
    # plot_heatmap(pearsonCorrelation(DTR_n[:, LTrain == 0]), 'plots/features/corrHeatmap_notPulsar_raw')
    # plot_heatmap(pearsonCorrelation(DTR_g), 'plots/features/corrHeatmap_all_gauss')
    # plot_heatmap(pearsonCorrelation(DTR_g[:, LTrain == 1]), 'plots/features/corrHeatmap_pulsar_gauss')
    # plot_heatmap(pearsonCorrelation(DTR_g[:, LTrain == 0]), 'plots/features/corrHeatmap_notPulsar_gauss')
    
    ### Gaussian models: MVG Full, Naive, Tied, Tied Naive
    # GaussianClassifiers(DTR_n, DTR_g, LTrain)
    LogisticRegressionClassifiers(DTR_n, DTR_g, LTrain)    
    # SupportVectorMachineClassifiers(splits, gSplits)    
    # GaussianMixturesClassifiers(splits, gSplits)
