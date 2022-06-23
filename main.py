# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:33:02 2022

@author: Salvo
"""

import numpy
import utils
import models
import matplotlib.pyplot as plt

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
    
def plot_heatmap(corrMatrix, color, name):
    fig, ax = plt.subplots()

    im, _ = utils.heatmap(corrMatrix, features, features, ax=ax,
                       cmap=color, cbarlabel="Pearson Correlation")    
    fig.tight_layout()
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
    
def GaussianClassifiers(DTrain, LTrain):
    with open('Figures/MVG/results.txt', 'a') as f:
        utils.multiplePrint('MVG Classifiers — min DCF on the validation set using 3-fold', f)
        utils.multiplePrint('\n', f)
        
        PCAs = [8, 7, 6, 5]
        priors = [0.500, 0.100, 0.900]
        
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        
        for dims in PCAs:
                
            utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                
            llrFull = []
            llrNaive = []
            llrTied = []
            llrNaiveTied = []
            minDcf = numpy.zeros((4, len(priors)))
                
            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)                
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                            
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
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
            utils.multiplePrint(f'MVG Full:\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}', f)
            utils.multiplePrint(f'MVG Diag:\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}', f)
            utils.multiplePrint(f'MVG Tied:\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}', f)
            utils.multiplePrint(f'MVG Tied Diag:\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}', f)
            
        utils.multiplePrint('\n', f)
        
        for dims in PCAs:
            
            utils.multiplePrint(f'Gaussianized features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
            
            llrFull = []
            llrNaive = []
            llrTied = []
            llrNaiveTied = []
            minDcf = numpy.zeros((4, len(priors)))
                
            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
                DTR, DTE = utils.gaussianize(DTR, DTE)
                
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                            
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
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
            utils.multiplePrint(f'MVG Full:\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}', f)
            utils.multiplePrint(f'MVG Diag:\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}', f)
            utils.multiplePrint(f'MVG Tied:\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}', f)
            utils.multiplePrint(f'MVG Tied Diag:\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}', f)
    
def LogisticRegressionClassifiers(DTrain, LTrain):
    def PlotLinearLRMinDCF():        
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        priors = [0.500, 0.100, 0.900]
        lambdas = numpy.logspace(-5, 5, 15) 
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
        plt.savefig('Figures/LR/LinearLR_raw.png')
        
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
                    
                    DTR, DTE = utils.gaussianize(DTR, DTE)
                    
                    scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l, prior)])
                   
                minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/LR/LinearLR_gauss.png')
    
    def TrainLinearLR():
        with open('Figures/LR/Lin_results.txt', 'a') as f:
            splits = utils.K_folds_split(DTrain, LTrain, folds=3)
            LTE = numpy.hstack([s[1] for s in splits])
            utils.multiplePrint('Linear LR Classifiers — min DCF on the validation set using 3-fold', f)
            PCAs = [8, 7, 6, 5]
            priors = [0.500, 0.100, 0.900]
            l = 1e-5
            
            for dims in PCAs:                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)

                minDcfs = numpy.zeros((len(priors), len(priors)))
                 
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    scoreLLR = []
                    
                    for i in range(len(splits)):
                        trainSplits = splits[0:i] + splits[i+1:]
                        LTR = numpy.hstack([trS[1] for trS in trainSplits])
                        DTE = splits[i][0]
                        DTR = numpy.hstack([trS[0] for trS in trainSplits])
                        
                        if dims != len(features):
                            P, _, _ = utils.compute_PCA(DTR, dims)                
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)
                        
                        scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l, prior)])
                       
                    minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
                    minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
                    minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[1]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[2]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[2][0]}\t{minDcfs[2][1]}\t{minDcfs[2][2]}', f)
            
            utils.multiplePrint('\n', f)
            
            for dims in PCAs:                    
                utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                
                minDcfs = numpy.zeros((len(priors), len(priors)))
                 
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    scoreLLR = []
        
                    for i in range(len(splits)):
                        trainSplits = splits[0:i] + splits[i+1:]
                        LTR = numpy.hstack([trS[1] for trS in trainSplits])
                        DTE = splits[i][0]
                        DTR = numpy.hstack([trS[0] for trS in trainSplits])
                        
                        DTR, DTE = utils.gaussianize(DTR, DTE)
                        
                        if dims != len(features):
                            P, _, _ = utils.compute_PCA(DTR, dims)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)
                        
                        scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l, prior)])
                       
                    minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
                    minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
                    minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[1]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[2]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[2][0]}\t{minDcfs[2][1]}\t{minDcfs[2][2]}', f)
            
    def PlotQuadraticLRMinDCF():
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        priors = [0.500, 0.100, 0.900]
        lambdas = numpy.logspace(-5, 5, num=15)
        minDcfs = numpy.zeros((len(priors), len(lambdas)))
        
        for pidx in range(len(priors)):
            prior = priors[pidx]
            print(f'Prior: {prior}')
                    
            for lidx in range(len(lambdas)):
                l = lambdas[lidx]
                scoreLLR = []
                print(f'Lambda: {l}')
    
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
        plt.savefig('Figures/LR/QuadraticLR_raw.png')
        
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
                    
                    DTR, DTE = utils.gaussianize(DTR, DTE)
                    
                    scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l, prior)])
                   
                minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/LR/QuadraticLR_gauss.png')
    
    def TrainQuadraticLR():
        with open('Figures/LR/Quad_results.txt', 'a') as f:
            splits = utils.K_folds_split(DTrain, LTrain, folds=3)
            LTE = numpy.hstack([s[1] for s in splits])
            utils.multiplePrint('Quadratic LR Classifiers — min DCF on the validation set using 3-fold', f)
            PCAs = [8, 7, 6, 5]
            priors = [0.500, 0.100, 0.900]
            l = 1e-5
            
            for dims in PCAs:                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                
                minDcfs = numpy.zeros((len(priors), len(priors)))
                 
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    scoreLLR = []
                    
                    for i in range(len(splits)):
                        trainSplits = splits[0:i] + splits[i+1:]
                        LTR = numpy.hstack([trS[1] for trS in trainSplits])
                        DTE = splits[i][0]
                        DTR = numpy.hstack([trS[0] for trS in trainSplits])
                        
                        if dims != len(features):
                            P, _, _ = utils.compute_PCA(DTR, dims)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)
                        
                        scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l, prior)])
                       
                    minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
                    minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
                    minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                
            for dims in PCAs:                    
                utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                
                minDcfs = numpy.zeros((len(priors), len(priors)))
                 
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    scoreLLR = []
                    
                    for i in range(len(splits)):
                        trainSplits = splits[0:i] + splits[i+1:]
                        LTR = numpy.hstack([trS[1] for trS in trainSplits])
                        DTE = splits[i][0]
                        DTR = numpy.hstack([trS[0] for trS in trainSplits])
                        
                        DTR, DTE = utils.gaussianize(DTR, DTE)
                        
                        if dims != len(features):
                            P, _, _ = utils.compute_PCA(DTR, dims)
                            DTR = numpy.dot(P.T, DTR)
                            DTE = numpy.dot(P.T, DTE)
                        
                        scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l, prior)])
                       
                    minDcfs[pidx][0] += utils.computeMinDCF(scoreLLR, priors[0], 1, 1, LTE)
                    minDcfs[pidx][1] += utils.computeMinDCF(scoreLLR, priors[1], 1, 1, LTE)
                    minDcfs[pidx][2] += utils.computeMinDCF(scoreLLR, priors[2], 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
    
    # PlotLinearLRMinDCF()
    TrainLinearLR()
    # PlotQuadraticLRMinDCF()
    TrainQuadraticLR()

def SupportVectorMachineClassifiers(DTrain, LTrain):
    priors = [0.500, 0.100, 0.900]
    polyKernel = lambda x1, x2: (numpy.dot(x1.T, x2) + 1) ** 2
    rbfKernel = lambda width: lambda x1, x2: numpy.exp(- width * (utils.mcol((x1**2).sum(0)) + utils.mrow((x2**2).sum(0)) - 2*numpy.dot(x1.T, x2)))

    def PlotLinearSVMMinDCF():
        Cs = numpy.logspace(-4,-1, 15) 
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        minDcfs = numpy.zeros((2, len(priors), len(Cs)))
        
        for cidx in range(len(Cs)):
            C = Cs[cidx]
            scoreLSVM = []
            scoreLSVMBal = []
            print(f'C: {C}')
            
            for i in range(len(splits)):
                print(f'Split: {i}')
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0.5)])
                
            for pidx in range(len(priors)):
                prior = priors[pidx]
                print(f'Prior: {prior}')
                minDcfs[0][pidx][cidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                minDcfs[1][pidx][cidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/LinearSVM_raw.png')
        
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/LinearSVMBal_raw.png')
        
        minDcfs = numpy.zeros((2, len(priors), len(Cs)))
        
        for cidx in range(len(Cs)):
            C = Cs[cidx]
            scoreLSVM = []
            scoreLSVMBal = []
            print(f'C: {C}')
            
            for i in range(len(splits)):
                print(f'Split: {i}')
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
                DTR, DTE = utils.gaussianize(DTR, DTE)
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0.5)])
                
            for pidx in range(len(priors)):
                prior = priors[pidx]
                print(f'Prior: {prior}')
                minDcfs[0][pidx][cidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                minDcfs[1][pidx][cidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/LinearSVM_gauss.png')
        
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/LinearSVMBal_gauss.png')

    def TrainLinearSVMClassifiers():
        with open('Figures/SVM/Lin_results.txt', 'a') as f:
            utils.multiplePrint('Linear SVM Classifiers — min DCF on the validation set using 3-fold', f)
            
            PCAs = [8, 7, 6, 5]
            C = 1e-1
            splits = utils.K_folds_split(DTrain, LTrain, folds=3)
            LTE = numpy.hstack([s[1] for s in splits])
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                
                for i in range(len(splits)):
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                        
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Linear SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                
            for dims in PCAs:
                    
                utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                
                for i in range(len(splits)):
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    DTR, DTE = utils.gaussianize(DTR, DTE)
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                                
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, 0.1, C, DTE, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Linear SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
    
    def PlotQuadraticSVMMinDCF():
        Cs = numpy.logspace(-4,-1, 15) 
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        minDcfs = numpy.zeros((2, len(priors), len(Cs)))
        
        for cidx in range(len(Cs)):
            C = Cs[cidx]
            scoreLSVM = []
            scoreLSVMBal = []
            print(f'C: {C}')
            
            for i in range(len(splits)):
                print(f'Split: {i}')
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0.5)])
                
            for pidx in range(len(priors)):
                prior = priors[pidx]
                print(f'Prior: {prior}')
                minDcfs[0][pidx][cidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                minDcfs[1][pidx][cidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/QuadSVM_raw.png')
        
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/QuadSVMBal_raw.png')
        
        minDcfs = numpy.zeros((2, len(priors), len(Cs)))
        
        for cidx in range(len(Cs)):
            C = Cs[cidx]
            scoreLSVM = []
            scoreLSVMBal = []
            print(f'C: {C}')
            
            for i in range(len(splits)):
                print(f'Split: {i}')
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
                DTR, DTE = utils.gaussianize(DTR, DTE)
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0.5)])
                
            for pidx in range(len(priors)):
                prior = priors[pidx]
                print(f'Prior: {prior}')
                minDcfs[0][pidx][cidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                minDcfs[1][pidx][cidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/QuadSVM_gauss.png')
        
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/QuadSVMBal_gauss.png')
    
    def TrainQuadraticSVMClassifiers():
        with open('Figures/SVM/Quad_results.txt', 'a') as f:
            utils.multiplePrint('Quadratic SVM Classifiers — min DCF on the validation set using 3-fold', f)
            
            PCAs = [8, 7, 6, 5]
            C = 1e-1
            splits = utils.K_folds_split(DTrain, LTrain, folds=3)
            LTE = numpy.hstack([s[1] for s in splits])
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                    
                for i in range(len(splits)):
                    print(f'Split: {i}')
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                    
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Quadratic SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
            
            utils.multiplePrint('\n', f)
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                
                for i in range(len(splits)):
                    print(f'Split: {i}')
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    DTR, DTE = utils.gaussianize(DTR, DTE)
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                    
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, polyKernel, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Quadratic SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
    
    def PlotRBFSVMMinDCF():
        Cs = numpy.logspace(-4,-1, 15) 
        widths = [1e-5, 1e-4, 1e-3]
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        minDcfs = numpy.zeros((2, len(widths), len(Cs)))
        
        for cidx in range(len(Cs)):
            C = Cs[cidx]
            print(f'C: {C}')
            for widx in range(len(widths)):
                width = widths[widx]
                print(f'width: {width}')
                scoreLSVM = []
                scoreLSVMBal = []
                
                for i in range(len(splits)):
                    print(f'Split: {i}')
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0.5)])
                    
                minDcfs[0][widx][cidx] += utils.computeMinDCF(scoreLSVM, 0.5, 1, 1, LTE)
                minDcfs[1][widx][cidx] += utils.computeMinDCF(scoreLSVMBal, 0.5, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/RBFSVM_raw.png')
        
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/RBFSVMBal_raw.png')
        
        minDcfs = numpy.zeros((2, len(widths), len(Cs)))
        
        for cidx in range(len(Cs)):
            C = Cs[cidx]
            for widx in range(len(widths)):
                width = widths[widx]
                scoreLSVM = []
                scoreLSVMBal = []
                print(f'C: {C}')
                
                for i in range(len(splits)):
                    print(f'Split: {i}')
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                   
                    DTR, DTE = utils.gaussianize(DTR, DTE)
                    
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0.5)])
                    
                minDcfs[0][widx][cidx] += utils.computeMinDCF(scoreLSVM, 0.5, 1, 1, LTE)
                minDcfs[1][widx][cidx] += utils.computeMinDCF(scoreLSVMBal, 0.5, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/RBFSVM_gauss.png')
        
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/RBFSVMBal_gauss.png')
    
    def TrainRBFSVMClassifiers():
        with open('Figures/SVM/RBF_results.txt', 'a') as f:
            utils.multiplePrint('RBF SVM Classifiers — min DCF on the validation set using 3-fold', f)
            
            PCAs = [8, 7, 6, 5]
            C = 1e1
            width = 1e1-1
            splits = utils.K_folds_split(DTrain, LTrain, folds=3)
            LTE = numpy.hstack([s[1] for s in splits])
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                    
                for i in range(len(splits)):
                    print(f'Split: {i}')
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                        
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'RBF SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'RBF SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                
                utils.multiplePrint('\n', f)
                
            for dims in PCAs:
                    
                utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                
                for i in range(len(splits)):
                    print(f'Split: {i}')
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    DTR, DTE = utils.gaussianize(DTR, DTE)
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                        
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, 0.1, C, DTE, rbfKernel(width), 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'RBF SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'RBF SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
    
    print('Linear Plot')
    # PlotLinearSVMMinDCF()
    # TrainLinearSVMClassifiers()
    print('Quad Plot')
    PlotQuadraticSVMMinDCF()
    # TrainQuadraticSVMClassifiers()
    print('RBF Plot')
    # PlotRBFSVMMinDCF()
    # TrainRBFSVMClassifiers()

def GaussianMixturesClassifiers(DTrain, LTrain):
    priors = [0.500, 0.100, 0.900]
    
    def PlotGMMMinDCF():
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        components = numpy.array([2**i for i in range(7)])
        minDcfs = numpy.zeros((4, len(components)))
        minDcfsG = numpy.zeros((4, len(components)))
        
        for cidx in range(len(components)):
            c = components[cidx]
            llrFull = []
            llrNaive = []
            llrTied = []
            llrNaiveTied = []
            print(f'Components: {c}')
            
            for i in range(len(splits)):
                print(f'Split: {i}')
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                            
                llrFull = numpy.hstack([llrFull, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='Full')])
                llrNaive = numpy.hstack([llrNaive, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='Diag')])
                llrTied = numpy.hstack([llrTied, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='Tied')])
                llrNaiveTied = numpy.hstack([llrNaiveTied, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='TiedDiag')])
            
            minDcfs[0][cidx] += utils.computeMinDCF(llrFull, 0.5, 1, 1, LTE)
            minDcfs[1][cidx] += utils.computeMinDCF(llrNaive, 0.5, 1, 1, LTE)
            minDcfs[2][cidx] += utils.computeMinDCF(llrTied, 0.5, 1, 1, LTE)
            minDcfs[3][cidx] += utils.computeMinDCF(llrNaiveTied, 0.5, 1, 1, LTE)
            
        for cidx in range(len(components)):
            c = components[cidx]
            llrFull = []
            llrNaive = []
            llrTied = []
            llrNaiveTied = []
            print(f'Components: {c}')
            
            for i in range(len(splits)):
                print(f'Split: {i}')
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
                DTR, DTE = utils.gaussianize(DTR, DTE)
                
                llrFull = numpy.hstack([llrFull, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='Full')])
                llrNaive = numpy.hstack([llrNaive, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='Diag')])
                llrTied = numpy.hstack([llrTied, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='Tied')])
                llrNaiveTied = numpy.hstack([llrNaiveTied, models.trainGMM(DTR, LTR, DTE, 0.1, c, covType='TiedDiag')])
            
            minDcfsG[0][cidx] += utils.computeMinDCF(llrFull, 0.5, 1, 1, LTE)
            minDcfsG[1][cidx] += utils.computeMinDCF(llrNaive, 0.5, 1, 1, LTE)
            minDcfsG[2][cidx] += utils.computeMinDCF(llrTied, 0.5, 1, 1, LTE)
            minDcfsG[3][cidx] += utils.computeMinDCF(llrNaiveTied, 0.5, 1, 1, LTE)
            
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout = True)
        width = 0.25
        ind = numpy.arange(len(components)) 
        b1n = ax1.bar(ind, minDcfs[0], width, color = 'r')
        b1g = ax1.bar(ind+width, minDcfsG[0], width, color='g')
        ax1.legend((b1n, b1g), ('Raw', 'Gaussianized'))
        ax1.title.set_text('Full')
        b2n = ax2.bar(ind, minDcfs[1], width, color = 'r')
        b2g = ax2.bar(ind+width, minDcfsG[1], width, color='g')
        ax2.legend((b2n, b2g), ('Raw', 'Gaussianized'))
        ax2.title.set_text('Diag')
        b3n = ax3.bar(ind, minDcfs[2], width, color = 'r')
        b3g = ax3.bar(ind+width, minDcfsG[2], width, color='g')
        ax3.legend((b3n, b3g), ('Raw', 'Gaussianized'))
        ax3.title.set_text('Tied')
        b4n = ax4.bar(ind, minDcfs[3], width, color = 'r')
        b4g = ax4.bar(ind+width, minDcfsG[3], width, color='g')
        ax4.legend((b4n, b4g), ('Raw', 'Gaussianized'))
        ax4.title.set_text('Tied Diag')
        plt.xticks(ind+width, components)
        plt.savefig('Figures/GMM/GMM.png')
    
    def TrainGMMClassifiers():
        with open('Figures/GMM/results.txt', 'a') as f:
            utils.multiplePrint('MVG Classifiers — min DCF on the validation set using 3-fold', f)
            splits = utils.K_folds_split(DTrain, LTrain, folds=3)
            LTE = numpy.hstack([s[1] for s in splits])
            PCAs = [8, 7, 6, 5]
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                    
                llrFull = []
                llrNaive = []
                llrTied = []
                llrNaiveTied = []
                minDcf = numpy.zeros((4, len(priors)))
    
                for i in range(len(splits)):
                    print(f'Split: {i}')
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                    
                    llrFull = numpy.hstack([llrFull, models.trainGMM(DTR, LTR, DTE, 0.1, 8, covType='Full')])
                    llrNaive = numpy.hstack([llrNaive, models.trainGMM(DTR, LTR, DTE, 0.1, 16, covType='Diag')])
                    llrTied = numpy.hstack([llrTied, models.trainGMM(DTR, LTR, DTE, 0.1, 32, covType='Tied')])
                    llrNaiveTied = numpy.hstack([llrNaiveTied, models.trainGMM(DTR, LTR, DTE, 0.1, 32, covType='TiedDiag')])
                
                for p in range(len(priors)):
                    prior = priors[p]
                        
                    minDcf[0][p] += utils.computeMinDCF(llrFull, prior, 1, 1, LTE)
                    minDcf[1][p] += utils.computeMinDCF(llrNaive, prior, 1, 1, LTE)
                    minDcf[2][p] += utils.computeMinDCF(llrTied, prior, 1, 1, LTE)
                    minDcf[3][p] += utils.computeMinDCF(llrNaiveTied, prior, 1, 1, LTE)
            
                minDcf = numpy.around(minDcf, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'GMM Full (64 gau):\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}', f)
                utils.multiplePrint(f'GMM Diag (128 gau):\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}', f)
                utils.multiplePrint(f'GMM Tied (512 gau):\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}', f)
                utils.multiplePrint(f'GMM Tied Diag (512 gau):\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}', f)
                
            utils.multiplePrint('\n', f)
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Gaussianized features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                    
                llrFull = []
                llrNaive = []
                llrTied = []
                llrNaiveTied = []
                minDcf = numpy.zeros((4, len(priors)))
    
                for i in range(len(splits)):
                    trainSplits = splits[0:i] + splits[i+1:]
                    LTR = numpy.hstack([trS[1] for trS in trainSplits])
                    DTE = splits[i][0]
                    DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
                    DTR, DTE = utils.gaussianize(DTR, DTE)
                    
                    if dims != len(features):
                        P, _, _ = utils.compute_PCA(DTR, dims)
                        DTR = numpy.dot(P.T, DTR)
                        DTE = numpy.dot(P.T, DTE)
                    
                    llrFull = numpy.hstack([llrFull, models.trainGMM(DTR, LTR, DTE, 0.1, 8, covType='Full')])
                    llrNaive = numpy.hstack([llrNaive, models.trainGMM(DTR, LTR, DTE, 0.1, 16, covType='Diag')])
                    llrTied = numpy.hstack([llrTied, models.trainGMM(DTR, LTR, DTE, 0.1, 32, covType='Tied')])
                    llrNaiveTied = numpy.hstack([llrNaiveTied, models.trainGMM(DTR, LTR, DTE, 0.1, 32, covType='TiedDiag')])
                
                for p in range(len(priors)):
                    prior = priors[p]
                        
                    minDcf[0][p] += utils.computeMinDCF(llrFull, prior, 1, 1, LTE)
                    minDcf[1][p] += utils.computeMinDCF(llrNaive, prior, 1, 1, LTE)
                    minDcf[2][p] += utils.computeMinDCF(llrTied, prior, 1, 1, LTE)
                    minDcf[3][p] += utils.computeMinDCF(llrNaiveTied, prior, 1, 1, LTE)
            
                minDcf = numpy.around(minDcf, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}')
                utils.multiplePrint(f'GMM Full (64 gau):\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}', f)
                utils.multiplePrint(f'GMM Diag (128 gau):\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}', f)
                utils.multiplePrint(f'GMM Tied (512 gau):\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}', f)
                utils.multiplePrint(f'GMM Tied Diag (512 gau):\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}', f)
    
    PlotGMMMinDCF()
    # TrainGMMClassifiers()
    
if __name__ == '__main__':
    DTrain, LTrain = load_dataset('./data/Train.txt')
    DEval, LEval = load_dataset('./data/Test.txt')
    
    DTR_n, DEV_n = utils.z_normalize(DTrain, DEval)
    # plot_hist(DTR_n, LTrain, 'Figures/features/hist_znorm')
    
    DTR_g, DEV_g = utils.gaussianize(DTR_n, DEV_n)
    # plot_hist(DTR_g, LTrain, 'Figures/features/hist_gauss')
    
    # plot_heatmap(utils.pearsonCorrelation(DTR_n), 'Greys', 'Figures/features/corrHeatmap_all_raw')
    # plot_heatmap(utils.pearsonCorrelation(DTR_n[:, LTrain == 1]), 'Reds', 'Figures/features/corrHeatmap_pulsar_raw')
    # plot_heatmap(utils.pearsonCorrelation(DTR_n[:, LTrain == 0]), 'Blues', 'Figures/features/corrHeatmap_notPulsar_raw')
    # plot_heatmap(utils.pearsonCorrelation(DTR_g), 'Greys', 'Figures/features/corrHeatmap_all_gauss')
    # plot_heatmap(utils.pearsonCorrelation(DTR_g[:, LTrain == 1]), 'Reds', 'Figures/features/corrHeatmap_pulsar_gauss')
    # plot_heatmap(utils.pearsonCorrelation(DTR_g[:, LTrain == 0]), 'Blues', 'Figures/features/corrHeatmap_notPulsar_gauss')
    
    
    
    # GaussianClassifiers(DTR_n, LTrain)
    # LogisticRegressionClassifiers(DTR_n, LTrain)    
    SupportVectorMachineClassifiers(DTR_n, LTrain)    
    # GaussianMixturesClassifiers(DTR_n, LTrain)
