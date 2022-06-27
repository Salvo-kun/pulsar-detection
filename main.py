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
                    
        for lidx in range(len(lambdas)):
            l = lambdas[lidx]
            scoreLLR = []
            
            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR_priorW(DTR, LTR, DTE, l)])
                
            for pidx in range(len(priors)):
                prior = priors[pidx]
                minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/LR/LinearLR_raw.png')
        
        minDcfs = numpy.zeros((len(priors), len(lambdas)))
        
        # for lidx in range(len(lambdas)):
        #     l = lambdas[lidx]
        #     scoreLLR = []
            
        #     for i in range(len(splits)):
        #         trainSplits = splits[0:i] + splits[i+1:]
        #         LTR = numpy.hstack([trS[1] for trS in trainSplits])
        #         DTE = splits[i][0]
        #         DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
        #         DTR, DTE = utils.gaussianize(DTR, DTE)
                
        #         scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR_priorW(DTR, LTR, DTE, l)])
                
        #     for pidx in range(len(priors)):
        #         prior = priors[pidx]
        #         minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                
        # plt.figure()
        # plt.xscale('log')
        # plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        # plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        # plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        # plt.legend()
        # plt.savefig('Figures/LR/LinearLR_gauss.png')
    
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
                minDcfs = numpy.zeros((len(priors)))
                
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
                    
                    scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR_priorW(DTR, LTR, DTE, l)])
                        
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[pidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0]}\t{minDcfs[1]}\t{minDcfs[2]}', f)
            
            utils.multiplePrint('\n', f)
            
            # for dims in PCAs:                    
            #     utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                
            #     minDcfs = numpy.zeros((len(priors)))
                
            #     scoreLLR = []
            #     for i in range(len(splits)):
            #         trainSplits = splits[0:i] + splits[i+1:]
            #         LTR = numpy.hstack([trS[1] for trS in trainSplits])
            #         DTE = splits[i][0]
            #         DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
            #         DTR, DTE = utils.gaussianize(DTR, DTE)
                    
            #         if dims != len(features):
            #             P, _, _ = utils.compute_PCA(DTR, dims)
            #             DTR = numpy.dot(P.T, DTR)
            #             DTE = numpy.dot(P.T, DTE)
                    
            #         scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR(DTR, LTR, DTE, l)])
                        
            #     for pidx in range(len(priors)):
            #         prior = priors[pidx]
            #         minDcfs[pidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                    
            #     minDcfs = numpy.around(minDcfs, 3)
            #     utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
            #     utils.multiplePrint(f'Linear LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0]}\t{minDcfs[1]}\t{minDcfs[2]}', f)
            
    def PlotQuadraticLRMinDCF():
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        priors = [0.500, 0.100, 0.900]
        lambdas = numpy.logspace(-5, 5, num=15)
        minDcfs = numpy.zeros((len(priors), len(lambdas)))
        
        for lidx in range(len(lambdas)):
            l = lambdas[lidx]
            scoreLLR = []
            print(f'Lambda: {l}')

            for i in range(len(splits)):
                trainSplits = splits[0:i] + splits[i+1:]
                LTR = numpy.hstack([trS[1] for trS in trainSplits])
                DTE = splits[i][0]
                DTR = numpy.hstack([trS[0] for trS in trainSplits])
                                                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR_priorW(DTR, LTR, DTE, l)])
                
            for pidx in range(len(priors)):
                prior = priors[pidx]
                print(f'Prior: {prior}')
                minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                
        plt.figure()
        plt.xscale('log')
        plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        plt.legend()
        plt.savefig('Figures/LR/QuadraticLR_raw.png')
        
        minDcfs = numpy.zeros((len(priors), len(lambdas)))
        
        # for lidx in range(len(lambdas)):
        #     l = lambdas[lidx]
        #     scoreLLR = []
            
        #     for i in range(len(splits)):
        #         trainSplits = splits[0:i] + splits[i+1:]
        #         LTR = numpy.hstack([trS[1] for trS in trainSplits])
        #         DTE = splits[i][0]
        #         DTR = numpy.hstack([trS[0] for trS in trainSplits])
                
        #         DTR, DTE = utils.gaussianize(DTR, DTE)
                
        #         scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR_priorW(DTR, LTR, DTE, l)])
               
        #     for pidx in range(len(priors)):
        #         prior = priors[pidx]
        #         minDcfs[pidx][lidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                
        # plt.figure()
        # plt.xscale('log')
        # plt.plot(lambdas, minDcfs[0], color='r', label=f'Min DCF (Prior = {priors[0]})')
        # plt.plot(lambdas, minDcfs[1], color='b', label=f'Min DCF (Prior = {priors[1]})')
        # plt.plot(lambdas, minDcfs[2], color='g', label=f'Min DCF (Prior = {priors[2]})')
        # plt.legend()
        # plt.savefig('Figures/LR/QuadraticLR_gauss.png')
    
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
                
                minDcfs = numpy.zeros((len(priors)))
                
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
                    
                    scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR_priorW(DTR, LTR, DTE, l)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[pidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0]}\t{minDcfs[1]}\t{minDcfs[2]}', f)
                
            # for dims in PCAs:                    
            #     utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                
            #     minDcfs = numpy.zeros((len(priors)))
                
            #     scoreLLR = []
                
            #     for i in range(len(splits)):
            #         trainSplits = splits[0:i] + splits[i+1:]
            #         LTR = numpy.hstack([trS[1] for trS in trainSplits])
            #         DTE = splits[i][0]
            #         DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
            #         DTR, DTE = utils.gaussianize(DTR, DTE)
                    
            #         if dims != len(features):
            #             P, _, _ = utils.compute_PCA(DTR, dims)
            #             DTR = numpy.dot(P.T, DTR)
            #             DTE = numpy.dot(P.T, DTE)
                    
            #         scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR(DTR, LTR, DTE, l)])
                    
            #     for pidx in range(len(priors)):
            #         prior = priors[pidx]
            #         minDcfs[pidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                    
            #     minDcfs = numpy.around(minDcfs, 3)
            #     utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
            #     utils.multiplePrint(f'Quadratic LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0]}\t{minDcfs[1]}\t{minDcfs[2]}', f)
    
    PlotLinearLRMinDCF()
    TrainLinearLR()
    PlotQuadraticLRMinDCF()
    TrainQuadraticLR()

def SupportVectorMachineClassifiers(DTrain, LTrain):
    priors = [0.500, 0.100, 0.900]
    polyKernel = lambda x1, x2: (numpy.dot(x1.T, x2) + 1) ** 2
    # rbfKernel = lambda width: lambda x1, x2: numpy.exp(-width*(numpy.linalg.norm(x1-x2)**2)) 
    rbfKernel = lambda width: lambda x1, x2: numpy.exp(- width * (utils.mcol((x1**2).sum(0)) + utils.mrow((x2**2).sum(0)) - 2*numpy.dot(x1.T, x2)))
    K = 1
    
    def PlotLinearSVMMinDCF():
        Cs = numpy.logspace(-4,-2, 10) 
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
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0.5)])
                
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
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0.5)])
                
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
            C = 1e-2
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
                        
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear SVM (C = {C}, K ={K}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Linear SVM Balanced (Prior= 0.5, C = {C}, K ={K}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                
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
                                
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear SVM (C = {C}, K ={K}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Linear SVM Balanced (Prior= 0.5, C = {C}, K ={K}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
    
    def PlotQuadraticSVMMinDCF():
        Cs = numpy.logspace(-4,-2, 15) 
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
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0.5)])
                
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
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0.5)])
                
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
            C = 1e-2
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
                    
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0.5)])
                    
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
                    
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Quadratic SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
    
    def PlotRBFSVMMinDCF():
        Cs = numpy.logspace(-4, 0, 10) 
        widths = [1e-3, 1e-1, 10]
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
                    
                    # scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0.5)])
                    
                # minDcfs[0][widx][cidx] += utils.computeMinDCF(scoreLSVM, 0.5, 1, 1, LTE)
                minDcfs[1][widx][cidx] += utils.computeMinDCF(scoreLSVMBal, 0.5, 1, 1, LTE)
                
        # plt.figure()
        # plt.xscale('log')
        # plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        # plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        # plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        # plt.legend()
        # plt.savefig('Figures/SVM/RBFSVM_raw.png')
        
        plt.figure()
        plt.xscale('log')
        plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        plt.legend()
        plt.savefig('Figures/SVM/RBFSVMBal_raw.png')
        
        minDcfs = numpy.zeros((2, len(widths), len(Cs)))
        
        # for cidx in range(len(Cs)):
        #     C = Cs[cidx]
        #     for widx in range(len(widths)):
        #         width = widths[widx]
        #         scoreLSVM = []
        #         scoreLSVMBal = []
        #         print(f'C: {C}')
                
        #         for i in range(len(splits)):
        #             print(f'Split: {i}')
        #             trainSplits = splits[0:i] + splits[i+1:]
        #             LTR = numpy.hstack([trS[1] for trS in trainSplits])
        #             DTE = splits[i][0]
        #             DTR = numpy.hstack([trS[0] for trS in trainSplits])
                   
        #             DTR, DTE = utils.gaussianize(DTR, DTE)
                    
        #             scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0)])
        #             scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0.5)])
                    
        #         minDcfs[0][widx][cidx] += utils.computeMinDCF(scoreLSVM, 0.5, 1, 1, LTE)
        #         minDcfs[1][widx][cidx] += utils.computeMinDCF(scoreLSVMBal, 0.5, 1, 1, LTE)
                
        # plt.figure()
        # plt.xscale('log')
        # plt.plot(Cs, minDcfs[0][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        # plt.plot(Cs, minDcfs[0][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        # plt.plot(Cs, minDcfs[0][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        # plt.legend()
        # plt.savefig('Figures/SVM/RBFSVM_gauss.png')
        
        # plt.figure()
        # plt.xscale('log')
        # plt.plot(Cs, minDcfs[1][0], color='r', label=f'Min DCF (Width = {widths[0]})')
        # plt.plot(Cs, minDcfs[1][1], color='b', label=f'Min DCF (Width = {widths[1]})')
        # plt.plot(Cs, minDcfs[1][2], color='g', label=f'Min DCF (Width = {widths[2]})')
        # plt.legend()
        # plt.savefig('Figures/SVM/RBFSVMBal_gauss.png')
    
    def TrainRBFSVMClassifiers():
        with open('Figures/SVM/RBF_results.txt', 'a') as f:
            utils.multiplePrint('RBF SVM Classifiers — min DCF on the validation set using 3-fold', f)
            
            PCAs = [8, 7]
            C = 1.0
            width = 0.1
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
                        
                    scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0)])
                    scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'RBF SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'RBF SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                
                utils.multiplePrint('\n', f)
                
            # for dims in PCAs:
                    
            #     utils.multiplePrint(f'Gaussian features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
            #     minDcfs = numpy.zeros((2, len(priors)))
            #     scoreLSVM = []
            #     scoreLSVMBal = []
                
            #     for i in range(len(splits)):
            #         print(f'Split: {i}')
            #         trainSplits = splits[0:i] + splits[i+1:]
            #         LTR = numpy.hstack([trS[1] for trS in trainSplits])
            #         DTE = splits[i][0]
            #         DTR = numpy.hstack([trS[0] for trS in trainSplits])
                    
            #         DTR, DTE = utils.gaussianize(DTR, DTE)
                    
            #         if dims != len(features):
            #             P, _, _ = utils.compute_PCA(DTR, dims)
            #             DTR = numpy.dot(P.T, DTR)
            #             DTE = numpy.dot(P.T, DTE)
                        
            #         scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0)])
            #         scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0.5)])
                    
            #     for pidx in range(len(priors)):
            #         prior = priors[pidx]
            #         minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
            #         minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
            #     minDcfs = numpy.around(minDcfs, 3)
            #     utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
            #     utils.multiplePrint(f'RBF SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
            #     utils.multiplePrint(f'RBF SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
    
    print('Linear Plot')
    # PlotLinearSVMMinDCF()
    # TrainLinearSVMClassifiers()
    print('Quad Plot')
    # PlotQuadraticSVMMinDCF()
    # TrainQuadraticSVMClassifiers()
    print('RBF Plot')
    # PlotRBFSVMMinDCF()
    TrainRBFSVMClassifiers()

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
            utils.multiplePrint('GMM Classifiers — min DCF on the validation set using 3-fold', f)
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
                    llrTied = numpy.hstack([llrTied, models.trainGMM(DTR, LTR, DTE, 0.1, 64, covType='Tied')])
                    llrNaiveTied = numpy.hstack([llrNaiveTied, models.trainGMM(DTR, LTR, DTE, 0.1, 64, covType='TiedDiag')])
                
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
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'GMM Full (64 gau):\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}', f)
                utils.multiplePrint(f'GMM Diag (128 gau):\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}', f)
                utils.multiplePrint(f'GMM Tied (512 gau):\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}', f)
                utils.multiplePrint(f'GMM Tied Diag (512 gau):\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}', f)
    
    # PlotGMMMinDCF()
    TrainGMMClassifiers()
    
def ComputeStatsBestModels(DTrain, LTrain):
    
    def BestMVG():
        ## MVG Tied Raw features PCA = 7
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        
        llrTied = []
        calLlrTied = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.linspace(-3, 3, 21)
        dcfs = []
        calDcfs = []
        minDcfs = []
        dims = 7    
    
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
            
            scores = models.compute_tied_llr(DTR, LTR, DTE)
            llrTied = numpy.hstack([llrTied, scores])
            calLlrTied = numpy.hstack([calLlrTied, models.compute_linear_LR_priorW(utils.mrow(scores), splits[i][1], utils.mrow(scores), 1e-3, 0.5)])
        
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrTied, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrTied, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrTied, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrTied, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrTied, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrTied, negPrior, 1, 1, LTE))
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/TiedMVGminDCFVsActualDCF.png')
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/TiedMVGminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Best/results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('MVG Tied Raw features PCA = 7', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'MVG Tied:\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
    
    def BestLR():
        ## Quadratic LR (Prior=0.5, Lambda=1e-05) PCA = 7
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        
        llrQLR = []
        calLlrQLR = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.hstack([numpy.linspace(-3, 3, 21), priors])
        dcfs = []
        minDcfs = []
        calDcfs = []
        dims = 7    
        l = 1e-5
        
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
            
            scores = models.compute_quadratic_LR_priorW(DTR, LTR, DTE, l)
            llrQLR = numpy.hstack([llrQLR, scores])
            calLlrQLR = numpy.hstack([calLlrQLR, models.compute_linear_LR_priorW(utils.mrow(scores), splits[i][1], utils.mrow(scores), 1e-3, 0.5)])
            
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrQLR, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQLR, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQLR, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrQLR, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQLR, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQLR, negPrior, 1, 1, LTE))
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/QuadLRminDCFVsActualDCF.png')
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/QuadLRminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Best/results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('Quadratic LR Raw Features PCA = 7', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'Quadratic LR (Prior=0.5, Lambda=1e-05):\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
    
    def BestSVM():
        ## Quadratic SVM Balanced (Prior= 0.5, C = 0.01) PCA = 7
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        polyKernel = lambda x1, x2: (numpy.dot(x1.T, x2) + 1) ** 2
        K = 1
        C = 1e-2
        llrQSVM = []
        calLlrQSVM = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.hstack([numpy.linspace(-3, 3, 21), priors])
        dcfs = []
        minDcfs = []
        calDcfs = []
        dims = 7
        
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
            
            scores = models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0.5)
            llrQSVM = numpy.hstack([llrQSVM, scores])
            calLlrQSVM = numpy.hstack([calLlrQSVM, models.compute_linear_LR_priorW(utils.mrow(scores), splits[i][1], utils.mrow(scores), 1e-3, 0.5)])
            
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrQSVM, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQSVM, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQSVM, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrQSVM, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQSVM, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQSVM, negPrior, 1, 1, LTE))
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/QuadSVMminDCFVsActualDCF.png')
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/QuadSVMminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Best/results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('Quadratic SVM Raw Features PCA = 7', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'Quadratic SVM Balanced (Prior= 0.5, C = 0.01):\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
    
    
    def BestGMM():
        ## GMM Full Cov 8 components
        splits = utils.K_folds_split(DTrain, LTrain, folds=3)
        LTE = numpy.hstack([s[1] for s in splits])
        
        llrFull = []
        calLlrFull = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.hstack([numpy.linspace(-3, 3, 21), priors])
        dcfs = []
        minDcfs = []
        calDcfs = []
    
        for i in range(len(splits)):
            print(f'Split: {i}')
            trainSplits = splits[0:i] + splits[i+1:]
            LTR = numpy.hstack([trS[1] for trS in trainSplits])
            DTE = splits[i][0]
            DTR = numpy.hstack([trS[0] for trS in trainSplits])
            
            scores = models.trainGMM(DTR, LTR, DTE, 0.1, 8, covType='Full')
            llrFull = numpy.hstack([llrFull, scores])
            calLlrFull = numpy.hstack([calLlrFull, models.compute_linear_LR_priorW(utils.mrow(scores), splits[i][1], utils.mrow(scores), 1e-3, 0.5)])
            
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrFull, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrFull, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrFull, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrFull, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrFull, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrFull, negPrior, 1, 1, LTE))
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/FullCov8CompGMMminDCFVsActualDCF.png')
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Best/FullCov8CompGMMminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Best/results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('GMM Full Cov Raw Features no PCA', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'GMM Full Cov 8 components:\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
    
    
    BestMVG()
    BestLR()
    BestSVM()
    BestGMM()
    
def EvaluateAllModels(DTrain, LTrain, DEval, LEval):
    priors = [0.5, 0.1, 0.9]
    polyKernel = lambda x1, x2: (numpy.dot(x1.T, x2) + 1) ** 2
    rbfKernel = lambda width: lambda x1, x2: numpy.exp(-width*(numpy.linalg.norm(x1-x2)**2)) 
    
    def TrainGaussianClassifiers(DTrain, LTrain, DEval, LEval):
        with open('Figures/Eval/results.txt', 'a') as f:
            utils.multiplePrint('MVG Classifiers — min DCF on the validation set using 3-fold', f)            
            PCAs = [8, 7]
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                    
                llrFull = []
                llrNaive = []
                llrTied = []
                llrNaiveTied = []
                minDcf = numpy.zeros((4, len(priors)))
                DTR = DTrain
                LTR = LTrain
                DTE = DEval
                LTE = LEval
                    
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
    
    def TrainLinearLR(DTrain, LTrain, DEval, LEval):
        with open('Figures/Eval/results.txt', 'a') as f:
            utils.multiplePrint('Linear LR Classifiers — min DCF on the validation set using 3-fold', f)
            PCAs = [8, 7]
            l = 1e-5
            
            for dims in PCAs:                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((len(priors)))
                scoreLLR = []
                DTR = DTrain
                LTR = LTrain
                DTE = DEval
                LTE = LEval
                    
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_linear_LR_priorW(DTR, LTR, DTE, l)])
                        
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[pidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0]}\t{minDcfs[1]}\t{minDcfs[2]}', f)
            
            utils.multiplePrint('\n', f)
    
    def TrainQuadraticLR(DTrain, LTrain, DEval, LEval):
        with open('Figures/Eval/results.txt', 'a') as f:
            utils.multiplePrint('Quadratic LR Classifiers — min DCF on the validation set using 3-fold', f)
            PCAs = [8, 7]
            l = 1e-5
            
            for dims in PCAs:
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((len(priors)))
                scoreLLR = []
                DTR = DTrain
                LTR = LTrain
                DTE = DEval
                LTE = LEval                
                    
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                
                scoreLLR = numpy.hstack([scoreLLR, models.compute_quadratic_LR_priorW(DTR, LTR, DTE, l)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[pidx] += utils.computeMinDCF(scoreLLR, prior, 1, 1, LTE)
                    
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic LR (Prior={priors[0]}, Lambda={l}):\t\t\t\t\t\t{minDcfs[0]}\t{minDcfs[1]}\t{minDcfs[2]}', f)
                
            utils.multiplePrint('\n', f)
    
    def TrainLinearSVMClassifiers(DTrain, LTrain, DEval, LEval):
        with open('Figures/Eval/results.txt', 'a') as f:
            utils.multiplePrint('Linear SVM Classifiers — min DCF on the validation set using 3-fold', f)
            
            PCAs = [8, 7]
            K = 1
            C = 1e-2
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                DTR = DTrain
                LTR = LTrain
                DTE = DEval
                LTE = LEval
                    
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                    
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainLinearSVM(DTR, LTR, K, C, DTE, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Linear SVM (C = {C}, K ={K}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Linear SVM Balanced (Prior= 0.5, C = {C}, K ={K}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                
            utils.multiplePrint('\n', f)
    
    def TrainQuadraticSVMClassifiers(DTrain, LTrain, DEval, LEval):
        with open('Figures/Eval/results.txt', 'a') as f:
            utils.multiplePrint('Quadratic SVM Classifiers — min DCF on the validation set using 3-fold', f)
            
            PCAs = [8, 7]
            K = 1
            C = 1e-2
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                DTR = DTrain
                LTR = LTrain
                DTE = DEval
                LTE = LEval
                    
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'Quadratic SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'Quadratic SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
            
            utils.multiplePrint('\n', f)
    
    def TrainRBFSVMClassifiers(DTrain, LTrain, DEval, LEval):
        with open('Figures/Eval/results.txt', 'a') as f:
            utils.multiplePrint('RBF SVM Classifiers — min DCF on the validation set using 3-fold', f)
            
            PCAs = [8, 7]
            K = 1
            C = 1e-2
            width = 1e1-3
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                minDcfs = numpy.zeros((2, len(priors)))
                scoreLSVM = []
                scoreLSVMBal = []
                DTR = DTrain
                LTR = LTrain
                DTE = DEval
                LTE = LEval
                    
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                    
                scoreLSVM = numpy.hstack([scoreLSVM, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0)])
                scoreLSVMBal = numpy.hstack([scoreLSVMBal, models.trainNonLinearSVM(DTR, LTR, K, C, DTE, rbfKernel(width), 0.5)])
                    
                for pidx in range(len(priors)):
                    prior = priors[pidx]
                    minDcfs[0][pidx] += utils.computeMinDCF(scoreLSVM, prior, 1, 1, LTE)
                    minDcfs[1][pidx] += utils.computeMinDCF(scoreLSVMBal, prior, 1, 1, LTE)
                        
                        
                minDcfs = numpy.around(minDcfs, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'RBF SVM (C = {C}):\t\t\t\t\t\t{minDcfs[0][0]}\t{minDcfs[0][1]}\t{minDcfs[0][2]}', f)
                utils.multiplePrint(f'RBF SVM Balanced (Prior= 0.5, C = {C}):\t\t\t\t\t\t{minDcfs[1][0]}\t{minDcfs[1][1]}\t{minDcfs[1][2]}', f)
                
                utils.multiplePrint('\n', f)
    
    def TrainGMMClassifiers(DTrain, LTrain, DEval, LEval):
        with open('Figures/Eval/results.txt', 'a') as f:
            utils.multiplePrint('GMM Classifiers — min DCF on the test set using 3-fold', f)
            PCAs = [8, 7]
            
            for dims in PCAs:
                    
                utils.multiplePrint(f'Raw features — {"no PCA" if dims == len(features) else "PCA (m = " + str(dims) + ")"}', f)
                    
                llrFull = []
                llrNaive = []
                llrTied = []
                llrNaiveTied = []
                minDcf = numpy.zeros((4, len(priors)))
                DTR = DTrain
                LTR = LTrain
                DTE = DEval
                LTE = LEval
                    
                if dims != len(features):
                    P, _, _ = utils.compute_PCA(DTR, dims)
                    DTR = numpy.dot(P.T, DTR)
                    DTE = numpy.dot(P.T, DTE)
                
                llrFull = numpy.hstack([llrFull, models.trainGMM(DTR, LTR, DTE, 0.1, 8, covType='Full')])
                llrNaive = numpy.hstack([llrNaive, models.trainGMM(DTR, LTR, DTE, 0.1, 16, covType='Diag')])
                llrTied = numpy.hstack([llrTied, models.trainGMM(DTR, LTR, DTE, 0.1, 64, covType='Tied')])
                llrNaiveTied = numpy.hstack([llrNaiveTied, models.trainGMM(DTR, LTR, DTE, 0.1, 64, covType='TiedDiag')])
                
                for p in range(len(priors)):
                    prior = priors[p]
                        
                    minDcf[0][p] += utils.computeMinDCF(llrFull, prior, 1, 1, LTE)
                    minDcf[1][p] += utils.computeMinDCF(llrNaive, prior, 1, 1, LTE)
                    minDcf[2][p] += utils.computeMinDCF(llrTied, prior, 1, 1, LTE)
                    minDcf[3][p] += utils.computeMinDCF(llrNaiveTied, prior, 1, 1, LTE)
            
                minDcf = numpy.around(minDcf, 3)
                utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t{priors[0]}\t\t{priors[1]}\t\t{priors[2]}', f)
                utils.multiplePrint(f'GMM Full (8 gau):\t\t\t\t\t\t{minDcf[0][0]}\t{minDcf[0][1]}\t{minDcf[0][2]}', f)
                utils.multiplePrint(f'GMM Diag (16 gau):\t\t\t\t\t\t{minDcf[1][0]}\t{minDcf[1][1]}\t{minDcf[1][2]}', f)
                utils.multiplePrint(f'GMM Tied (64 gau):\t\t\t\t\t\t{minDcf[2][0]}\t{minDcf[2][1]}\t{minDcf[2][2]}', f)
                utils.multiplePrint(f'GMM Tied Diag (64 gau):\t\t\t\t\t{minDcf[3][0]}\t{minDcf[3][1]}\t{minDcf[3][2]}', f)
                
            utils.multiplePrint('\n', f)
    
    # TrainGaussianClassifiers(DTrain, LTrain, DEval, LEval)
    TrainLinearLR(DTrain, LTrain, DEval, LEval)
    TrainQuadraticLR(DTrain, LTrain, DEval, LEval)
    # TrainLinearSVMClassifiers(DTrain, LTrain, DEval, LEval)
    # TrainQuadraticSVMClassifiers(DTrain, LTrain, DEval, LEval)
    # TrainRBFSVMClassifiers(DTrain, LTrain, DEval, LEval)
    # TrainGMMClassifiers(DTrain, LTrain, DEval, LEval)
    
def ComputeStatsBestModelsAfterEval(DTrain, LTrain, DEval, LEval):
    def BestMVG():
        ## MVG Tied Raw features PCA = 7
        
        llrTied = []
        calLlrTied = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.linspace(-3, 3, 21)
        dcfs = []
        calDcfs = []
        minDcfs = []
        dims = 7
        DTR = DTrain
        LTR = LTrain
        DTE = DEval
        LTE = LEval
            
        if dims != len(features):
            P, _, _ = utils.compute_PCA(DTR, dims)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
        
        scores = models.compute_tied_llr(DTR, LTR, DTE)
        llrTied = numpy.hstack([llrTied, scores])
        calLlrTied = numpy.hstack([calLlrTied, models.compute_linear_LR_priorW(utils.mrow(models.compute_tied_llr(DTR, LTR, DTR)), LTR, utils.mrow(scores), 1e-3, 0.5)])
    
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrTied, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrTied, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrTied, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrTied, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrTied, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrTied, negPrior, 1, 1, LTE))
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/TiedMVGminDCFVsActualDCF.png')
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/TiedMVGminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Eval/best_results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('MVG Tied Raw features PCA = 7', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'MVG Tied:\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
        
        return utils.computeParametersForRoc(llrTied, LTE)
    
    def BestLR():
        ## Quadratic LR (Prior=0.5, Lambda=1e-05) PCA = 7
        
        llrQLR = []
        calLlrQLR = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.hstack([numpy.linspace(-3, 3, 21), priors])
        dcfs = []
        minDcfs = []
        calDcfs = []
        dims = 7    
        l = 1e-5
        DTR = DTrain
        LTR = LTrain
        DTE = DEval
        LTE = LEval
        
            
        if dims != len(features):
            P, _, _ = utils.compute_PCA(DTR, dims)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
        
        scores = models.compute_quadratic_LR_priorW(DTR, LTR, DTE, l)
        llrQLR = numpy.hstack([llrQLR, scores])
        calLlrQLR = numpy.hstack([calLlrQLR, models.compute_linear_LR_priorW(utils.mrow(models.compute_quadratic_LR_priorW(DTR, LTR, DTR, l)), LTR, utils.mrow(scores), 1e-3, 0.5)])
            
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrQLR, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQLR, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQLR, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrQLR, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQLR, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQLR, negPrior, 1, 1, LTE))
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/QuadLRminDCFVsActualDCF.png')
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/QuadLRminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Eval/best_results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('Quadratic LR Raw Features PCA = 7', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'Quadratic LR (Prior=0.5, Lambda=1e-05):\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
        
        return utils.computeParametersForRoc(llrQLR, LTE)
    
    def BestSVM():
        ## Quadratic SVM Balanced (Prior= 0.5, C = 0.01) PCA = 7
        polyKernel = lambda x1, x2: (numpy.dot(x1.T, x2) + 1) ** 2
        K = 1
        llrQSVM = []
        calLlrQSVM = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.hstack([numpy.linspace(-3, 3, 21), priors])
        dcfs = []
        minDcfs = []
        calDcfs = []
        dims = 7    
        C = 1e-2
        DTR = DTrain
        LTR = LTrain
        DTE = DEval
        LTE = LEval
        
            
        if dims != len(features):
            P, _, _ = utils.compute_PCA(DTR, dims)
            DTR = numpy.dot(P.T, DTR)
            DTE = numpy.dot(P.T, DTE)
        
        scores = models.trainNonLinearSVM(DTR, LTR, K, C, DTE, polyKernel, 0.5)
        llrQSVM = numpy.hstack([llrQSVM, scores])
        calLlrQSVM = numpy.hstack([calLlrQSVM, models.compute_linear_LR_priorW(utils.mrow(models.trainNonLinearSVM(DTR, LTR, K, C, DTR, polyKernel, 0.5)), LTR, utils.mrow(scores), 1e-3, 0.5)])
            
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrQSVM, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQSVM, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQSVM, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrQSVM, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrQSVM, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrQSVM, negPrior, 1, 1, LTE))
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/QuadSVMminDCFVsActualDCF.png')
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/QuadSVMminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Eval/best_results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('Quadratic SVM Raw Features PCA = 7', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'Quadratic SVM Balanced (Prior= 0.5, C = 0.01):\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
        
        return utils.computeParametersForRoc(llrQSVM, LTE)
    
    def BestGMM():
        ## GMM Full Cov 8 components
        
        llrFull = []
        calLlrFull = []
        priors = numpy.array([0.5, 0.1, 0.9])
        effPriorLogOdds = numpy.hstack([numpy.linspace(-3, 3, 21), priors])
        dcfs = []
        minDcfs = []
        calDcfs = []
        DTR = DTrain
        LTR = LTrain
        DTE = DEval
        LTE = LEval
            
        scores = models.trainGMM(DTR, LTR, DTE, 0.1, 8, covType='Full')
        llrFull = numpy.hstack([llrFull, scores])
        calLlrFull = numpy.hstack([calLlrFull, models.compute_linear_LR_priorW(utils.mrow(models.trainGMM(DTR, LTR, DTR, 0.1, 8, covType='Full')), LTR, utils.mrow(scores), 1e-3, 0.5)])
        
        for eplo in effPriorLogOdds:
            negPrior = 1 / (1 + numpy.exp(-eplo))
            dcfs.append(utils.computeDCF(llrFull, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrFull, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrFull, negPrior, 1, 1, LTE))
        
        for negPrior in priors:
            dcfs.append(utils.computeDCF(llrFull, negPrior, 1, 1, LTE)[1])
            calDcfs.append(utils.computeDCF(calLlrFull, negPrior, 1, 1, LTE)[1])
            minDcfs.append(utils.computeMinDCF(llrFull, negPrior, 1, 1, LTE))
                    
        plt.figure()
        plt.plot(effPriorLogOdds[:21], dcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/FullCov8CompGMMminDCFVsActualDCF.png')
        
        plt.figure()
        plt.plot(effPriorLogOdds[:21], calDcfs[:21], label='DCF', color='r')
        plt.plot(effPriorLogOdds[:21], minDcfs[:21], label='min DCF', color='b', linestyle='dashed')
        plt.legend()
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.savefig('Figures/Eval/FullCov8CompGMMminDCFVsActualDCFCalibrated.png')
        
        with open('Figures/Eval/best_results.txt', 'a') as f:
            minDcfs = numpy.around(minDcfs[-3::], 3)
            dcfs = numpy.around(dcfs[-3::], 3)
            calDcfs = numpy.around(calDcfs[-3::], 3)
            utils.multiplePrint('GMM Full Cov Raw Features no PCA', f)
            utils.multiplePrint(f'Priors:\t\t\t\t\t\t\t\t\t\t\t\t\t\t{priors[0]}\t\t\t\t{priors[1]}\t\t\t\t{priors[2]}', f)
            utils.multiplePrint(f'GMM Full Cov 8 components:\t\t\t\t\t\t{minDcfs[0]} {dcfs[0]} {calDcfs[0]}\t{minDcfs[1]} {dcfs[1]} {calDcfs[1]}\t{minDcfs[2]} {dcfs[2]} {calDcfs[2]}', f)
        
        return utils.computeParametersForRoc(llrFull, LTE)
    
    TprMVG, FprMVG = BestMVG()
    TprLR, FprLR = BestLR()
    TprSVM, FprSVM = BestSVM()
    TprGMM, FprGMM = BestGMM()
    
    plt.figure()
    plt.plot(FprMVG, TprMVG, label='MVG Tied', color='r')
    plt.plot(FprLR, TprLR, label='Quad LR', color='b')
    plt.plot(FprSVM, TprSVM, label='Quad SVM', color='y')
    plt.plot(FprGMM, TprGMM, label='GMM Full 8 comp', color='g')
    plt.legend()
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('Figures/Eval/ROC.png')
    
    FnrMVG = 1 - numpy.array(TprMVG)
    FnrLR = 1 - numpy.array(TprLR)
    FnrGMM = 1 - numpy.array(TprGMM)
    FnrSVM = 1 - numpy.array(TprSVM)
    
    plt.figure()
    plt.plot(FprMVG, FnrMVG, label='MVG Tied', color='r')
    plt.plot(FprLR, FnrLR, label='Quad LR', color='b')
    plt.plot(FprSVM, FnrSVM, label='Quad SVM', color='y')
    plt.plot(FprGMM, FnrGMM, label='GMM Full 8 comp', color='g')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('FNR')
    plt.savefig('Figures/Eval/DET.png')
    

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
    
    # ComputeStatsBestModels(DTR_n, LTrain)
    # EvaluateAllModels(DTR_n, LTrain, DEV_n, LEval)
    # ComputeStatsBestModelsAfterEval(DTR_n, LTrain, DEV_n, LEval)
