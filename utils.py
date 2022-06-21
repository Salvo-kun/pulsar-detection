# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:36:28 2022

@author: Salvo
"""

import numpy
import scipy.stats

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def compute_mean(D):
    return mcol(D.mean(1 if D.ndim == 2 else 0))

def compute_cov(X, Y = None):
    if Y is None:
        return numpy.dot(center_data(X), center_data(X).T)/X.shape[1 if X.ndim == 2 else 0]
    else:
        return numpy.dot(center_data(X), center_data(Y).T)/X.shape[1 if X.ndim == 2 else 0]

def center_data(D):
    return D - compute_mean(D)

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
            pearsonCorrelationMatrix[i][j] += abs(compute_cov(DTR[i], DTR[j])/(compute_cov(DTR[i]) ** 0.5 * compute_cov(DTR[j]) ** 0.5))
        
    return pearsonCorrelationMatrix

def compute_PCA(C, D, L, m):    
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, D)
    
    return DP

def apply_PCA(D, L, m):
    DC = center_data(D)
    C = (numpy.dot(DC, DC.T))/D.shape[1]
    return compute_PCA(C, D, L, m)

def K_folds_split(dataset, labels, folds=3):
    dataset_split = []
    fold_size = int(dataset.shape[1] / folds)
    idxs = numpy.random.permutation(dataset.shape[1])
    for i in range(folds):
        if idxs.shape[0] < 2*fold_size:
            idx = idxs[::]
        else:
            idx = idxs[:fold_size]
        dataset_split.append((dataset[:, idx], labels[idx]))
        idxs = numpy.delete(idxs, [range(fold_size)])
    return dataset_split

def computeConfusionMatrix(predictions, labels):
    C = numpy.zeros((2, 2))
    
    C[0, 0] = ((predictions == 0) * (labels == 0)).sum()
    C[0, 1] = ((predictions == 0) * (labels == 1)).sum()
    C[1, 0] = ((predictions == 1) * (labels == 0)).sum()
    C[1, 1] = ((predictions == 1) * (labels == 1)).sum()
    
    return C

def computeMinDCF(ll_ratios, negPrior, Cfn, Cfp, labels):
    sorted_lls = numpy.array(ll_ratios)
    sorted_lls.sort()
    ths = numpy.concatenate([numpy.array([-numpy.inf]), sorted_lls, numpy.array([numpy.inf])])
    dcfs = []
    
    for th in ths:
        dcf, dcf_norm = computeDCF(ll_ratios, negPrior, Cfn, Cfp, labels, th)
        dcfs.append(dcf_norm)
    
    return numpy.array(dcfs).min()

def computeBinaryOptimalBayesDecisions(ll_ratios, negPrior, Cfn, Cfp, th = None):
    if th is None:
        th = numpy.log((1 - negPrior) * Cfp) - numpy.log(negPrior * Cfn)
    predictions = numpy.where(ll_ratios > th, 1, 0)
    return predictions

def computeFPR(FP, TN):
    return FP/(FP + TN)

def computeFNR(FN, TP):
    return FN/(FN + TP)

def computeTPR(FN, TP):
    return 1 - computeFNR(FN, TP)

def computeDCF(ll_ratios, negPrior, Cfn, Cfp, labels, th = None):
    predictions = computeBinaryOptimalBayesDecisions(ll_ratios, negPrior, Cfn, Cfp, th)
    confMatrix = computeConfusionMatrix(predictions, labels)
    dummyDCF = numpy.min([negPrior * Cfn, (1 - negPrior) * Cfp])
    FPR = computeFPR(confMatrix[1][0], confMatrix[0][0])
    FNR = computeFNR(confMatrix[0][1], confMatrix[1][1])
    DCF = negPrior * Cfn * FNR + (1 - negPrior) * Cfp * FPR
    
    return DCF, DCF/dummyDCF

def computeParametersForRoc(ll_ratios, labels):
    sorted_lls = numpy.array(ll_ratios)
    sorted_lls.sort()
    ths = numpy.concatenate([numpy.array([-numpy.inf]), sorted_lls, numpy.array([numpy.inf])])
    TPRs = []
    FPRs = []
    
    for th in ths:
        predictions = computeBinaryOptimalBayesDecisions(ll_ratios, None, None, None, th)
        confMatrix = computeConfusionMatrix(predictions, labels)
        TPR = computeTPR(confMatrix[0][1], confMatrix[1][1])
        TPRs.append(TPR)
        FPR = computeFPR(confMatrix[1][0], confMatrix[0][0])
        FPRs.append(FPR)

    return TPRs, FPRs