# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:36:28 2022

@author: Salvo
"""

import numpy

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

def compute_PCA(D, dims):
    DC = center_data(D)
    C = (numpy.dot(DC, DC.T))/D.shape[1]
    s, U = numpy.linalg.eigh(C)
    return U[:, ::-1][:, 0:dims], s[::-1], DC

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

def computeConfusionMatrix(predictions, labels, classesDict):
    numClasses = len(classesDict)
    matrix = numpy.zeros((numClasses, numClasses))
    
    for i in range(len(labels)):
        matrix[predictions[i]][labels[i]] += 1
    
    return matrix

def computeMinDCF(ll_ratios, negPrior, Cfn, Cfp, labels, classesDict):
    sorted_lls = numpy.array(ll_ratios)
    sorted_lls.sort()
    ths = numpy.concatenate([numpy.array([-numpy.inf]), sorted_lls, numpy.array([numpy.inf])])
    dcfs = []
    
    for th in ths:
        dcf, dcf_norm = computeDCF(ll_ratios, negPrior, Cfn, Cfp, labels, classesDict, th)
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

def computeDCF(ll_ratios, negPrior, Cfn, Cfp, labels, classesDict, th = None):
    predictions = computeBinaryOptimalBayesDecisions(ll_ratios, negPrior, Cfn, Cfp, th)
    confMatrix = computeConfusionMatrix(predictions, labels, classesDict)
    dummyDCF = numpy.min([negPrior * Cfn, (1 - negPrior) * Cfp])
    FPR = computeFPR(confMatrix[1][0], confMatrix[0][0])
    FNR = computeFNR(confMatrix[0][1], confMatrix[1][1])
    DCF = negPrior * Cfn * FNR + (1 - negPrior) * Cfp * FPR
    
    return DCF, DCF/dummyDCF

def computeParametersForRoc(ll_ratios, labels, classesDict):
    sorted_lls = numpy.array(ll_ratios)
    sorted_lls.sort()
    ths = numpy.concatenate([numpy.array([-numpy.inf]), sorted_lls, numpy.array([numpy.inf])])
    TPRs = []
    FPRs = []
    
    for th in ths:
        predictions = computeBinaryOptimalBayesDecisions(ll_ratios, None, None, None, th)
        confMatrix = computeConfusionMatrix(predictions, labels, classesDict)
        TPR = computeTPR(confMatrix[0][1], confMatrix[1][1])
        TPRs.append(TPR)
        FPR = computeFPR(confMatrix[1][0], confMatrix[0][0])
        FPRs.append(FPR)

    return TPRs, FPRs