# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:22:41 2022

@author: Salvo
"""

import numpy
import scipy
import utils
import time

def logpdf_GAU_ND(X, mu, C):
    P = numpy.linalg.inv(C)
    res = - 0.5 * X.shape[0] * numpy.log(2 * numpy.pi)
    res += - 0.5 * numpy.linalg.slogdet(C)[1]

    # We do not use numpy.dot because we need element-wise multiplication between tewo vectors, since we are interested in computing
    # the pdf for each sample feature.
    # We use sum(0), i.e. we collapse into one row, because if we have many samples, we still want to obtain a unique pdf
    # for each feature of the vector
    res += - 0.5 * ((X - mu) * numpy.dot(P, X - mu)).sum(0)
    return res


def loglikelihood(X, m_ML, C_ML):
    return logpdf_GAU_ND(X, m_ML, C_ML).sum(0)


def compute_estimates(D, L):
    res = {}
    for i in range(2):
        Di = D[:, L == i]
        mu = utils.compute_mean(Di)
        sigma = utils.compute_cov(Di)
        res[i] = (mu, sigma)
    return res

def compute_tied_estimates(D, L):
    means = {}
    sigma = 0.0
    for i in range(2):
        Di = D[:, L == i]
        mu = utils.compute_mean(Di)
        sigma += utils.compute_cov(Di)*Di.shape[1]
        means[i] = mu
    sigma /= D.shape[1]
    return means, sigma


def compute_full_llr(DTR, LTR, DTE):
    logSj = numpy.zeros((2, DTE.shape[1]))
    estimates = compute_estimates(DTR, LTR)

    for i in range(2):
        mu, C = estimates[i]
        logSj[i, :] = logpdf_GAU_ND(DTE, mu, C)

    return logSj[1, :] - logSj[0, :]

def compute_naive_llr(DTR, LTR, DTE):
    logSj = numpy.zeros((2, DTE.shape[1]))
    estimates = compute_estimates(DTR, LTR)

    for i in range(2):
        mu, C = estimates[i]
        C = C * numpy.eye(C.shape[0])
        logSj[i, :] = logpdf_GAU_ND(DTE, mu, C)

    return logSj[1, :] - logSj[0, :]

def compute_tied_llr(DTR, LTR, DTE):
    logSj = numpy.zeros((2, DTE.shape[1]))
    means, C = compute_tied_estimates(DTR, LTR)

    for i in range(2):
        mu = means[i]
        logSj[i, :] = logpdf_GAU_ND(DTE, mu, C) 

    return logSj[1, :] - logSj[0, :]


def compute_tied_naive_bayes_llr(DTR, LTR, DTE):
    logSj = numpy.zeros((2, DTE.shape[1]))
    means, C = compute_tied_estimates(DTR, LTR)
    C = C * numpy.eye(C.shape[0])

    for i in range(2):
        mu = means[i]
        logSj[i, :] = logpdf_GAU_ND(DTE, mu, C)

    return logSj[1, :] - logSj[0, :]

def logreg_obj_wrap(DTR, LTR, l):
    z = LTR * 2.0 - 1.0
    def logreg_obj(v):
        w, b = utils.mcol(v[0:-1]), v[-1]
        reg = 0.5 * l * numpy.linalg.norm(w) ** 2
        s = numpy.dot(w.T, DTR) + b 
        avg_risk = (numpy.logaddexp(0, -s*z)).mean()
        return reg + avg_risk
    return logreg_obj

def compute_linear_LR(DTR, LTR, DTE, l, p = None):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    t = time.time()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad = True, factr=1.0)
    print(f'Elapsed {time.time() - t} seconds')
    print(x)
    w, b = utils.mcol(x[0:-1]), x[-1]
    calibration = 0 if p == None else numpy.log(p/(1-p))
    scores = numpy.dot(w.T, DTE) + b - calibration
    
    return scores[0, :]

def logreg_obj_wrap_priorW(DTR, LTR, l, p):
    z = LTR * 2.0 - 1.0
    def logreg_obj(v):
        w, b = utils.mcol(v[0:-1]), v[-1]
        reg = 0.5 * l * numpy.linalg.norm(w) ** 2
        s = (numpy.dot(w.T, DTR) + b).ravel()
        avg_risk_0 = (numpy.logaddexp(0, -s[LTR == 0]*z[LTR == 0])).mean()*(1-p)
        avg_risk_1 = (numpy.logaddexp(0, -s[LTR == 1]*z[LTR == 1])).mean()*p
        return reg + avg_risk_1 + avg_risk_0
    return logreg_obj

def compute_linear_LR_priorW(DTR, LTR, DTE, l, p = 0.5):
    logreg_obj = logreg_obj_wrap_priorW(DTR, LTR, l, p)
    t = time.time()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad = True, factr=1.0)
    print(f'Elapsed {time.time() - t} seconds')
    print(x)
    w, b = utils.mcol(x[0:-1]), x[-1]
    calibration = 0 if p == None else numpy.log(p/(1-p))
    scores = numpy.dot(w.T, DTE) + b - calibration
    
    return scores[0, :]

def expandFeatures(x):
    x = utils.mcol(x)
    expX = utils.mcol(numpy.dot(x, x.T))
    return numpy.vstack([expX, x])

def compute_quadratic_LR(DTR, LTR, DTE, l, p = None):
    DTR_ext = numpy.hstack([expandFeatures(DTR[:, i]) for i in range(DTR.shape[1])])
    DTE_ext = numpy.hstack([expandFeatures(DTE[:, i]) for i in range(DTE.shape[1])])
    logreg_quad_obj = logreg_obj_wrap(DTR_ext, LTR, l)
    t = time.time()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_quad_obj, numpy.zeros(DTR_ext.shape[0] + 1), approx_grad = True, factr=1.0)
    print(f'Elapsed {time.time() - t} seconds')
    w, b = utils.mcol(x[0:-1]), x[-1]
    calibration = 0 if p == None else numpy.log(p/(1-p))
    scores = numpy.dot(w.T, DTE_ext) + b - calibration
    
    return scores[0, :]

def compute_quadratic_LR_priorW(DTR, LTR, DTE, l, p = 0.5):
    DTR_ext = numpy.hstack([expandFeatures(DTR[:, i]) for i in range(DTR.shape[1])])
    DTE_ext = numpy.hstack([expandFeatures(DTE[:, i]) for i in range(DTE.shape[1])])
    logreg_quad_obj = logreg_obj_wrap_priorW(DTR_ext, LTR, l, p)
    t = time.time()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_quad_obj, numpy.zeros(DTR_ext.shape[0] + 1), approx_grad = True, factr=1.0)
    print(f'Elapsed {time.time() - t} seconds')
    w, b = utils.mcol(x[0:-1]), x[-1]
    calibration = 0 if p == None else numpy.log(p/(1-p))
    scores = numpy.dot(w.T, DTE_ext) + b - calibration
    
    return scores[0, :]

def trainLinearSVM(DTR, LTR, K, C, DTE, p = 0):
    Z = LTR * 2.0 - 1.0
    X_hat = numpy.vstack([DTR, K * numpy.ones((1, DTR.shape[1]))])
    G = numpy.dot(X_hat.T, X_hat)
    H_hat = utils.mcol(Z) * utils.mrow(Z) * G
    empP = (LTR == 1).sum()/len(LTR)
    alphaBounds = numpy.array([(0, C)] * LTR.shape[0])
    
    if p != 0:
        alphaBounds[LTR == 1] = (0, C*p/empP)
        alphaBounds[LTR == 0] = (0, C*(1-p)/(1-empP))
    
    def computeDualLoss(alpha):   
        return 0.5 * numpy.dot(numpy.dot(utils.mrow(alpha), H_hat), alpha) - alpha.sum(), numpy.dot(H_hat, alpha) - 1
        
    def computePrimalFromDual(alpha):
        w_hat = numpy.dot(alpha, (Z * X_hat).T)
        w = w_hat[:-1]
        b = w_hat[-1::]                
        return w_hat, w, b
    
    def computeSVMScore(w, b):
        return numpy.dot(w.T, DTE) + b*K
    
    t = time.time()
    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(
        computeDualLoss, 
        numpy.zeros(DTR.shape[1]), 
        bounds = alphaBounds, 
        factr=1.0,
        maxfun=100000,
        maxiter=100000)

    w_hat, w, b = computePrimalFromDual(alphaStar)    
    score = computeSVMScore(w, b)
    print(f'Elapsed {time.time() - t} seconds')

    return score
    
def trainNonLinearSVM(DTR, LTR, K, C, DTE, kernel, p = 0):
    Z = LTR * 2.0 - 1.0
    G = kernel(DTR, DTR) + K ** 2
    H_hat = utils.mcol(Z) * utils.mrow(Z) * G
    empP = (LTR == 1).sum()/len(LTR)
    alphaBounds = numpy.array([(0, C)] * LTR.shape[0])
    
    if p != 0:
        alphaBounds[LTR == 1] = (0, C*p/empP)
        alphaBounds[LTR == 0] = (0, C*(1-p)/(1-empP))
        
    def computeDualLoss(alpha):
        return 0.5 * numpy.dot(numpy.dot(utils.mrow(alpha), H_hat), alpha) - alpha.sum(), numpy.dot(H_hat, alpha) - 1

    def computeSVMScore(alpha):
        score = numpy.zeros(DTE.shape[1])
        # t = time.time()

        for j in range(DTE.shape[1]):
            # print(f'{j} - Elapsed {time.time() - t} seconds')
            for i in range(DTR.shape[1]):
                if alpha[i] > 0:
                    score[j] += alpha[i] * Z[i] * (kernel(DTR[:, i], DTE[:, j]) + K**2)
        # print(f'Elapsed {time.time() - t} seconds')
        # print(score)
        return score
    
    # def computeSVMScoreFast(alpha):
    #     score = numpy.zeros(DTE.shape[1])
    #     t = time.time()

    #     for j in range(DTE.shape[1]):
    #         print(f'{j} - Elapsed {time.time() - t} seconds')
    #         score[j] += (numpy.where(alpha > 0, alpha, 0) * Z * (kernel(DTR, DTE[:, j]) + (K**2))).sum()
    #     print(score)
    #     return score
    
    t = time.time()
    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(
        computeDualLoss, 
        numpy.zeros(DTR.shape[1]), 
        bounds = alphaBounds, 
        factr=1.0,
        maxfun=100000,
        maxiter=100000)
    
    # score = computeSVMScoreFast(alphaStar)
    score = computeSVMScore(alphaStar)
    # print(score - computeSVMScoreFast(alphaStar))
    print(f'Elapsed {time.time() - t} seconds')
    return score

def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    
    for g in range(len(gmm)):
        (w, mu, C) = gmm[g]
        S[g, :] = logpdf_GAU_ND(X, mu, C) + numpy.log(w)
        
    logdens = scipy.special.logsumexp(S, axis=0)
    
    return S, logdens

def GMM_EM(X, gmm, psi = 0.01, covType = 'Full'):
    thNew = None
    thOld = None
    N = X.shape[1]
    D = X.shape[0]
    
    while thOld == None or thNew - thOld > 1e-6:
        thOld = thNew
        logSj, logSjMarg = logpdf_GMM(X, gmm)
        thNew = logSjMarg.sum()/N
        
        P = numpy.exp(logSj - logSjMarg)
        
        if covType == 'Diag':
            newGmm = []
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (utils.mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (utils.mrow(gamma)*X).T)
                w = Z/N
                mu = utils.mcol(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                sigma *= numpy.eye(sigma.shape[0])
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, utils.mcol(s)*U.T)
                newGmm.append((w, mu, sigma))
            gmm = newGmm
        
        elif covType == 'Tied':
            newGmm = []
            sigmaTied = numpy.zeros((D, D))
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (utils.mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (utils.mrow(gamma)*X).T)
                w = Z/N
                mu = utils.mcol(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                sigmaTied += Z * sigma
                newGmm.append((w, mu))   
            gmm = newGmm
            sigmaTied /= N
            U, s, _ = numpy.linalg.svd(sigmaTied)
            s[s<psi] = psi
            sigmaTied = numpy.dot(U, utils.mcol(s)*U.T)
            
            newGmm = []
            for i in range(len(gmm)):
                (w, mu) = gmm[i]
                newGmm.append((w, mu, sigmaTied))
            
            gmm = newGmm
            
        elif covType == 'TiedDiag':
            newGmm = []
            sigmaTied = numpy.zeros((D, D))
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (utils.mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (utils.mrow(gamma)*X).T)
                w = Z/N
                mu = utils.mcol(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                sigmaTied += Z * sigma
                newGmm.append((w, mu))   
            gmm = newGmm
            sigmaTied /= N
            sigmaTied *= numpy.eye(sigma.shape[0])
            U, s, _ = numpy.linalg.svd(sigmaTied)
            s[s<psi] = psi
            sigmaTied = numpy.dot(U, utils.mcol(s)*U.T)
            
            newGmm = []
            for i in range(len(gmm)):
                (w, mu) = gmm[i]
                newGmm.append((w, mu, sigmaTied))
            
            gmm = newGmm
            
        else:
            newGmm = []
            for i in range(len(gmm)):
                gamma = P[i, :]
                Z = gamma.sum()
                F = (utils.mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (utils.mrow(gamma)*X).T)
                w = Z/N
                mu = utils.mcol(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, utils.mcol(s)*U.T)
                newGmm.append((w, mu, sigma))
            gmm = newGmm
        
        # print(f'LL {"Improved" if thOld == None or thNew >= thOld else "Worsened"}: {thNew}')
    
    return gmm

def GMM_LBG(X, alpha, nComponents, psi = 0.01, covType = 'Full'):
    gmm = [(1, utils.compute_mean(X), utils.compute_cov(X))]
    
    while len(gmm) <= nComponents:
        # print(f'\nGMM has {len(gmm)} components')
        gmm = GMM_EM(X, gmm, psi, covType)
                
        if len(gmm) == nComponents:
            break
        
        newGmm = []
        for i in range(len(gmm)):
            (w, mu, sigma) = gmm[i]
            U, s, Vh = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            newGmm.append((w/2, mu + d, sigma))
            newGmm.append((w/2, mu - d, sigma))
        gmm = newGmm
            
    return gmm

def trainGMM(DTR, LTR, DTE, alpha, nComponents, psi = 0.01, covType = 'Full'):
    DTR_0 = DTR[:, LTR == 0]
    gmm_c0 = GMM_LBG(DTR_0, alpha, nComponents, psi, covType)
    _, llr_0 = logpdf_GMM(DTE, gmm_c0)
    
    DTR_1 = DTR[:, LTR == 1]
    gmm_c1 = GMM_LBG(DTR_1, alpha, nComponents, psi, covType)
    _, llr_1 = logpdf_GMM(DTE, gmm_c1)

    
    return llr_1 - llr_0
    