# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:22:41 2022

@author: Salvo
"""

import numpy
import scipy
import utils

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

def compute_linear_LR(DTR, LTR, DTE, l, p):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad = True)
    w, b = utils.mcol(x[0:-1]), x[-1]
    scores = numpy.dot(w.T, DTE) + b - numpy.log(p/(1-p))
    
    return scores[0, :]

def compute_quadratic_LR(DTR, LTR, DTE, l, p):
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad = True)
    w, b = utils.mcol(x[0:-1]), x[-1]
    scores = numpy.dot(w.T, DTE) + b - numpy.log(p/(1-p))
    return scores
