# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: gmm
"""

from __future__ import print_function
import numpy as np
import os
import galsim

import numpy as np
from scipy import linalg

EPS = np.finfo(float).eps
min_covar=1.e-7

def logsumexp(arr, axis=0):
    """Computes the sum of arr assuming arr is in the log domain.

    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.utils.extmath import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    arr = np.rollaxis(arr, axis)
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    vmax = arr.max(axis=0)
    out = np.log(np.sum(np.exp(arr - vmax), axis=0))
    out += vmax
    return out

def log_multivariate_normal_density_manydims(X, means, covars):
    """Log probability for full covariance matrices.
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob

def log_multivariate_normal_density(X, means, covars):
    """Log probability for full covariance matrices.
    Do for just two dimensions (image data)
    """
    n_samples, n_dim = X.shape
    nmix = len(means)
    cv_det = covars[:,0,0] * covars[:,1,1] - covars[:,0,1] * covars[:,1,0]
    covars_inverse = np.zeros(covars.shape)
    covars_inverse[:,0,0] = covars[:,1,1]
    covars_inverse[:,1,1] = covars[:,0,0]
    covars_inverse[:,0,1] = -covars[:,1,0]
    covars_inverse[:,1,0] = -covars[:,0,1]
    covars_inverse /= cv_det[:,np.newaxis,np.newaxis]

    cv_log_det = np.log(cv_det)

    mu = X[:, np.newaxis, :] - means
    cv_log_expterm = np.einsum('ijk,jkl,ijl->ij', mu, covars_inverse, mu)

    log_prob = -0.5 * (n_dim * np.log(2 * np.pi) + cv_log_det + cv_log_expterm)

    return log_prob

def do_lpr(X, means, covars, weights):

    lpr = (log_multivariate_normal_density(X, means, covars,)
           + np.log(weights))

    return lpr

def do_estep(X, means, covars, weights):

    lpr = do_lpr(X, means, covars, weights)
    logprob = logsumexp(lpr, axis=1)
    responsibilities = np.exp(lpr - logprob[:, np.newaxis])

    return logprob, responsibilities

def do_mstep(X, I, responsibilities):
    """ Perform the Mstep of the EM algorithm and return the class weights.

    X is something like X[0] = [-100, -100]
    I is something like I[0] = 500

    """
    tau_ij__I_i = responsibilities * I[:, np.newaxis]
    inverse_weighted = tau_ij__I_i.sum(axis=0) + 10 * EPS
    sum_I_i = tau_ij__I_i.sum()

    weights = tau_ij__I_i.sum(axis=0)
    means = np.dot(tau_ij__I_i.T, X)
    means /= inverse_weighted[:, np.newaxis]
    # dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    # tauI is [i, k]
    # X is [i, j] -> Xp[k, i, j]
    # means is [k, j] -> mup[k, i, j]
    # xmu := (xi - muk)(xi - muk)^T should be [k, i, j1, j2]
    # then cov = np.dot(tauI.T, xmu)
    # cov is [k, j1, j2]
    Xp = X[np.newaxis, :, :]
    meansp = means[:, np.newaxis, :]
    xmup = Xp - meansp
    xmu = xmup[:,:,:,np.newaxis] * xmup[:,:,np.newaxis,:]
    # this is way slow dammit
    covars = (tau_ij__I_i.T[:,:,np.newaxis,np.newaxis] * xmu).sum(axis=1)
    covars /= inverse_weighted[:, np.newaxis, np.newaxis]
    covars += min_covar * np.eye(X.shape[1])

    return means, covars, weights

def EMGMM(star, n_gaussian=1, return_all=False, maxiter=100, tol=1e-04, logger=None):
    """Use expectation maximization to reduce a star's image to a mixture of gaussians

    Given image I_i represented as

        I_i         = sum_k \rho^k N(x_i | \mu^k, \Sigma^k)
                    = \sum_k \frac{\rho^k}{\sqrt{2 \pi \det \Sigma^k}} \exp{-\frac{1}{2} (x_i - \mu^k)^T (\Sigma^k)^{-1} (x_i - \mu^k)}

    E step:

        \tau_{ik}   = \frac{\rho^k N(x_i | \mu^k, \Sigma^k)}{\sum_\ell \rho^\ell N(x_i | \mu^\ell, \Sigma^\ell)}
                    = fraction of pixel i's value attributable to Gaussian k

    M step:

        \rho^k      = \frac{\sum_i \tau_{ik} I_i}{\sum_i I_i}
                    = fraction of flux in kth

        \mu^k       = \frac{\sum_i x_i \tau_{ik} I_i}{\sum_i \tau_{ik} I_i}
                    = centroid of kth component

        \Sigma^k    = \frac{\sum_i (x_i - \mu^k)(x_i - \mu^k)^T \tau_{ik} I_i}{\sum_i \tau_{ik} I_i}
                    = cov of kth component (note: mu^k is the one _just_ calculated!)

    Returns a list of gaussian profiles. Note: pixel models implicitly have accounted for pixel convolution, but the gaussians here do NOT convolve with a pixel when drawing them. You might think of this as (model convolved with pixel) = GMM

    TODO: add logger information
    """

    # extract I_i and x_i from star, where x_i are the u, v coordinates
    I, wt, u, v = star.data.getDataVector()
    X = np.vstack((u, v)).T

    # initial guess of parameters
    weights = np.random.random(n_gaussian)
    weights = weights * I.sum() / weights.sum()  # weight by
    means = np.ones((n_gaussian, 2)) * np.median(X, axis=0)[None]
    covars = np.zeros((n_gaussian, 2, 2))
    covars[:, 0, 0] = np.random.random(n_gaussian) + 0.2
    covars[:, 1, 1] = np.random.random(n_gaussian) + 0.2

    log_likelihood = []
    logprobs = []
    # iterate until sufficiently pleased
    for i in range(maxiter):
        # Expectation step
        curr_log_likelihood, responsibilities = do_estep(X, means, covars, weights)
        log_likelihood.append(curr_log_likelihood.sum())
        logprobs.append(curr_log_likelihood)

        # Check for convergence.
        if i > 5 and np.mean(abs(log_likelihood[-1] - log_likelihood[-2]) / I.sum()) < \
                tol:
            break

        # Maximization step
        means, covars, weights = do_mstep(X, I, responsibilities)

    # convert \Sigma^k to size and reduced shears like gsobject via shear module
    profs = []
    sizes = []
    e1s = []
    e2s = []
    sigmas = []
    for indx in range(n_gaussian):

        mean = means[indx]
        weight = weights[indx]
        covar = covars[indx]

        # figure out size
        size = covar[0,0] + covar[1,1]
        e1 = (covar[0,0] - covar[1,1]) / size
        e2 = 2 * covar[0,1] / size
        sigma = np.sqrt(np.sqrt(covar[0,0] * covar[1,1] - covar[0,1] * covar[1,0]))

        sizes.append(size)
        e1s.append(e1)
        e2s.append(e2)
        sigmas.append(sigma)

        # make gs object
        gauss = galsim.Gaussian(sigma=1.0)
        prof = gauss.dilate(sigma).shear(e1=e1, e2=e2).shift(mean[0], mean[1]) * weight

        profs.append(prof)

        # add to mixture
        if indx == 0:
            gmm = prof
        else:
            gmm = gmm + prof

    if return_all:
        return gmm, logprobs, weights, means, covars, log_likelihood, responsibilities, X, I, profs, sizes, e1s, e2s, sigmas
    else:
        return gmm
