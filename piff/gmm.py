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
import galsim

from .star import Star, StarData, StarFit
from .model import Model

def log_multivariate_normal_density(X, means, covars):
    """Log probability for full covariance matrices.
    Do for just two dimensions (image data)
    """
    n_samples, n_dim = X.shape
    # nmix = len(means)
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

    try:  # SciPy >= 0.19
        from scipy.special import logsumexp
    except ImportError:
        from scipy.misc import logsumexp
    logprob = logsumexp(lpr, axis=1)

    responsibilities = np.exp(lpr - logprob[:, np.newaxis])

    return logprob, responsibilities

def do_mstep(X, I, dX, responsibilities, force_model_center=False):
    """ Perform the Mstep of the EM algorithm and return the class weights.

    X is something like X[0] = [-100, -100]
    I is something like I[0] = 500

    """

    tau_ij__I_i = (I * dX)[:, np.newaxis] * responsibilities
    top_weights = tau_ij__I_i.sum(axis=0)
    bottom_weights = top_weights.sum() / (I * dX).sum()
    weights = top_weights / bottom_weights
    # add a small bit for top_weights
    top_weights += 1e-15
    means = np.dot(tau_ij__I_i.T, X)
    means /= top_weights[:, np.newaxis]
    if force_model_center:
        means = means * 0
    """
    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    tauI is [i, k]
    X is [i, j] -> Xp[k, i, j]
    means is [k, j] -> mup[k, i, j]
    xmu := (xi - muk)(xi - muk)^T should be [k, i, j1, j2]
    then cov = np.dot(tauI.T, xmu)
    cov is [k, j1, j2]
    """
    Xp = X[np.newaxis, :, :]
    meansp = means[:, np.newaxis, :]
    xmup = Xp - meansp
    xmu = xmup[:,:,:,np.newaxis] * xmup[:,:,np.newaxis,:]
    # this is way slow :(
    covars = (tau_ij__I_i.T[:,:,np.newaxis,np.newaxis] * xmu).sum(axis=1)
    covars /= top_weights[:, np.newaxis, np.newaxis]
    # add a small bit of covariance to the diagonal
    covars += 1e-7 * np.eye(X.shape[1])

    return means, covars, weights

def do_em(I, X, dX, n_gaussian=1, maxiter=100, tol=1e-04, force_model_center=False, logger=None):
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
    """
    # initial guess of parameters
    weights = np.random.random(n_gaussian)
    weights = weights * I.sum() / weights.sum()  # weight by
    if force_model_center:
        means = np.zeros((n_gaussian, 2))

    else:
        means = np.ones((n_gaussian, 2)) * np.median(X, axis=0)[None]

    covars = np.zeros((n_gaussian, 2, 2))
    covars[:, 0, 0] = np.random.random(n_gaussian) + 0.2
    covars[:, 1, 1] = np.random.random(n_gaussian) + 0.2
    if logger:
        logger.log(5, 'Input guesses are')
        logger.log(5, 'N, weight, mean_u, mean_v, covar_00, covar_01, covar_11')
        for indx in range(n_gaussian):
            logger.log(5, '{0}, {1:.2e}, {2:+.2e}, {3:+.2e}, {4:.2e}, {5:+.2e}, {6:+.2e}'.format(indx, weights[indx], means[indx][0], means[indx][1], covars[indx, 0, 0], covars[indx, 0, 1], covars[indx, 1, 1]))

    log_likelihood = []
    logprobs = []
    # iterate until sufficiently pleased
    if logger:
        logger.log(5, 'Beginning iteration')
    for i in range(maxiter):
        # Expectation step
        curr_log_likelihood, responsibilities = do_estep(X, means, covars, weights)
        log_likelihood.append(curr_log_likelihood.sum())
        logprobs.append(curr_log_likelihood)

        # Check for convergence. Require it take a few steps before giving up
        if i > 2 and np.mean(abs(log_likelihood[-1] - log_likelihood[-2]) / I.sum()) < \
                tol:
            break

        # Maximization step
        means, covars, weights = do_mstep(X, I, dX, responsibilities, force_model_center)
        if logger:
            logger.log(5, 'Step {0}, current log likelihood {1:.4e}'.format(i, log_likelihood[-1]))
            for indx in range(n_gaussian):
                logger.log(5, '{0}, {1:.2e}, {2:+.2e}, {3:+.2e}, {4:.2e}, {5:+.2e}, {6:+.2e}'.format(indx, weights[indx], means[indx][0], means[indx][1], covars[indx, 0, 0], covars[indx, 0, 1], covars[indx, 1, 1]))

    return weights, means, covars, logprobs, log_likelihood, responsibilities

def EMGMM(star, n_gaussian=1, return_all=False, maxiter=100, tol=1e-04, force_model_center=False, logger=None):
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

    weights, means, covars, logprobs, log_likelihood, responsibilities = do_em(I, X, n_gaussian, maxiter, tol, force_model_center, logger=logger)

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

    if logger:
        # log values
        logger.log(5, 'Measured GMM of star in {0} steps with final value of {1:.2e}. Values for each gaussian are:'.format(len(log_likelihood), log_likelihood[-1]))
        logger.log(5, 'N, weight, mean_u, mean_v, sigma, e1, e2')
        for indx in range(n_gaussian):
            logger.log(5, '{0}, {1:.2e}, {2:+.2e}, {3:+.2e}, {4:.2e}, {5:+.2e}, {6:+.2e}'.format(indx, weights[indx], means[indx][0], means[indx][1], sigmas[indx], e1s[indx], e2s[indx]))

    if return_all:
        return gmm, logprobs, weights, means, covars, log_likelihood, responsibilities, X, I, profs, sizes, e1s, e2s, sigmas
    else:
        return gmm


class GaussianMixtureModel(Model):
    """ Model that takes an image and fits a gaussian mixture via expectation maximization

    :param n_gaussian:          Number of gaussians to use
    :param maxiter:             Number of iterations allowed
    :param tol:                 Tolerance before stopping
    :param force_model_center:  If True, centroid is fixed at origin [default: True]
    :param logger:   A logger object for logging debug info. [default: None]
    """

    def __init__(self, n_gaussian=5, maxiter=100, tol=1e-4, force_model_center=True, order_by='weight', logger=None):
        self.kwargs = {'n_gaussian': n_gaussian,
                       'maxiter': maxiter,
                       'tol': tol,
                       'force_model_center': force_model_center,
                       'order_by': order_by,
                       }
        if self.kwargs['force_model_center']:
            self._niter = 4
        else:
            self._niter = 6
        self._ngauss = self.kwargs['n_gaussian']
        self._nparams = self._niter * self._ngauss

    def fit(self, star, logger=None):
        if 'other_model' in star.data.properties:
            logger.warn('Warning! Cannot use other_model in GMM fit!')

        n_gaussian = self._ngauss
        maxiter = self.kwargs['maxiter']
        tol = self.kwargs['tol']
        force_model_center = self.kwargs['force_model_center']

        I, wt, u, v = star.data.getDataVector(include_zero_weight=True)
        # calculate dX == area of pixel
        ny, nx = star.data.image.array.shape
        ugrid = u.reshape(ny, nx)
        vgrid = v.reshape(ny, nx)
        # assume constant pixel size. Could be relaxed
        du = np.median(np.diff(ugrid, axis=1))
        dv = np.median(np.diff(vgrid, axis=0))
        # put into same shape as I for the future in case we want to relax
        # constant pixel size requirement
        dX = du * dv * np.ones(I.shape)

        # now include the mask
        mask = wt != 0.
        I = I[mask]
        wt = wt[mask]
        u = u[mask]
        v = v[mask]
        dX = dX[mask]
        X = np.vstack((u, v)).T

        weights, means, covars, logprobs, log_likelihood, responsibilities = do_em(I, X, dX, n_gaussian=n_gaussian, maxiter=maxiter, tol=tol, force_model_center=force_model_center, logger=logger)
        shapes = self._convert_covars_to_shapes(covars)

        # create params
        params = np.zeros(self._nparams)
        for indx in range(n_gaussian):
            if force_model_center:
                params[indx * self._niter] = weights[indx]
                params[indx * self._niter + 1] = shapes[indx][0]
                params[indx * self._niter + 2] = shapes[indx][1]
                params[indx * self._niter + 3] = shapes[indx][2]
            else:
                params[indx * self._niter] = weights[indx]
                params[indx * self._niter + 1] = means[indx][0]
                params[indx * self._niter + 2] = means[indx][1]
                params[indx * self._niter + 3] = shapes[indx][0]
                params[indx * self._niter + 4] = shapes[indx][1]
                params[indx * self._niter + 5] = shapes[indx][2]

        # order params
        params = self._reorder_params(params, logger=logger)

        if logger:
            # log values
            logger.debug('Measured GMM of star. Values for each gaussian are:')
            if force_model_center:
                logger.debug('N, weight, sigma, g1, g2')
                for indx in range(n_gaussian):
                    logger.debug('{0}, {1:.2e}, {2:+.2e}, {3:+.2e}, {4:.2e}'.format(indx, *params[self._niter * indx: self._niter * (indx + 1)]))
            else:
                logger.debug('N, weight, mean_u, mean_v, sigma, g1, g2')
                for indx in range(n_gaussian):
                    logger.debug('{0}, {1:.2e}, {2:+.2e}, {3:+.2e}, {4:.2e}, {5:+.2e}, {6:+.2e}'.format(indx, *params[self._niter * indx: self._niter * (indx + 1)]))

        # Also need to compute chisq
        fit_nochisq = StarFit(params, flux=star.fit.flux, center=star.fit.center)
        prof = self.getProfile(Star(star.data, fit_nochisq))
        model_image = star.image.copy()
        if 'other_model' in star.data.properties:
            prof = galsim.Convolve([star.data.properties['other_model'], prof])
        prof.drawImage(model_image, method='no_pixel',
                                     offset=(star.image_pos - model_image.trueCenter()))
        chisq = np.sum(star.weight.array * (star.image.array - model_image.array)**2)
        dof = np.count_nonzero(star.weight.array) - self._nparams
        fit = StarFit(params, flux=star.fit.flux, center=star.fit.center, chisq=chisq, dof=dof)
        return Star(star.data, fit)

    @staticmethod
    def _convert_covars_to_shapes(covars):
        import galsim
        shapes = []
        for covar in covars:
            size = covar[0,0] + covar[1,1]
            e1 = (covar[0,0] - covar[1,1]) / size
            e2 = 2 * covar[0,1] / size
            sigma = np.sqrt(np.sqrt(covar[0,0] * covar[1,1] - covar[0,1] * covar[1,0]))
            shear = galsim.Shear(e1=e1, e2=e2)
            shapes.append([sigma, shear.getG1(), shear.getG2()])
        return shapes

    @staticmethod
    def _convert_shapes_to_covars(shapes):
        import galsim
        covars = []
        for shape in shapes:
            sigma, g1, g2 = shape
            shear = galsim.Shear(g1=g1, g2=g2)
            e1 = shear.getE1()
            e2 = shear.getE2()
            e0 = np.sqrt(4 * sigma ** 4 / (1 - e1 ** 2 - e2 ** 2))
            half_e0 = e0 * 0.5
            half_e1 = e1 * half_e0
            half_e2 = e2 * half_e0
            covar = [[half_e0 + half_e1, half_e2], [half_e2, half_e0 - half_e1]]
            covars.append(covar)
        return covars

    def _reorder_params(self, params, order_by=None, logger=None):
        if not order_by:
            order_by = self.kwargs['order_by']

        if order_by == 'weight':
            # get weights
            values = params[::self._niter]
        elif order_by == 'size':
            # get sizes
            if self.kwargs['force_model_center']:
                values = params[1::self._niter]
            else:
                values = params[3::self._niter]
        else:
            raise KeyError('Unknown order_by: {0}'.format(order_by))

        # reorder
        order = np.argsort(values)

        # update params
        params_new = params.reshape(self._ngauss, self._niter)[order].reshape(self._ngauss * self._niter)
        # TODO: easy test: put in params = np.arange(self._niter * self._ngauss)[::-1]
        # params_new = np.array([[range(i * self._niter, self._niter * (i + 1))] for i in range(self._ngauss - 1,-1,-1)]).flatten()

        if logger:
            # log values
            logger.log(5, 'Reordered GMM by {0}. Order was'.format(order_by))
            logger.log(5, 'N, params')
            for indx in range(self._ngauss):
                params_indx = params[self._niter * indx: self._niter * (indx + 1)]
                param_str = '{0}'.format(indx)
                for p in params_indx:
                    param_str += ', {0:+.3e}'.format(p)
                logger.log(5, param_str)
            logger.log(5, 'Order now is:')
            logger.log(5, 'N, params')
            for indx in range(self._ngauss):
                params_indx = params_new[self._niter * indx: self._niter * (indx + 1)]
                param_str = '{0}'.format(indx)
                for p in params_indx:
                    param_str += ', {0:+.3e}'.format(p)
                logger.log(5, param_str)

        return params_new

    def draw(self, star):
        """Draw the model on the given image.

        :param star:    A Star instance with the fitted parameters to use for drawing and a
                        data field that acts as a template image for the drawn model.

        :returns: a new Star instance with the data field having an image of the drawn model.
        """
        prof = self.getProfile(star)
        image = star.image.copy()
        # never use pixelization
        prof.drawImage(image, method='no_pixel', offset=(star.image_pos-image.trueCenter()))
        data = StarData(image, star.image_pos, star.weight, star.data.pointing, properties=star.data.properties, _xyuv_set=True)
        return Star(data, star.fit)

    def getProfile(self, star):
        """Get a version of the model as a GalSim GSObject

        :param star:        A star with a fit that has A numpy array with list
                            of either [ weight, size, g1, g2 ] or  [ weight,
                            cenu, cenv, size, g1, g2 ] depending on if the
                            center of the model is being forced to (0.0, 0.0)
                            or not.

        :returns: a galsim.GSObject instance
        """
        params = star.fit.params
        for indx in range(self.kwargs['n_gaussian']):
            params_indx = params[self._niter * indx: self._niter * (indx + 1)]
            if self.kwargs['force_model_center']:
                weight, sigma, g1, g2 = params_indx
                prof = galsim.Gaussian(sigma=1.0).dilate(sigma).shear(g1=g1, g2=g2) * weight
            else:
                weight, mu_u, mu_v, sigma, g1, g2 = params_indx
                prof = galsim.Gaussian(sigma=1.0).dilate(sigma).shear(g1=g1, g2=g2).shift(mu_u, mu_v) * weight
            if indx == 0:
                gmm = prof
            else:
                gmm = gmm + prof
        # TODO: do I need the star.fit.flux?
        gmm = gmm.shift(star.fit.center) * star.fit.flux
        return gmm
