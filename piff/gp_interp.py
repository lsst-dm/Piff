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
.. module:: gp_interp
"""

import numpy as np
import copy

from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process.kernels import Hyperparameter

from .interp import Interp
from .star import Star, StarFit


class GPInterp(Interp):
    """
    An interpolator that uses sklearn.gaussian_process to interpolate a single surface.

    :param keys:        A list of star attributes to interpolate from
    :param kernel:      A string that can be `eval`ed to make a
                        sklearn.gaussian_process.kernels.Kernel object.  The reprs of
                        sklearn.gaussian_process.kernels will work, as well as the repr of a
                        custom piff AnisotropicRBF or ExplicitKernel object.  [default: 'RBF()']
    :param optimize:    Boolean indicating whether or not to try and optimize the kernel by
                        maximizing the marginal likelihood.  [default: True]
    :param optimize_frac:  Optional fraction of stars to use during optimization of hyperparameters.
                        Setting this below 1.0 may significantly speed up the optimization step of
                        this code.  [default: 1.0]
    :param n_restarts_optimizer:  Number of times to restart optimization to search for better
                        hyperparameters.  See scikit-learn docs for more details.  Note that value
                        of 0 implies one optimization iteration is performed.  [default: 0]
    :param npca:        Number of principal components to keep.  [default: 0, which means don't
                        decompose PSF parameters into principle components]
    :param normalize:   Whether to normalize the interpolation parameters to have a mean of 0.
                        Normally, the parameters being interpolated are not mean 0, so you would
                        want this to be True, but if your parameters have an a priori mean of 0,
                        then subtracting off the realized mean would be invalid.  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, keys=('u','v'), kernel='RBF()', optimize=True, optimize_frac=1.,
                 n_restarts_optimizer=0, npca=0, normalize=True, logger=None):
        from sklearn.gaussian_process import GaussianProcessRegressor

        if optimize_frac < 0 or optimize_frac > 1:
            raise ValueError("optimize_frac must be between 0 and 1")

        self.keys = keys
        self.kernel = kernel
        self.optimize_frac = optimize_frac
        self.n_restarts_optimizer = n_restarts_optimizer
        self.npca = npca
        self.degenerate_points = False

        self.kwargs = {
            'keys': keys,
            'kernel': kernel,
            'optimize': optimize,
            'optimize_frac': optimize_frac,
            'n_restarts_optimizer': n_restarts_optimizer,
            'npca': npca,
        }
        optimizer = 'fmin_l_bfgs_b' if optimize else None
        self.gp_template = GaussianProcessRegressor(
                self._eval_kernel(self.kernel), optimizer=optimizer, normalize_y=normalize,
                n_restarts_optimizer=self.n_restarts_optimizer)

    @staticmethod
    def _eval_kernel(kernel):
        # Some import trickery to get all subclasses of sklearn.gaussian_process.kernels.Kernel
        # into the local namespace without doing "from sklearn.gaussian_process.kernels import *"
        # and without importing them all manually.
        def recurse_subclasses(cls):
            out = []
            for c in cls.__subclasses__():
                out.append(c)
                out.extend(recurse_subclasses(c))
            return out
        clses = recurse_subclasses(Kernel)
        for cls in clses:
            module = __import__(cls.__module__, globals(), locals(), cls)
            execstr = "{0} = module.{0}".format(cls.__name__)
            exec(execstr, globals(), locals())

        from numpy import array

        try:
            k = eval(kernel)
        except:
            raise RuntimeError("Failed to evaluate kernel string {0!r}".format(kernel))
        return k

    def _fit(self, gp, X, y, logger=None):
        """Update the GaussianProcessRegressor with data

        :param gp:      The GaussianProcessRegressor to update.
        :param X:       The independent covariates.  (n_samples, n_features)
        :param y:       The dependent responses.  (n_samples)
        :param logger:  A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.debug('Start GPInterp _fit: %s',gp.kernel)
            logger.debug('gp.fit with mean y = %s',np.mean(y))
        if gp.optimizer is not None and self.optimize_frac != 1:
            nstar = len(y)
            nchoose = int(nstar*self.optimize_frac)
            if logger:
                logger.info(
                    "Fitting GP hyperparameters using {} of {} stars".format(nchoose, nstar))
            choose = np.random.choice(nstar, nchoose, replace=False)
            Xtrain = X[choose]
            ytrain = y[choose]
            gp.fit(Xtrain, np.vstack(ytrain))
            init_kernel, fit_kernel = gp.kernel, gp.kernel_
            # Temporarily disable optimization to install all stars
            optimizer, gp.optimizer = gp.optimizer, None
            gp.kernel = gp.kernel_
            gp.fit(X, np.vstack(y))
            assert gp.kernel_ == fit_kernel
            gp.kernel = init_kernel
            gp.optimizer = optimizer
        else:
            gp.fit(X, np.vstack(y))
        if logger:
            logger.debug('After fit: kernel = %s',gp.kernel_)

    def _predict(self, Xstar):
        """ Predict responses given covariates.
        :param X:  The independent covariates at which to interpolate.  (n_samples, n_features).
        :returns:  Regressed parameters  (n_samples, n_targets)
        """
        ystar = np.array([gp.predict(Xstar)[:,0] for gp in self.gps]).T
        if self.npca > 0:
            ystar = self._pca.inverse_transform(ystar)
        return ystar

    def getProperties(self, star, logger=None):
        """Extract the appropriate properties to use as the independent variables for the
        interpolation.

        Take self.keys from star.data

        :param star:    A Star instances from which to extract the properties to use.

        :returns:       A np vector of these properties.
        """
        return np.array([star.data[key] for key in self.keys])

    def initialize(self, stars, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        :param stars:   A list of Star instances to interpolate between
        :param logger:  A logger object for logging debug info. [default: None]
        """
        self.nparams = len(stars[0].fit.params)
        if self.npca > 0:
            self.nparams = self.npca
        self.gps = [copy.deepcopy(self.gp_template) for i in range(self.nparams)]
        return stars

    def solve(self, stars, logger=None):
        """Solve for the GP interpolation.

        :param stars:    A list of Star instances to interpolate between
        :param logger:   A logger object for logging debug info. [default: None]
        """
        X = np.array([self.getProperties(star) for star in stars])
        y = np.array([star.fit.params for star in stars])
        if logger:
            logger.debug('Start solve: y = %s',y)
        if self.npca > 0:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.npca, whiten=True)
            self._pca.fit(y)
            y = self._pca.transform(y)
        if logger:
            logger.debug('After npca: y = %s',y)
        # Save these so serialization can reinstall them into gp.
        self._X = X
        self._y = y
        for i in range(self.nparams):
            gp = self.gps[i]
            self._fit(self.gps[i], X, y[:,i], logger=logger)
            if logger:
                logger.info('param %d: %s',i,gp.kernel_)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance with its StarFit member holding the interpolated parameters
        """
        # because of sklearn formatting, call interpolateList and take 0th entry
        return self.interpolateList([star], logger=logger)[0]

    def interpolateList(self, stars, logger=None):
        """Perform the interpolation for a list of stars.

        :param star_list:   A list of Star instances to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of new Star instances with interpolated parameters
        """
        Xstar = np.array([self.getProperties(star) for star in stars])
        y = self._predict(Xstar)
        if logger:
            logger.debug('star properties = %s',Xstar)
            logger.debug('interpolated params = ',y)
        fitted_stars = []
        for y0, star in zip(y, stars):
            if star.fit is None:
                fit = StarFit(y)
            else:
                fit = star.fit.newParams(y0)
            fitted_stars.append(Star(star.data, fit))
        return fitted_stars

    def _finish_write(self, fits, extname):
        # Note, we're only storing the training data and hyperparameters here, which means the
        # Cholesky decomposition will have to be re-computed when this object is read back from
        # disk.
        init_theta = np.array([gp.kernel.theta for gp in self.gps])
        fit_theta = np.array([gp.kernel_.theta for gp in self.gps])

        dtypes = [('INIT_THETA', init_theta.dtype, init_theta.shape),
                  ('FIT_THETA', fit_theta.dtype, fit_theta.shape),
                  ('X', self._X.dtype, self._X.shape),
                  ('Y', self._y.dtype, self._y.shape)]

        data = np.empty(1, dtype=dtypes)
        data['INIT_THETA'] = init_theta
        data['FIT_THETA'] = fit_theta
        data['X'] = self._X
        data['Y'] = self._y

        fits.write_table(data, extname=extname+'_kernel')

    def _finish_read(self, fits, extname):
        data = fits[extname+'_kernel'].read()
        # Run fit to set up GP, but don't actually do any hyperparameter optimization.  Just
        # set the GP up using the current hyperparameters.
        init_theta = np.atleast_1d(data['INIT_THETA'][0])
        fit_theta = np.atleast_1d(data['FIT_THETA'][0])
        self._X = np.atleast_1d(data['X'][0])
        self._y = np.atleast_1d(data['Y'][0])
        self.nparams = len(init_theta)
        self.gps = [copy.deepcopy(self.gp_template) for i in range(self.nparams)]
        for i in range(self.nparams):
            gp = self.gps[i]
            gp.kernel.theta = fit_theta[i]
            gp.optimizer = None
            self._fit(gp, self._X, self._y[:,i])
            gp.optimizer = self.gp_template.optimizer
            # Now that gp is setup, we can restore it's initial kernel.
            gp.kernel.theta = init_theta[i]


class FunctionKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A kernel that wraps an arbitrary python function.

    .. note::

        This kernel is not serializable directly.  If you want a serializable kernel, you
        should probably use ExplicitKernel instead.

    :param  fn:     A Python function to use as the function of the difference in the properties.
                    E.g. if the properties are (u,v), then fn(du,dv) should return the
                    correlation at that separation.
    """
    def __init__(self, fn):
        self.fn = fn
        self._fn = fn
        self.theta = []

    def __call__(self, X, Y=None, eval_gradient=False):

        X = np.atleast_2d(X)
        if Y is None:
            Y = X

        # Only writen for 2D covariance at the moment
        xshift = np.subtract.outer(X[:,0], Y[:,0])
        yshift = np.subtract.outer(X[:,1], Y[:,1])

        k = self._fn(xshift, yshift)

        if eval_gradient:
            # We don't have any hyper-parameters, so just return an empty array with 0 length
            # in the 3rd dimension
            return k, np.empty(k.shape + (0,))
        else:
            return k


class ExplicitKernel(FunctionKernel):
    """ A kernel that wraps an arbitrary python function, input as a string.

    :param  fn:  String that can be combined with 'lambda du,dv:' to eval into a lambda expression.
                 For example, fn="np.exp(-0.5*(du**2+dv**2)/0.1**2)" would make a Gaussian
                 kernel with scale length of 0.1.
    """
    def __init__(self, fn):
        self.fn = fn

        self.kwargs = {
            'fn': fn
        }

        self._fn = self._eval_fn(fn)
        self.theta = []

    @staticmethod
    def _eval_fn(fn):
        # Some potentially useful imports for the eval string.
        import math
        import numpy
        import numpy as np
        return eval("lambda du,dv:" + fn)


class AnisotropicRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ A GaussianProcessRegressor Kernel representing a radial basis function (essentially a
    squared exponential or Gaussian) but with arbitrary anisotropic covariance.

    While the parameter for this kernel, an inverse covariance matrix, can be specified directly
    with the `invLam` kwarg, it may be more convenient to instead specify a characteristic
    scale-length for each axis using the `scale_length` kwarg.  Note that a list or array is
    required so that the dimensionality of the kernel can be determined from its length.

    For optimization, it's necessary to reparameterize the inverse covariance matrix in such a way
    as to ensure that it's always positive definite.  To this end, we define `theta` (abbreviated
    `th` below) such that

    invLam = L * L.T
    L = [[exp(th[0])  0              0           ...    0                 0           ]
         [th[n]       exp(th[1])]    0           ...    0                 0           ]
         [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
         [...         ...            ...         ...    ...               ...         ]
         [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
         [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]

    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.

    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """
    def __init__(self, invLam=None, scale_length=None, bounds=(-5,5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError("Cannot set both invLam and scale_length in AnisotropicRBF.")
            invLam = np.diag(1./np.array(scale_length)**2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim*(self.ndim+1)//2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        from scipy.spatial.distance import pdist, cdist, squareform
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric='mahalanobis', VI=self.invLam)
            K = np.exp(-0.5 * dists**2)

        if eval_gradient:
            if self.hyperparameter_cholesky_factor.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # dK_pq/dth_k = -0.5 * K_pq *
                #               ((x_p_i-x_q_i) * dInvLam_ij/dth_k * (x_q_j - x_q_j))
                # dInvLam_ij/dth_k = dL_ij/dth_k * L_ij.T  +  L_ij * dL_ij.T/dth_k
                # dL_ij/dth_k is a matrix with all zeros except for one element.  That element is
                # L_ij if k indicates one of the theta parameters landing on the Cholesky diagonal,
                # and is 1.0 if k indicates one of the thetas in the lower triangular region.
                L_grad = np.zeros((self.ntheta, self.ndim, self.ndim), dtype=float)
                L_grad[(np.arange(self.ndim),)+self._d] = self._L[self._d]
                L_grad[(np.arange(self.ndim, self.ntheta),)+self._t] = 1.0

                half_invLam_grad = np.dot(L_grad, self._L.T)
                invLam_grad = half_invLam_grad + np.transpose(half_invLam_grad, (0, 2, 1))

                dX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                dist_grad = np.einsum("ijk,lkm,ijm->ijl", dX, invLam_grad, dX)
                K_gradient = -0.5 * K[:, :, np.newaxis] * dist_grad
                return K, K_gradient
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter("CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta))

    def get_params(self, deep=True):
        return {"invLam":self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[:self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim:]
        self.invLam = np.dot(self._L, self._L.T)

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds
