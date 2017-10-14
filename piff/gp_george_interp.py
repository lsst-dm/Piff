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
.. module:: gp_george_interp
"""

import numpy as np
import copy
import inspect
import scipy.optimize as op
from scipy.spatial.distance import cdist

import george
import george.kernels as Kernel
import treecorr

from .interp import Interp
from .star import Star, StarFit


class GPGeorgeInterp(Interp):
    """
    An interpolator that uses sklearn.gaussian_process to interpolate a single surface.

    :param keys:        A list of star attributes to interpolate from
    :param kernel:      A string that can be `eval`ed to make a
                        sklearn.gaussian_process.kernels.Kernel object.  The reprs of
                        sklearn.gaussian_process.kernels will work, as well as the repr of a
                        custom piff AnisotropicRBF or ExplicitKernel object.  [default: 'RBF()']
    :param optimize:    Boolean indicating whether or not to try and optimize the kernel by
                        maximizing the marginal likelihood.  [default: True]
    :param n_restarts_optimizer:  Number of times to restart optimization to search for better
                        hyperparameters.  See scikit-learn docs for more details.  Note that value
                        of 0 implies one optimization iteration is performed.  [default: 0]
    :param normalize:   Whether to normalize the interpolation parameters to have a mean of 0.
                        Normally, the parameters being interpolated are not mean 0, so you would
                        want this to be True, but if your parameters have an a priori mean of 0,
                        then subtracting off the realized mean would be invalid.  [default: True]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, keys=('u','v'), kernel="ExpSquaredKernel(metric=[[1., 0.], [0., 1.]], ndim=2)",
                 white_noise=None,optimize=True, optimize_pcf=True, n_restarts_optimizer=0,
                 normalize=True, logger=None):
        self.keys = keys
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.degenerate_points = False
        self.normalize = normalize

        if white_noise is not None:
            self.white_noise = np.log(white_noise)
        else:
            self.white_noise = None

        self.kwargs = {
            'keys': keys,
            'kernel': kernel,
            'optimize': optimize,
            'optimize_pcf':optimize_pcf,
            'n_restarts_optimizer': n_restarts_optimizer
        }
        self.count = 0
        self.optimizer = optimize
        self.optimize_pcf = optimize_pcf
        self._optimizer_init = copy.deepcopy(optimize)

        if type(kernel) is str:
            self.gp_template = [george.GP(self._eval_kernel(self.kernel),
                                          white_noise=self.white_noise)]
        else:
            if type(kernel) is not list:
                raise TypeError("kernel should be a string or a list of string")
            else:
                self.gp_template = [george.GP(self._eval_kernel(ker),
                                              white_noise=self.white_noise) for ker in self.kernel] 
        self._2pcf = []
        self._2pcf_dist = []
        self._2pcf_fit = []
        
    @staticmethod
    def _eval_kernel(kernel):
        # Some import trickery to get all subclasses of sklearn.gaussian_process.kernels.Kernel
        # into the local namespace without doing "from sklearn.gaussian_process.kernels import *"
        # and without importing them all manually.
        def recurse_subclasses(cls):
            out = []
            for name, obj in inspect.getmembers(Kernel):
                if inspect.isclass(obj):
                    out.append(obj)
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

    def _fit(self, gp, X, y, y_err=None, logger=None):
        """Update the GaussianProcessRegressor with data

        :param gp:      The GaussianProcessRegressor to update.
        :param X:       The independent covariates.  (n_samples, n_features)
        :param y:       The dependent responses.  (n_samples)
        :param logger:  A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.debug('Start GPGeorgeInterp _fit: %s',gp.kernel)
            logger.debug('gp.fit with mean y = %s',np.mean(y))

        if y_err is None:
            gp.compute(X, np.zeros(len(y)))
        else:
            gp.compute(X, y_err)

        if self.optimizer:
            if self.optimize_pcf:
                self._optimizer_2pcf(gp,X,y,y_err)
            else:
                self._optimizer_max_likelihood(gp,y)

        if logger:
            logger.debug('After fit: kernel = %s',gp.kernel)

    def _optimizer_2pcf(self,gp,X,y,y_err):
        print "j utilise 2pcf"
        #dist_matrix = cdist(X,X)
        #dist_vector = dist_matrix.ravel()
        #sep_min = np.percentile(dist_vector,1.)
        #sep_max = np.percentile(dist_vector,99.)
        #size_bin = sep_max/10.

        size_x = np.max(X[:,0]) - np.min(X[:,0])
        size_y = np.max(X[:,1]) - np.min(X[:,1])
        rho = float(len(X[:,0])) / (size_x * size_y)
        MIN = np.sqrt(1./rho)
        MAX = np.sqrt(size_x**2 + size_y**2)/2.

        if y_err is None or np.sum(y_err) == 0 :
            cat = treecorr.Catalog(x=X[:,0], y=X[:,1], k=(y-np.mean(y)))
        else:
            cat = treecorr.Catalog(x=X[:,0], y=X[:,1], k=(y-np.mean(y)), w=1./y_err**2)
        kk = treecorr.KKCorrelation(min_sep=MIN, max_sep=MAX, nbins=20)
        kk.process(cat)

        distance = np.exp(kk.logr)
        Coord = np.array([distance,np.zeros_like(distance)]).T
        print len(distance)

        def kernel(param):
            gp.set_parameter_vector(param)
            ker = gp.get_matrix(Coord, x2=np.zeros_like(Coord))
            pcf = ker[:,0]
            return pcf

        def chi2(param):
            residual = kk.xi - kernel(param)
            return np.sum(residual**2)

        print "start minimization"
        p0 = gp.get_parameter_vector()
        results = op.fmin(chi2, p0, disp=False)
        gp.set_parameter_vector(results)
        print "I am done"

        self._2pcf.append(kk.xi)
        self._2pcf_dist.append(distance)
        self._2pcf_fit.append(kernel(gp.get_parameter_vector()))

    def _optimizer_max_likelihood(self,gp,y):
        print "j utilise gradient likelihood"
        self.count +=1
        def nll(p):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(y, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        def grad_nll(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y, quiet=True)

        p0 = gp.get_parameter_vector()
        results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
        gp.set_parameter_vector(results['x'])

        if self.nparams == self.count:
            self.optimizer = False
            self.count = 0            
            
    def _predict(self, Xstar):
        """ Predict responses given covariates.
        :param X:  The independent covariates at which to interpolate.  (n_samples, n_features).
        :returns:  Regressed parameters  (n_samples, n_targets)
        """
        ystar = np.array([gp.predict(self._y[:,i]-self._mean[i], Xstar, return_cov=False) for i,gp in enumerate(self.gps)]).T
        for i in range(self.nparams):
            ystar[:,i] += self._mean[i]
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
        if len(self.gp_template)==1:
            self.gps = [copy.deepcopy(self.gp_template[0]) for i in range(self.nparams)]
        else:
            if len(self.gp_template)!= self.nparams:
                raise ValueError("numbers of kernel provided should be 1 (same for all parameters) or " \
                "equal to the number of params (%i), number kernel provided: %i"%((self.nparams,len(self.gp_template))))
            else:
                self.gps = [copy.deepcopy(gps) for gps in self.gp_template]
        return stars

    def solve(self, stars, logger=None):
        """Solve for the GP interpolation.

        :param stars:    A list of Star instances to interpolate between
        :param logger:   A logger object for logging debug info. [default: None]
        """
        X = np.array([self.getProperties(star) for star in stars])
        y = np.array([star.fit.params for star in stars])
        y_err = np.array([np.sqrt(np.diag(star.fit.params_cov)) for star in stars])

        if logger:
            logger.debug('Start solve: y = %s',y)
        # Save these so serialization can reinstall them into gp.
        self._X = X
        self._y = y
        self._y_err = y_err
        if self.normalize:
            self._mean = np.mean(y,axis=0)
        else:
            self._mean = np.zeros(self.nparams)
        self._init_theta = []
        for i in range(self.nparams):
            gp = self.gps[i]
            self._init_theta.append(gp.get_parameter_vector())
            self._fit(self.gps[i], X, y[:,i]-self._mean[i], y_err=y_err[:,i], logger=logger)
            if logger:
                logger.info('param %d: %s',i,gp.kernel)

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
        init_theta = np.array([self._init_theta[i] for i in range(self.nparams)])
        fit_theta = np.array([gp.get_parameter_vector() for gp in self.gps])

        dtypes = [('INIT_THETA', init_theta.dtype, init_theta.shape),
                  ('FIT_THETA', fit_theta.dtype, fit_theta.shape),
                  ('X', self._X.dtype, self._X.shape),
                  ('Y', self._y.dtype, self._y.shape),
                  ('Y_ERR', self._y_err.dtype, self._y_err.shape)]

        data = np.empty(1, dtype=dtypes)
        data['INIT_THETA'] = init_theta
        data['FIT_THETA'] = fit_theta
        data['X'] = self._X
        data['Y'] = self._y
        data['Y_ERR'] = self._y_err

        fits.write_table(data, extname=extname+'_kernel')

    def _finish_read(self, fits, extname):
        data = fits[extname+'_kernel'].read()
        # Run fit to set up GP, but don't actually do any hyperparameter optimization.  Just
        # set the GP up using the current hyperparameters.
        init_theta = np.atleast_1d(data['INIT_THETA'][0])
        fit_theta = np.atleast_1d(data['FIT_THETA'][0])

        self._X = np.atleast_1d(data['X'][0])
        self._y = np.atleast_1d(data['Y'][0])
        self._y_err = np.atleast_1d(data['Y_ERR'][0])
        self._init_theta = init_theta
        self.nparams = len(init_theta)
        if self.normalize:
            self._mean = np.mean(self._y,axis=0)
        else:
            self._mean = np.zeros(self.nparams)
        if len(self.gp_template)==1:
            self.gps = [copy.deepcopy(self.gp_template[0]) for i in range(self.nparams)]
        else:
            if len(self.gp_template)!= self.nparams:
                raise ValueError("numbers of kernel provided should be 1 (same for all parameters) or " \
                "equal to the number of params (%i), number kernel provided: %i"%((self.nparams,len(self.gp_template))))
            else:
                self.gps = [copy.deepcopy(gps) for gps in self.gp_template]
        #print self._mean
        for i in range(self.nparams):
            gp = self.gps[i]
            gp.set_parameter_vector(fit_theta[i])
            self.optimizer = False
            self._fit(gp, self._X, self._y[:,i]-self._mean[i], y_err=self._y_err[:,i])
            self.optimizer = self._optimizer_init 
            # Now that gp is setup, we can restore it's initial kernel.

