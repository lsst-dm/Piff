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
.. module:: gp_corr
"""

from __future__ import print_function
import numpy as np
import scipy
import copy
import galsim

from .gp_interp import GPInterp, FunctionKernel
from .interp import Interp
from .star import Star, StarFit


class convert_fn_r:
    def __init__(self, fn): self.fn = fn
    def __call__(self, du, dv):
        #print('du,dv = ',du,dv)
        r = np.sqrt(du**2+dv**2)
        #print('r = ',r)
        f = self.fn(r)
        #print('f = ',f)
        #try:
            #print('f = ',f.tolist())
            #svd = np.linalg.svd(f)
            #print('svd = ',svd)
            #chol = scipy.linalg.cholesky(f)
            #print('chol = ',chol)
        #except Exception as e:
            #print(e)
            #pass
        return f

class EmpiricalGPInterp(GPInterp):
    """
    An interpolator that uses sklearn.gaussian_process to interpolate a single surface
    and generates the kernel automatically using an empirical 2pt correlation function
    of the measured values.

    .. note:: 

        This interpolator currently only works for 2 interpolation properties.  Furthermore,
        these properties are assumed to be positions (e.g. u,v).

    :param keys:        A list of star attributes to interpolate from
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
    :param nbins:       The number of bins to use for the correlation function. [default: 20]
    :param rmin:        The minimum separation at which to estimate the correlation function. 
                        [default: 30 (arcsec)]
    :param rmax:        The maximum separation at which to estimate the correlation function.
                        [default: 6000 (arcsec)]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    def __init__(self, keys=('u','v'), optimize=True, optimize_frac=1.,
                 n_restarts_optimizer=0, npca=0, normalize=True,
                 nbins=20, rmin=30, rmax=6000, logger=None):

        self.keys = keys
        self.optimize_frac = optimize_frac
        self.n_restarts_optimizer = n_restarts_optimizer
        self.npca = npca
        self.degenerate_points = False
        self.nbins = nbins
        self.rmin = rmin
        self.rmax = rmax

        self.kwargs = {
            'keys': keys,
            'optimize': optimize,
            'optimize_frac': optimize_frac,
            'n_restarts_optimizer': n_restarts_optimizer,
            'npca': npca,
            'nbins': nbins,
            'rmin': rmin,
            'rmax': rmax
        }
        self.optimizer = 'fmin_l_bfgs_b' if optimize else None
        self.normalize = normalize

    def initialize(self, stars, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        :param stars:   A list of Star instances to interpolate between
        :param logger:  A logger object for logging debug info. [default: None]
        """
        self.nparams = len(stars[0].fit.params)
        if self.npca > 0:
            self.nparams = self.npca
        return stars

    def solve(self, stars, logger=None):
        """Solve for the GP interpolation.

        In this class, this is also where the 2pt correlation function is calculated.

        :param stars:   A list of Star instances to interpolate between
        :param logger:  A logger object for logging debug info. [default: None]
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        import treecorr
        X = np.array([self.getProperties(star) for star in stars])
        y = np.array([star.fit.params for star in stars])
        if X.shape[1] != 2:
            raise ValueError("EmpiricalGPInterp only works for 2 interpolation properties.")
        if self.npca > 0:
            from sklearn.decomposition import PCA
            self._pca = PCA(n_components=self.npca, whiten=True)
            self._pca.fit(y)
            y = self._pca.transform(y)
        # Save these so serialization can reinstall them into gp.
        self._X = X
        self._y = y
        self.gps = []
        logr = []
        xi = []
        vark = []
        u = X[:,0]  # Just aliases to help me reason about the below code properly.
        v = X[:,1]
        #print('u = ',u)
        #print('v = ',v)
        for i in range(self.nparams):
            print('Start param ',i)
            yi = y[:,i]
            #print('yi = ',yi)
            #print('yi => ',yi-np.mean(yi))
            cat = treecorr.Catalog(k=yi-np.mean(yi), x=u, y=v)
            kk = treecorr.KKCorrelation(nbins=self.nbins, min_sep=self.rmin, max_sep=self.rmax)
            kk.process(cat)
            #print('kk.meanlogr = ',kk.meanlogr)
            #print('kk.xi = ',kk.xi)
            #print('kk.varxi = ',kk.varxi)
            mask = kk.npairs > 0
            vk = treecorr.calculateVarK(cat)
            #vk *= 2
            print('vark = ',vk)
            kernel = self._make_kernel(kk.meanlogr[mask], kk.xi[mask], vk)
            #print('made kernel: ',kernel)
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=self.optimizer,
                                          normalize_y=self.normalize)
            print('made gp: ',gp.kernel)
            logr.append(kk.meanlogr[mask])
            xi.append(kk.xi[mask])
            vark.append(vk)
            self.gps.append(gp)
            self._fit(gp, X, yi)
            print('after fit: ',gp.kernel_)
        self.logr = np.array(logr)
        self.xi = np.array(xi)
        self.vark = np.array(vark)
        return stars

    def _make_kernel(self, logr, xi, vark):
        """Make the kernel from logr and xi.

        :param logr:    The log of the separation
        :param xi:      The correlation function
        :param vark:    The 1pt variance of the parameter
        """
        from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel

        # We need to make sure the function is callable at smaller and larger scales than where
        # we measured it.
        # For small scales, we set fn(0) to be the maximum measured value at any separation.
        # Normally, this will be xi[0], but it might not, and there is a mathematical constraint
        # that xi(r) <= xi(0), so make sure that holds in our case.
        # For large scales, we just set anything larger than where we measured it to 0.
        # And since we actually interpolate in logs, use 1.e-100 for 0 and 1.e100 for inf.

        #print('make_kernel:')
        #print('logr = ',logr)
        #print('xi = ',xi)
        #print('vark = ',vark)
        bin_size = (logr[-1] - logr[0]) / (len(logr)-1)
        #print('bin_size = ',bin_size)
        r = np.concatenate(([0], np.exp(logr), [np.exp(logr[-1] + bin_size), 1.e100]))
        xi = np.concatenate(([vark], xi, [0, 0]))
        print('r = ',r)
        print('xi = ',xi)
        fn = galsim.LookupTable(x=r, f=xi, interpolant='linear')
        #print('fn = ',fn)
        #print('fn(3) = ',fn(3))
        #print('fn(3.e-10) = ',fn(3.e-10))
        #print('fn(3.e10) = ',fn(3.e10))
        #f = lambda du,dv: fn(np.sqrt(du**2+dv**2))
        f = convert_fn_r(fn)
        #print('f(3,4) = ',f(3,4))
        return (FunctionKernel(f) *
            ConstantKernel(constant_value=1., constant_value_bounds=(1.e-2,1.e2))
            + WhiteKernel(noise_level=vark, noise_level_bounds=(vark*0.001,vark*10)))

    def _finish_write(self, fits, extname):
        # Note, we're only storing the training data and hyperparameters here, which means the
        # Cholesky decomposition will have to be re-computed when this object is read back from
        # disk.
        for gp in self.gps:
            print('Writing gp: ',gp.kernel, gp.kernel_)
        init_theta = np.array([gp.kernel.theta for gp in self.gps])
        fit_theta = np.array([gp.kernel_.theta for gp in self.gps])

        #store_theta = (len(self.gps[0].kernel.theta.shape) >= 2)
        store_theta = np.prod(self.gps[0].kernel.theta.shape) > 0

        dtypes = [('LOGR', self.logr.dtype, self.logr.shape),
                  ('XI', self.xi.dtype, self.xi.shape),
                  ('VARK', self.vark.dtype, self.vark.shape),
                  ('X', self._X.dtype, self._X.shape),
                  ('Y', self._y.dtype, self._y.shape)]
        if store_theta:
            dtypes += [('INIT_THETA', init_theta.dtype, init_theta.shape),
                       ('FIT_THETA', fit_theta.dtype, fit_theta.shape)]
        #print('dtypes = ',dtypes)

        data = np.empty(1, dtype=dtypes)
        if store_theta:
            data['INIT_THETA'] = init_theta
            data['FIT_THETA'] = fit_theta
        data['LOGR'] = self.logr
        data['XI'] = self.xi
        data['VARK'] = self.vark
        data['X'] = self._X
        data['Y'] = self._y
        #print('data = ',data)

        fits.write_table(data, extname=extname+'_kernel')
        #print('wrote table')

    def _finish_read(self, fits, extname):
        from sklearn.gaussian_process import GaussianProcessRegressor
        data = fits[extname+'_kernel'].read()
        #print('data = ',data)
        #print('Y = ',data['Y'])
        # Run fit to set up GP, but don't actually do any hyperparameter optimization.  Just
        # set the GP up using the current hyperparameters.
        self.logr = np.atleast_1d(data['LOGR'][0])
        self.xi = np.atleast_1d(data['XI'][0])
        self.vark = np.atleast_1d(data['VARK'][0])
        self._X = np.atleast_1d(data['X'][0])
        self._y = np.atleast_1d(data['Y'][0])
        #print('_Y = ',self._y)
        self.nparams = len(self.xi)
        self.gps = []
        if 'INIT_THETA' in data.dtype.names:
            use_theta = True
            init_theta = np.atleast_1d(data['INIT_THETA'][0])
            fit_theta = np.atleast_1d(data['FIT_THETA'][0])
        else:
            use_theta = False
        for i in range(self.nparams):
            kernel = self._make_kernel(self.logr[i], self.xi[i], self.vark[i])
            gp = GaussianProcessRegressor(kernel=kernel, optimizer=self.optimizer,
                                          normalize_y=self.normalize)
            if use_theta:
                gp.kernel.theta = fit_theta[i]
                gp.optimizer = None
            self._fit(gp, self._X, self._y[:,i])
            gp.optimizer = self.optimizer
            # Now that gp is setup, we can restore it's initial kernel.
            if use_theta:
                gp.kernel.theta = init_theta[i]
            self.gps.append(gp)
        for gp in self.gps:
            print('Read gp: ',gp.kernel, gp.kernel_)

