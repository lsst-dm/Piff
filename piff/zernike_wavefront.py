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
.. module:: optatmo
"""

# QUESTION: Why NOT make r0, g1, g2 also zernike dependent?

from __future__ import print_function

import numpy as np

import galsim

from .model import Model, ModelFitError
from .interp import Interp
from .psf import PSF

from .optical_model import Optical
from .star import Star, StarFit, StarData
# from .util import write_kwargs, read_kwargs, make_dtype, adjust_value

from .des.decam_wavefront import DECamWavefront
from .des.decaminfo import DECamInfo

from .util import hsm, hsm_error

from LearnPSF.zernike import DoubleZernike

from time import time

# bad practice oops
class ZernikeOpticalWavefrontPSF(PSF):
    """A PSF class that uses the wavefront to model the PSF

    We need to fit the following variables:
        Constant optical sigma or kolmogorov, g1, g2 in model
        misalignments in interpolant: these are the interp.misalignment terms
    """
    def __init__(self, knn_file_name, knn_extname, n_fit_stars=0, pupil_plane_im=None,  extra_interp_properties=None, weights=np.array([0.5, 1, 1]), max_shapes=np.array([1e10, 1e10, 1e10]), fitter_kwargs={}, interp_kwargs={}, model_kwargs={}, engine='galsim_fast', template='des', fitter_algorithm='scipy', min_snr=0, logger=None, max_focal_zernike=10):
        """

        :param knn_file_name:               Fits file containing the wavefront
        :param knn_extname:                 Extension name
        :param n_fit_stars:                 [default: 0] If > 0, randomly sample only n_fit_stars for the fit
        :param extra_interp_properties:     A list of any extra properties that will be used for
                                            the interpolation in addition to (u,v).
                                            [default: None]
        :param weights:                     Array or list of weights for comparing gaussian shapes in fit
                                            [default: [0.5, 1, 1], so downweight size]
        :param max_shapes:                  Array or list of maximum e0, e1, e2 values for stars used in fit. If they are excess of this, then cut.
                                            [default: [1e10, 1e10, 1e10], which should be >>> any measured values]
        :param fitter_kwargs:               kwargs to pass to fitter
        :param fitter_algorithm:            fitter to use for measuring wavefront. Default is scipy but also can use lmfit
        :param min_snr:                     minimum snr from star property required for fit
        :param max_focal_zernike:           up to what zernike term (noll basis) will we fit?
        """

        # TODO: trying with distance weighting to 40 nearest neighbors
        self.interp_kwargs = {'n_neighbors': 40, 'weights': 'distance', 'algorithm': 'auto'}
        self.interp_kwargs.update(interp_kwargs)

        # it turns out this part can also be slow!
        if logger:
            logger.info("Making interp")
        if knn_file_name == '':
            pass
        else:
            self.interp = DECamWavefront(knn_file_name, knn_extname, logger=logger, **self.interp_kwargs)

        if logger:
            logger.info("Making DECamInfo")
        self.decaminfo = DECamInfo()

        self.weights = np.array(weights)
        # normalize weights
        self.weights /= self.weights.sum()

        # max shapes
        self.max_shapes = np.array(max_shapes)

        if extra_interp_properties is None:
            self.extra_interp_properties = []
        else:
            self.extra_interp_properties = extra_interp_properties

        self.kwargs = {
            'pupil_plane_im': pupil_plane_im,
            'knn_file_name': knn_file_name,
            'knn_extname': knn_extname,
            'fitter_algorithm': fitter_algorithm,
            'n_fit_stars': n_fit_stars,
            'template': template,
            'engine': engine,
            'min_snr': min_snr,
            'max_focal_zernike': max_focal_zernike,
            }

        self.double_zernike = DoubleZernike(j_max=max(max_focal_zernike, 11))  # TODO: should not hardcode this...

        # load up the model after kwargs are set
        if logger:
            logger.info("Loading optical engine")
        self._engines = ['galsim', 'galsim_fast']
        self._model(template=template, engine=engine, **model_kwargs)


        # put in the variable names and initial values
        self.fitter_kwargs = {
            'r0': 0.15, 'fix_r0': False,    'limit_r0': (0.05, 0.4), 'error_r0': 1e-2,
            'g1': 0,   'fix_g1': False,    'limit_g1': (-0.2, 0.2),  'error_g1': 1e-2,
            'g2': 0,   'fix_g2': False,    'limit_g2': (-0.2, 0.2),  'error_g2': 1e-2,
            }
        self.keys = [
            'r0', 'g1', 'g2',
             ]
        # throw in default zernike parameters
        for zi in range(4, 11):  # do NOT include spherical
            for dxy in range(1, 10 + 1):
                zkey = 'zUV{0:02d}_zXY{1:02d}'.format(zi, dxy)
                self.keys.append(zkey)
                # initial value
                self.fitter_kwargs[zkey] = 0
                # fix if greater than max focal zernike
                if dxy > self.kwargs['max_focal_zernike']:
                    self.fitter_kwargs['fix_' + zkey] = True
                else:
                    self.fitter_kwargs['fix_' + zkey] = False
                # we shall not specify in advance a zernike limit
                # self.fitter_kwargs['limit_' + zkey] = (-2, 2)
                # initial guess for error in parameter
                zerror = 1e-4
                self.fitter_kwargs['error_' + zkey] = zerror

        # algorithm specific defaults
        if self.kwargs['fitter_algorithm'] == 'scipy':
            pass
        else:
            raise NotImplementedError('fitter {0} not implemented.'.format(self.kwargs['fitter_algorithm']))

        # update with user kwargs
        self.fitter_kwargs.update(fitter_kwargs)

        self._n_iter = 0
        self._time = time()

    def disable_atmosphere(self, logger=None):
        """Disable atmosphere within OpticalWavefrontPSF"""
        if logger:
            logger.info("Disabling atmosphere in OpticalWavefrontPSF")
        self.fitter_kwargs['r0'] = None
        self.fitter_kwargs['g1'] = None
        self.fitter_kwargs['g2'] = None

    def enable_atmosphere(self, r0, g1, g2, logger=None):
        """Turn the atmosphere back on.

        :param r0: seeing parameter
        :param g1,2: ellipticity
        :param logger: logger

        Note: r0, g1, g2 might still be fixed.
        """
        if logger:
            logger.info("Enabling atmosphere in OpticalWavefrontPSF")
        self.fitter_kwargs['r0'] = r0
        self.fitter_kwargs['g1'] = g1
        self.fitter_kwargs['g2'] = g2

    def _model(self, template='des', engine='galsim_fast', **model_kwargs):
        """Load up the modeling parameters
        :param template:    What template to load into the Optical class. Default is 'des'
        :param engine:      Changes gsparams for the Optical class.
        """
        if engine == 'galsim_fast':
            # pass in gsparams object to speed up everything
            gsparams = galsim.GSParams(minimum_fft_size=32,  # 128
                                       # maximum_fft_size=4096,  # 4096
                                       # stepk_minimum_hlr=5,  # 5
                                       # folding_threshold=5e-3,  # 5e-3
                                       # maxk_threshold=1e-3,  # 1e-3
                                       # kvalue_accuracy=1e-5,  # 1e-5
                                       # xvalue_accuracy=1e-5,  # 1e-5
                                       # table_spacing=1.,  # 1
                                       )
            pad_factor = 0.5
            oversampling = 0.5
            self.model = Optical(template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], gsparams=gsparams, pad_factor=pad_factor, oversampling=oversampling, **model_kwargs)
        elif engine == 'galsim':
            self.model = Optical(template=template, pupil_plane_im=self.kwargs['pupil_plane_im'], **model_kwargs)
        else:
            raise Exception('Invalid engine! {0}'.format(engine))

    @staticmethod
    def _measure_shape_errors(stars, logger=None):
        """Measure errors on galaxy shapes
        """
        errors = []
        for star_i, star in enumerate(stars):
            if logger:
                logger.log(5, "Measuring error of star {0}".format(star_i))
            try:
                sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2 = hsm_error(star, logger=logger)
                errors.append([sigma_e0, sigma_e1, sigma_e2])
            except (ValueError, ModelFitError, RuntimeError):
                # hsm nan'd out
                if logger:
                    logger.log(5, "Star failed moment parameter; setting errors to nan!")
                errors.append([np.nan, np.nan, np.nan])
        errors = np.array(errors)
        return errors

    @staticmethod
    def _measure_shapes(stars, logger=None):
        """Measure shapes using piff.util.hsm

        However, we have a special desired basis: we want the linear combinations of flux normalized, centered 2nd order moments.

        This emans we must do two things:

        1) convert sigma to Mxx + Myy
        2) convert g1, g2 to Mxx - Myy, 2 * Mxy

        Well. First we must realize that g != e!

        g from e:
            absesq = e1**2 + e2**2
            _e2g = 1. / (1. + sqrt(1.-absesq))
            g = (e1 + 1j * e2) * e2g
        e from g:
            absgsq = g1**2 + g2**2
            g2e = 2. / (1.+absgsq)
            e = (g1 + 1j * g2) * g2e

        OK so the above e are _normalized_ e!

        Next: sigma = (mxx myy - mxy mxy) ** 0.25
        but e0 = mxx + myy; e1 = mxx - myy; e2 = 2 mxy
        so sigma^4 = (e0^2 - e1^2 - e2^2) / 4
                   = e0^2 (1 - e1norm^2 - e2norm^2) / 4
        This gets you e0, which then gets you unnormalized e1 and e2

        """
        shapes = []
        for star_i, star in enumerate(stars):
            if logger:
                logger.log(5, "Measuring shape of star {0}".format(star_i))
            flux, u0, v0, sigma, g1, g2, flag = hsm(star, logger=logger)
            if flag != 0:
                # failed fit!
                if logger:
                    logger.log(5, "Star failed moment parameter; setting shapes to nan!")
                shapes.append([np.nan, np.nan, np.nan])
            else:
                # convert g to normalized e
                absgsq = g1 ** 2 + g2 ** 2
                e1norm = g1 * 2. / (1. + absgsq)
                e2norm = g2 * 2. / (1. + absgsq)
                # convert sigma to e0
                e0 = np.sqrt(4 * sigma ** 4 / (1. - e1norm ** 2 - e2norm ** 2))
                # un normalize normalized e
                e1 = e0 * e1norm
                e2 = e0 * e2norm

                # # old, and incorrect way!
                # e0 = sigma ** 2
                # e1 = e0 * g1
                # e2 = e0 * g2
                shapes.append([e0, e1, e2])
        shapes = np.array(shapes)

        return shapes

    def _fit_init(self, stars, wcs, pointing, logger=None):
        """Sets up all the tasks you would need to start the fit. Useful for when you want to test PSF functionality but don't want to run the expensive fit.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.warning("Start fitting OpticalWavefrontPSF using %s stars", len(stars))
        self.wcs = wcs
        self.pointing = pointing
        self.stars = stars

        # make sure stars have focal_x and focal_y
        zernikes, u, v, fx, fy = self._stars_to_parameters(stars, logger=logger)
        for star, fxi, fyi in zip(self.stars, fx, fy):
            star.data.properties['focal_x'] = fxi
            star.data.properties['focal_y'] = fyi


        self._fit_stars = stars
        choice = np.arange(len(stars))
        if self.kwargs['n_fit_stars'] and self.kwargs['n_fit_stars'] < len(self._fit_stars):
            choice = np.sort(np.random.choice(len(self._fit_stars), self.kwargs['n_fit_stars'], replace=False))
            self._fit_stars = [stars[i] for i in choice]
            if logger:
                logger.warning('Cutting from {0} to {1} stars'.format(len(self.stars), len(self._fit_stars)))
        # get the moments of the stars for comparison
        if logger:
            logger.warning("Start measuring the shapes")
        self._shapes = self._measure_shapes(self._fit_stars, logger=logger)
        self._errors = self._measure_shape_errors(self._fit_stars, logger=logger)
        # accumulate snrs
        snrs = np.array([star.data.properties['snr'] for star in self._fit_stars])
        # cut more stars if they fail shape measurement or error measurement, or if their shape values exceed a cut

        indx = ~np.any((self._shapes != self._shapes) +
                       (self._errors != self._errors) +
                       (np.abs(self._shapes) > self.max_shapes) +
                       (snrs < self.kwargs['min_snr'])[:, np.newaxis]
                       , axis=1)
        if logger:
            logger.warning("Cutting {0} stars out of {1}".format(sum(~indx), len(indx)))
        self._shapes = self._shapes[indx]
        self._errors = self._errors[indx]
        self._fit_stars = [star for star, ind in zip(self._fit_stars, indx) if ind]
        # make use_stars
        use_stars = np.in1d(np.arange(len(stars)), choice[indx], assume_unique=True)

        # modify stars to include whether used in fit.
        # This modifies both self.stars and stars
        for star, use_star in zip(self.stars, use_stars):
            star.data.properties['used_in_optical_wavefront'] = use_star

        self._logger = logger

    def fit(self, stars, wcs, pointing,
            logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self._fit_init(stars, wcs, pointing, logger=logger)

        if self.kwargs['fitter_algorithm'] == 'scipy':
            self._scipy_fit(logger=logger)
        # elif self.kwargs['fitter_algorithm'] == 'lmfit':
        #     self._lmfit_fit(logger=logger)
        else:
            raise NotImplementedError('fitter {0} not implemented!'.format(self.kwargs['fitter_algorithm']))

    @classmethod
    def _empty_zernikes(cls, stars):
        # put in starfits with empty params
        empty_params = np.zeros(8)
        empty_stars = [Star(star.data, StarFit(empty_params)) for star in stars]
        return empty_stars

    def _stars_to_parameters(self, stars, logger=None):
        """Takes in stars and returns their zernikes, u, v, and focal x and focal y coordinates.
        """
        zernikes = []
        u = []
        v = []
        fx = []
        fy = []

        # need the interpolator to NOT be misaligned!
        if self.kwargs['knn_file_name'] != '':
            if np.sum(np.abs(self.interp.misalignment)) > 0:
                if logger:
                    logger.warn('Warning! Resetting misalignment to zero!')
                self.interp.misalignment = 0 * self.interp.misalignment

            stars_interpolated = self.interp.interpolateList(self.decaminfo.pixel_to_focalList(stars))
        else:
            stars_interpolated = self._empty_zernikes(stars)
        for star in stars_interpolated:
            zernikes.append(star.fit.params)
            u.append(star.data.properties['u'])
            v.append(star.data.properties['v'])
            fx.append(star.data.properties['focal_x'])
            fy.append(star.data.properties['focal_y'])
        u = np.array(u)
        v = np.array(v)
        fx = np.array(fx)
        fy = np.array(fy)
        zernikes = np.array(zernikes)
        r = np.sqrt(fx ** 2 + fy ** 2) / 225.
        theta = np.arctan2(fy, fx)

        # construct misalignment array from fitter_kwargs and keys
        coefs = np.zeros((11, 10))
        for key in self.keys:
            if key in ['r0', 'g1', 'g2']:
                continue
            # zUV12_zXY34; kludgey as hell
            uv = int(key[3:5])
            xy = int(key[9:11])
            coefs[uv - 1, xy - 1] = self.fitter_kwargs[key]
        # get the modified zernikes
        dzernikes = self.double_zernike.realize(coefs, r, theta).T

        # apply misalignments to zernikes
        zernikes = zernikes + dzernikes[:, 3:]  # we start at z4

        return zernikes, u, v, fx, fy

    @staticmethod
    def _analytic_parameterization(params):
        """Put [r0, zernikes] into a useful set of polynomials for analytic fit.

        Assumes that params is shape [Nstars, r0 + zernikes]

        returns the parameters and the names assocaited with the parameters
        """
        # Note that r0 means 1.01/r0**2 and zernikes are mult by 2!!
        params = params.copy()
        params[:, 0] = 1. / params[:, 0] ** 2
        params[:, 0] = params[:, 0] * 0.01
        params[:, 1:3] = params[:, 1:3] * 1.
        params[:, 3:] = params[:, 3:] * 2
        psq = []
        pindx = []

        # TODO: can this be done without 3 for loops?
        param_names = ['r0', 'g1', 'g2'] + ['z{0:02d}'.format(i) for i in range(4, 12)]
        # only allow up to quadratic in zernike, but allow r0 to be quartic
        for pi in range(11):
            psq.append(params[:, pi])
            pindx.append([param_names[pi]])
            for pj in range(pi, 11):
                psq.append(params[:, pi] * params[:, pj])
                pindx.append([param_names[pi], param_names[pj]])
                if pi == 0:
                    for pk in range(pj, 11):
                        psq.append(params[:, pi] * params[:, pj] * params[:, pk])
                        pindx.append([param_names[pi], param_names[pj], param_names[pk]])
        psq = np.array(psq)
        pindx = np.array(pindx)
        return psq, pindx

    @staticmethod
    def _analytic_params_to_shapes(psq):
        """These come from looking at a lasso linear regression. Hardcoded."""

        e0 = 8.54e-01 * psq[0] + -9.19e-03 * psq[1] + -5.60e-04 * psq[2] + 2.34e-02 * psq[42] +  \
             1.98e-02 * psq[77] + 3.64e-02 * psq[100] + 2.43e-02 * psq[107] +  \
             2.32e-02 * psq[109] + 2.25e-02 * psq[117] +  \
             4.72e-02 * psq[124] + 4.78e-02 * psq[130] + 4.01e-02 * psq[135] +  \
             3.95e-02 * psq[139] + 8.96e-02 * psq[142]

        e1 = -3.56e-06 * psq[2] + 5.43e-02 * psq[3] + 1.70e-06 * psq[7] + 2.23e-05 * psq[11] +  \
             1.45e+00 * psq[13] + 2.65e-03 * psq[37] + 1.28e-02 * psq[57] +  \
             -3.32e-03 * psq[59] + 4.68e-03 * psq[61] +  \
             3.61e-03 * psq[65] + 4.83e-03 * psq[67] + -5.29e-05 * psq[70] +  \
             1.88e-04 * psq[74] + 2.66e-01 * psq[78] + 4.05e-02 * psq[102] +  \
             2.89e-02 * psq[122] + -8.41e-03 * psq[124] + 5.45e-02 * psq[126] +  \
             8.14e-03 * psq[130] + 5.40e-02 * psq[132] + -4.66e-05 * psq[135]

        e2 = 6.21e-02 * psq[4] + 8.52e-06 * psq[8] + 2.47e-05 * psq[10] + -5.61e-05 * psq[12] +  \
             1.44e+00 * psq[24] + 2.87e-03 * psq[36] + 1.21e-02 * psq[50] +  \
             6.47e-03 * psq[60] + -4.48e-03 * psq[62] +  \
             4.88e-03 * psq[66] + -1.28e-04 * psq[71] + 2.68e-01 * psq[89] +  \
             4.11e-02 * psq[101] + 2.98e-02 * psq[115] + 1.58e-02 * psq[125] +  \
             -5.49e-02 * psq[127] + 5.46e-02 * psq[131] + 1.88e-04 * psq[136] +  \
             6.51e-05 * psq[141]

        shapes = np.vstack((e0, e1, e2)).T

        return shapes

    def _analytic_stars_to_shapes(self, stars, logger=None):

        # get zernikes and coordinates
        zernikes, u, v, fx, fy = self._stars_to_parameters(stars, logger=logger)

        # stack zernikes, r0, g1, g2 into parameters
        r0 = self.fitter_kwargs['r0']
        g1 = self.fitter_kwargs['g1']
        g2 = self.fitter_kwargs['g2']
        ones = np.ones(len(zernikes))[:, None]
        # stack r0
        parameters = np.hstack((r0 * ones, g1 * ones, g2 * ones, zernikes))

        # psq
        psq, pindx = self._analytic_parameterization(parameters)

        # shapes
        model_shapes = self._analytic_params_to_shapes(psq)

        return model_shapes

    def interpolateList(self, stars, logger=None):
        zernikes, u, v, fx, fy = self._stars_to_parameters(stars, logger=logger)

        # put the zernikes into starfits
        stars_interpolated = []
        for star, zernike in zip(stars, zernikes):
            # create StarFit
            fit = StarFit(zernike)
            stars_interpolated.append(Star(star.data, fit))

        # update model with appropriate r0, g1, g2
        self.model.kwargs['r0'] = self.fitter_kwargs['r0']
        self.model.kolmogorov_kwargs['r0'] = self.fitter_kwargs['r0']
        self.model.g1 = self.fitter_kwargs['g1']
        self.model.g2 = self.fitter_kwargs['g2']

        return stars_interpolated

    def interpolate(self, star, logger=None):
        return self.interpolateList([star], logger=logger)[0]

    def chi2(self, vals, stars=None, logger=None):
        if not stars or stars == self._fit_stars:
            stars = self._fit_stars
            actual_shapes = self._shapes
            errors = self._errors
        else:
            actual_shapes = self._measure_shapes(stars, logger=logger)
            errors = self._measure_shape_errors(stars, logger=logger)

        # based on fitter_kwargs translate vals to the misalignments
        if logger:
            log = ['\n*******************************************************************************\n',
                    '* time \t|\t {0:.3e} \t|\t ncalls  \t|\t {1:04d} \t      *\n'.format(time() - self._time, self._n_iter),
                    '*******************************************************************************\n* ',]
        key_i = 0
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                self.fitter_kwargs[key] = vals[key_i]
                if logger:
                    log.append('{0} | {1:+.2e} * '.format(key, vals[key_i]))
                    if (key_i + 1) % 3 == 0:
                        log.append('\n* ')
                key_i += 1

        # now stars to shapes
        model_shapes = self._analytic_stars_to_shapes(stars, logger=logger)

        # chi2 of each star and shape
        chi2_long = np.sum((model_shapes - actual_shapes) ** 2 / errors ** 2, axis=0)
        dof = (3 * len(stars) - len(vals)) * np.sum(self.weights)
        # total chi2
        chi_squared = np.sum(self.weights * chi2_long) / dof

        if logger:

            log = log + [
                    '\n*******************************************************************************\n',
                    '* chi2 \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(*(chi2_long / dof)),
                    '*******************************************************************************',
                    ]
            if self._n_iter % 50 == 0:
                logger.info(''.join(log))
            else:
                logger.debug(''.join(log))

        self._n_iter += 1

        return chi_squared

    def _scipy_fit(self, logger=None):
        """Fit interpolated PSF model to star data by analytic relation. Should get us pretty close.

        :param logger:          A logger object for logging debug info. [default: None]

        :returns guess_values:  Reasonable first guesses for misalignment
        """
        from scipy.optimize import minimize

        # get p0 based on fitter_kwargs
        p0 = []
        if logger:
            logger.debug('Initial guess for analytic parameters:')
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                val = self.fitter_kwargs[key]
                p0.append(val)
                if logger:
                    logger.debug('{0}:\t{1:+.2e}'.format(key, val))

        # minimize chi2
        if logger:
            logger.info('Starting analytic guess. Initial chi2 = {0:.2e}'.format(self.chi2(p0)))
        res = minimize(self.chi2, p0, args=(None, logger,), options={'maxiter': 1000})
        if logger:
            logger.warn('Analytic finished. Final chi2 = {0:.2e}'.format(self.chi2(res.x)))

        # update fitter_kwargs
        if logger:
            logger.info('Analytic guess parameters:')
        key_i = 0
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                val = res.x[key_i]
                key_i += 1
                self.fitter_kwargs[key] = val
                if logger:
                    logger.info('{0}:\t{1:.2e}'.format(key, val))

    def drawStarList(self, stars):
        """Generate PSF images for given stars.

        :param stars:       List of Star instances holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           List of Star instances with its image filled with rendered PSF
        """
        return [self.drawStar(star) for star in stars]

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        # put in the focal coordinates
        star = self.decaminfo.pixel_to_focal(star)
        star = self.interpolate(star)


        # Render the image.
        model_star = self.model.draw(star)

        # # fix miscentering and flux via hsm
        # hsm_star = hsm(star)
        # # modify flux and fits
        # hsm_model = hsm(model_star)
        # model_star.fit.flux = model_star.fit.flux + hsm_star[0] - hsm_model[0]
        # model_star.fit.center = model_star.fit.center[0] + (hsm_star[1] - hsm_model[1]), model_star.fit.center[1] + (hsm_star[2] - hsm_model[2])
        # # final draw
        # model_star = self.model.draw(model_star)

        return model_star

    def getProfile(self, star):
        """Get galsim profile for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Galsim profile
        """
        # put in the focal coordinates
        star = self.decaminfo.pixel_to_focal(star)
        # Interpolate parameters to this position/properties:
        star = self.interpolate(star)
        # get the profile
        prof = self.model.getProfile(star)
        return prof

    def getParams(self, star):
        """Get params for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Params
        """
        # put in the focal coordinates
        star = self.decaminfo.pixel_to_focal(star)
        zernikes, u, v, fx, fy = self._stars_to_parameters([star])
        params = zernikes[0]
        return params

    def _finish_write(self, fits, extname, logger):
        """Save the misalignment parameters

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # save relevant entries in fitter_kwargs -- value and errors
        prefixes = ['', 'fix_', 'error_']
        data_types = [float, bool, float]

        # create dtype
        dtypes = [('weights', '3f4')]
        for key in self.keys:
            for prefix, data_type in zip(prefixes, data_types):
                combined_key = '{0}{1}'.format(prefix, key)
                dtypes.append((combined_key, data_type))

        data = np.zeros(1, dtype=dtypes)
        data['weights'][0] = self.weights
        for key in self.keys:
            for prefix in prefixes:
                combined_key = '{0}{1}'.format(prefix, key)
                data[combined_key][0] = self.fitter_kwargs[combined_key]

        fits.write_table(data, extname=extname + '_solution')

        # now save model and interp like a simplepsf
        # for the model, make sure the r0 and g1 and g2 and sigma are set in kwargs
        self.model.kwargs['sigma'] = self.model.sigma
        self.model.kwargs['g1'] = self.model.g1
        self.model.kwargs['g2'] = self.model.g2
        self.model.kwargs['r0'] = self.model.kolmogorov_kwargs['r0']
        self.model.write(fits, extname + '_model', logger)
        if logger:
            logger.info("Wrote the PSF model to extension %s",extname + '_model')

        if self.kwargs['knn_file_name'] != '':
            self.interp.write(fits, extname + '_interp', logger)
            if logger:
                logger.info("Wrote the PSF interp to extension %s",extname + '_interp')

    def _finish_read(self, fits, extname, logger):
        """Read in misalignment parameters

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # load relevant entries in fitter_kwargs
        # load weights
        data = fits[extname + '_solution'].read()
        self.weights = data['weights'][0]

        # load fitter_kwargs
        prefixes = ['', 'fix_', 'error_']
        for key in self.keys:  # self.keys defined in __init__, which comes first
            for prefix in prefixes:
                combined_key = '{0}{1}'.format(prefix, key)
                self.fitter_kwargs[combined_key] = data[combined_key][0]

        self._time = time()

        # load model and interp
        self.model = Model.read(fits, extname + '_model', logger)
        if self.kwargs['knn_file_name'] != '':
            self.interp = Interp.read(fits, extname + '_interp', logger)
