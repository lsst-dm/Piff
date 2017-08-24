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

from time import time

class OptAtmoPSF(PSF):
    """A PSF class that uses an OpticalWavefrontPSF and some other PSF together to model the PSF.
    """

    # TODO: is this bad practice? it lets me keep the classmethod below...
    psf_keys = ['optpsf', 'atmopsf']

    def __init__(self, optpsf, atmopsf):
        """
        :param psfs:     PSF instances to use, labelled by 'psf_0', 'psf_1', etc (for order in operation)
        """

        self.kwargs = {key: 0 for key in self.psf_keys}
        self.optpsf = optpsf
        self.atmopsf = atmopsf
        self.psfs = [self.optpsf, self.atmopsf]

    @classmethod
    def parseKwargs(cls, config_psf, logger):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        import piff

        config_psf = config_psf.copy()  # Don't alter the original dict.

        kwargs = {}
        # find the psf_kinds
        for psf_key in cls.psf_keys:
            config = config_psf.pop(psf_key)
            kwargs[psf_key] = piff.PSF.process(config, logger)

        return kwargs

    def fit(self, stars, wcs, pointing,
            logger=None, **kwargs):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self.stars = stars
        self.wcs = wcs
        self.pointing = pointing

        # TODO: star marking?

        # fit OpticalWavefrontPSF
        if logger:
            logger.info("Starting OpticalWavefrontPSF fit")

        # Check if the constant atmosphere piece of the optical wavefront is
        # turned off. Don't do anything if it is (might be intentional) but
        # send a warning (it is unconventional).
        if logger:
            for key in ['r0', 'g1', 'g2']:
                if self.optpsf.fitter_kwargs['fix_' + key]:
                    logger.warning("Warning! You have left {0} fixed. This parameter coarsely emulates the atmosphere (so that we can get near the correct location before we actually fit the atmosphere), so it is usually good to have on.".format(key))

        self.optpsf.fit(self.stars, wcs, pointing, logger=logger)

        # update stars from outlier rejection
        # TODO: mark stars probably changes this
        if self.optpsf.kwargs['n_fit_stars']:
            nremoved = self.optpsf.kwargs['n_fit_stars'] - len(self.optpsf._fit_stars)
        else:
            nremoved = len(self.stars) - len(self.optpsf._fit_stars)
        if logger:
            if nremoved > 0:
                logger.warning("Removed {0} stars in OpticalWavefrontPSF fit".format(nremoved))
            else:
                logger.info("Removed no stars in OpticalWavefrontPSF fit")

        # disable r0,g1,g2 from OpticalWavefrontPSF, since AtmoPSF deals with those bits.
        self.optpsf.disable_atmosphere(logger=logger)
        if logger:
            logger.debug("optpsf interp misalignment is")
            logger.debug(self.optpsf.interp.misalignment)

        # extract profiles for AtmoPSF
        if logger:
            logger.info("Extracting OpticalWavefrontPSF profiles")
        # TODO: add these profiles to the stars
        profiles = [self.optpsf.getProfile(star) for star in self.stars]
        # clean out starfits
        self.stars = [Star(star.data, None) for star in self.stars]
        # now add the profiles
        for star, profile in zip(self.stars, profiles):
            star.data.properties['other_model'] = profile

        # fit AtmoPSF
        if logger:
            logger.info("Fitting AtmospherePSF")
        # TODO: want to make sure when we draw stars in atmopsf that we draw the same with self.drawStar
        self.atmopsf.fit(self.stars, wcs, pointing, logger=logger)

        # update stars from outlier rejection
        nremoved = len(self.stars) - len(self.atmopsf.stars)
        if logger:
            if nremoved > 0:
                logger.warning("Removed {0} stars in AtmospherePSF fit".format(nremoved))
            else:
                logger.info("Removed no stars in AtmospherePSF fit")

        try:
            chisq = np.sum([s.fit.chisq for s in self.stars])
            dof   = np.sum([s.fit.dof for s in self.stars])
            if logger:
                logger.warn("Total chisq = %.2f / %d dof", chisq, dof)
        except:
            self.stars = self.atmopsf.stars
            try:
                chisq = np.sum([s.fit.chisq for s in self.stars])
                dof   = np.sum([s.fit.dof for s in self.stars])
                if logger:
                    logger.warn("Total chisq = %.2f / %d dof", chisq, dof)
            except:
                if logger:
                    logger.warn("Unable to produce chisq?!")

        # the job done, remove other_model from star properties
        for stars in [self.stars, self.atmopsf.stars, self.optpsf.stars, self.optpsf._fit_stars]:
            for star in stars:
                if 'other_model' in star.data.properties:
                    star.data.properties.pop('other_model')

    def getParams(self, star):
        """Get params for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Params
        """
        params = []
        for psf_i in self.psfs:
            params.append(psf_i.getParams(star))

        # "convolve"
        params = np.hstack(params)

        return params

    def getProfile(self, star):
        """Get galsim profile for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Galsim profile
        """
        import galsim

        profs = []
        for psf_i in self.psfs:
            # TODO: do I need to pass in gsparams?
            # TODO: do I need to apply shifts etc?
            prof = psf_i.getProfile(star)
            profs.append(prof)

        # convolve
        prof = galsim.Convolve(profs)

        return prof

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """

        # get profile and params
        prof = self.getProfile(star)
        params = self.getParams(star)

        image = star.image.copy()
        # TODO: method == pixel or no_pixel or auto?
        image = prof.drawImage(image, method='auto', offset=(star.image_pos-image.trueCenter()))
        # TODO: might need to update image pos?
        properties = star.data.properties.copy()
        for key in ['x', 'y', 'u', 'v']:
            # Get rid of keys that constructor doesn't want to see:
            properties.pop(key,None)
        data = StarData(image=image,
                        image_pos=star.data.image_pos,
                        weight=star.data.weight,
                        pointing=star.data.pointing,
                        field_pos=star.data.field_pos,
                        values_are_sb=star.data.values_are_sb,
                        orig_weight=star.data.orig_weight,
                        properties=properties)
        return Star(data, StarFit(params))

    def drawStarList(self, stars):
        """Generate PSF images for given stars.

        :param stars:       List of Star instances holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           List of Star instances with its image filled with rendered PSF
        """
        return [self.drawStar(star) for star in stars]

    def _finish_write(self, fits, extname, logger):
        """Finish the writing process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        for psf_key, psf_i in zip(self.psf_keys, self.psfs):
            psf_i._write(fits, extname + '_{0}'.format(psf_key), logger)

    def _finish_read(self, fits, extname, logger):
        """Finish the reading process with any class-specific steps.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        self.psfs = []
        for psf_key in self.psf_keys:
            self.psfs.append(PSF._read(fits, extname + '_{0}'.format(psf_key), logger))

        # TODO: because we have finished reading a presumably fitted model, make sure we have turned off the constant atmospheric piece used in the optical portion of the psf

class OpticalWavefrontPSF(PSF):
    """A PSF class that uses the wavefront to model the PSF

    We need to fit the following variables:
        Constant optical sigma or kolmogorov, g1, g2 in model
        misalignments in interpolant: these are the interp.misalignment terms
    """
    def __init__(self, knn_file_name, knn_extname, max_iterations=300, n_fit_stars=0, error_estimate=0.001, pupil_plane_im=None,  extra_interp_properties=None, weights=np.array([0.5, 1, 1]), max_shapes=np.array([1e10, 1e10, 1e10]), fitter_kwargs={}, interp_kwargs={}, model_kwargs={}, engine='galsim_fast', template='des', fitter_algorithm='minuit', guess_start=False, use_gradient=True, logger=None):
        """

        :param knn_file_name:               Fits file containing the wavefront
        :param knn_extname:                 Extension name
        :param max_iterations:              Maximum number of iterations to try. [default: 300]
        :param n_fit_stars:                 [default: 0] If > 0, randomly sample only n_fit_stars for the fit
        :param error_estimate:              [default: 0.01] Fudge factor in chi square until we get real errors for the hsm algorithm. Gives us how well we think we measure the moments.
        :param extra_interp_properties:     A list of any extra properties that will be used for
                                            the interpolation in addition to (u,v).
                                            [default: None]
        :param weights:                     Array or list of weights for comparing gaussian shapes in fit
                                            [default: [0.5, 1, 1], so downweight size]
        :param max_shapes:                  Array or list of maximum e0, e1, e2 values for stars used in fit. If they are excess of this, then cut.
                                            [default: [1e10, 1e10, 1e10], which should be >>> any measured values]
        :param fitter_kwargs:               kwargs to pass to fitter
        :param fitter_algorithm:            fitter to use for measuring wavefront. Default is minuit but also can use lmfit
        :param guess_start:                 if True, will adjust fitter kwargs for best guess
        :param use_gradient:                if True, will use gradient functions in fit
        """

        # TODO: trying with distance weighting to 40 nearest neighbors
        self.interp_kwargs = {'n_neighbors': 40, 'weights': 'distance', 'algorithm': 'auto'}
        self.interp_kwargs.update(interp_kwargs)

        # it turns out this part can also be slow!
        if logger:
            logger.info("Making interp")
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
            'max_iterations': max_iterations,
            'error_estimate': error_estimate,
            'template': template,
            'engine': engine,
            'guess_start': guess_start,
            'use_gradient': use_gradient,
            }

        # load up the model after kwargs are set
        if logger:
            logger.info("Loading optical engine")
        self._engines = ['galsim', 'galsim_fast']
        self._model(template=template, engine=engine, **model_kwargs)


        # put in the variable names and initial values
        self.fitter_kwargs = {
            'r0': 0.15, 'fix_r0': False,   'limit_r0': (0.01, 0.4), 'error_r0': 1e-2,
            'g1': 0,   'fix_g1': False,   'limit_g1': (-0.2, 0.2),  'error_g1': 1e-2,
            'g2': 0,   'fix_g2': False,   'limit_g2': (-0.2, 0.2),  'error_g2': 1e-2,
            }
        # throw in default zernike parameters
        for zi in range(4, 12):
            for dxy in ['d', 'x', 'y']:
                zkey = 'z{0:02d}{1}'.format(zi, dxy)
                # initial value
                self.fitter_kwargs[zkey] = 0
                # fix
                self.fitter_kwargs['fix_' + zkey] = False
                # we shall not specify in advance a zernike limit
                # self.fitter_kwargs['limit_' + zkey] = (-2, 2)
                # initial guess for error in parameter
                if dxy == 'd':
                    zerror = 1e-2
                else:
                    zerror = 1e-4
                self.fitter_kwargs['error_' + zkey] = zerror

        # algorithm specific defaults
        if self.kwargs['fitter_algorithm'] == 'minuit':
            self.fitter_kwargs.update({
                'throw_nan': False,
                'pedantic': True,
                'print_level': 2,
                'errordef': 1.0,  # guesstimated
                })
        elif self.kwargs['fitter_algorithm'] == 'lmfit':
            pass
        elif self.kwargs['fitter_algorithm'] == 'scipy':
            pass
        else:
            raise NotImplementedError('fitter {0} not implemented.'.format(self.kwargs['fitter_algorithm']))

        # update with user kwargs
        self.fitter_kwargs.update(fitter_kwargs)

        self.update_psf_params(logger=logger, **self.fitter_kwargs)

        self._time = time()
        self._n_iter = 0

        self.step_sizes = [
            1e-4, 1e-4, 1e-4,  # sizes
            1e-3, 1e-5, 1e-5,  # z04
            1e-3, 1e-5, 1e-5,  # z05
            1e-3, 1e-5, 1e-5,  # z06
            1e-3, 1e-5, 1e-5,  # z07
            1e-3, 1e-5, 1e-5,  # z08
            1e-3, 1e-5, 1e-5,  # z09
            1e-3, 1e-5, 1e-5,  # z10
            1e-3, 1e-5, 1e-5,  # z11
                ]
        self.keys = [
            'r0', 'g1', 'g2',
            'z04d', 'z04x', 'z04y',
            'z05d', 'z05x', 'z05y',
            'z06d', 'z06x', 'z06y',
            'z07d', 'z07x', 'z07y',
            'z08d', 'z08x', 'z08y',
            'z09d', 'z09x', 'z09y',
            'z10d', 'z10x', 'z10y',
            'z11d', 'z11x', 'z11y',
                ]

    def disable_atmosphere(self, logger=None):
        """Disable atmosphere within OpticalWavefrontPSF"""
        if logger:
            logger.info("Disabling atmosphere in OpticalWavefrontPSF")
        self.update_psf_params(r0=None, g1=None, g2=None, logger=logger)
        # note it in fitter_kwargs
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
        self.update_psf_params(r0=r0, g1=g1, g2=g2, logger=logger)
        # note it in fitter_kwargs
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
        # cut more stars if they fail shape measurement or error measurement, or if their shape values exceed a cut
        indx = ~np.any((self._shapes != self._shapes) +
                       (self._errors != self._errors) +
                       (np.abs(self._shapes) > self.max_shapes)
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

        if self.kwargs['guess_start']:
            # e0 = 0.12 + r0 ** -2 * 0.0082
            e0_mean = np.mean(self._shapes[:, 0])
            r0_guess = ((e0_mean - 0.12) / 0.0082) ** -0.5
            if logger:
                logger.info('Adjusting r0 to best fit guess of {0} from average size of {1}'.format(r0_guess, e0_mean))
            self.fitter_kwargs['r0'] = r0_guess
            if logger:
                logger.info('Starting analytic guesswork for fit.')
            self._analytic_fit(logger)

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

        if self.kwargs['fitter_algorithm'] == 'minuit':
            self._minuit_fit(logger=logger)
        elif self.kwargs['fitter_algorithm'] == 'lmfit':
            self._lmfit_fit(logger=logger)
        elif self.kwargs['fitter_algorithm'] == 'scipy':
            self._scipy_fit(logger=logger)
        else:
            raise NotImplementedError('fitter {0} not implemented!'.format(self.kwargs['fitter_algorithm']))

    def _stars_to_parameters(self, stars, logger=None):
        """Takes in stars and returns their zernikes, u, v, and focal x and focal y coordinates.
        """
        zernikes = []
        u = []
        v = []
        fx = []
        fy = []

        # need the interpolator to NOT be misaligned!
        if np.sum(np.abs(self.interp.misalignment)) > 0:
            if logger:
                logger.warn('Warning! Resetting misalignment to zero!')
            self.interp.misalignment = 0 * self.interp.misalignment

        stars_interpolated = self.interp.interpolateList(self.decaminfo.pixel_to_focalList(stars))
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

        return zernikes, u, v, fx, fy

    @staticmethod
    def _analytic_misalign_zernikes(zernikes, fx, fy, *vals):
        """Given zernikes, positions, and misalignment vector, misalign zernikes"""

        r0 = vals[0]
        misalignments = np.array(vals[1:])
        # stack r0
        params = np.hstack(((r0 * np.ones(len(zernikes)))[:, None], zernikes))

        # apply misalignment
        misalignment_arr = misalignments.reshape(8, 3)
        params[:, 1:] = params[:, 1:] + misalignment_arr[:, 0] + fx[:, None] * misalignment_arr[:, 2] + fy[:, None] * misalignment_arr[:, 1]
        return params

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
        params[:, 1:] = params[:, 1:] * 2
        psq = []
        pindx = []

        param_names = ['r0'] + ['z{0:02d}'.format(i) for i in range(4, 12)]
        # only allow up to quadratic in zernike, but allow r0 to be quartic
        for pi in range(9):
            psq.append(params[:, pi])
            pindx.append([param_names[pi]])
            for pj in range(pi, 9):
                psq.append(params[:, pi] * params[:, pj])
                pindx.append([param_names[pi], param_names[pj]])
                if pi == 0:
                    for pk in range(pj, 9):
                        psq.append(params[:, pi] * params[:, pj] * params[:, pk])
                        pindx.append([param_names[pi], param_names[pj], param_names[pk]])
        psq = np.array(psq)
        pindx = np.array(pindx)
        return psq, pindx

    @staticmethod
    def _analytic_params_to_shapes(psq):
        """These come from looking at a lasso linear regression. Hardcoded."""

        """
        e0 14 coeffs,
        std lasso 1.35e-02 std full 1.18e-02
        [None, None]:    +0.00e+00
        0 ['r0']:    +8.44e-01
        1 ['r0', 'r0']:    -6.57e-03
        2 ['r0', 'r0', 'r0']:    -9.20e-04
        19 ['r0', 'z04', 'z11']:    +2.37e-02
        54 ['r0', 'z11', 'z11']:    +2.39e-02
        56 ['z04', 'z04']:    +3.24e-02
        63 ['z04', 'z11']:    +2.06e-02
        65 ['z05', 'z05']:    +1.85e-02
        73 ['z06', 'z06']:    +1.88e-02
        80 ['z07', 'z07']:    +4.32e-02
        86 ['z08', 'z08']:    +4.32e-02
        91 ['z09', 'z09']:    +3.61e-02
        95 ['z10', 'z10']:    +3.68e-02
        98 ['z11', 'z11']:    +8.04e-02
        """

        e0 = 8.44e-01 * psq[0] + -6.57e-03 * psq[1] + -9.20e-04 * psq[2] + 2.37e-02 * psq[19] +  \
             2.39e-02 * psq[54] + 3.24e-02 * psq[56] + 2.06e-02 * psq[63] +  \
             1.85e-02 * psq[65] + 1.88e-02 * psq[73] +  \
             4.32e-02 * psq[80] + 4.32e-02 * psq[86] + 3.61e-02 * psq[91] +  \
             3.68e-02 * psq[95] + 8.04e-02 * psq[98]

        """
        e1 19 coeffs,
        std lasso 8.36e-03 std full 8.25e-03
        [None, None]:    +0.00e+00
        2 ['r0', 'r0', 'r0']:    +7.24e-06
        4 ['r0', 'r0', 'z05']:    +8.39e-06
        5 ['r0', 'r0', 'z06']:    -6.76e-06
        14 ['r0', 'z04', 'z06']:    +2.56e-03
        34 ['r0', 'z06', 'z11']:    +1.36e-02
        36 ['r0', 'z07', 'z07']:    -3.16e-03
        38 ['r0', 'z07', 'z09']:    +5.09e-03
        42 ['r0', 'z08', 'z08']:    +2.72e-03
        44 ['r0', 'z08', 'z10']:    +4.58e-03
        55 ['z04']:    -3.08e-05
        58 ['z04', 'z06']:    +3.98e-02
        72 ['z06']:    +2.20e-05
        78 ['z06', 'z11']:    +2.70e-02
        80 ['z07', 'z07']:    -7.62e-03
        82 ['z07', 'z09']:    +5.40e-02
        85 ['z08']:    -1.04e-05
        86 ['z08', 'z08']:    +8.60e-03
        88 ['z08', 'z10']:    +5.46e-02
        91 ['z09', 'z09']:    -5.76e-05
        """

        e1 = 7.24e-06 * psq[2] + 8.39e-06 * psq[4] + -6.76e-06 * psq[5] + 2.56e-03 * psq[14] +  \
             1.36e-02 * psq[34] + -3.16e-03 * psq[36] + 5.09e-03 * psq[38] +  \
             2.72e-03 * psq[42] + 4.58e-03 * psq[44] +  \
             -3.08e-05 * psq[55] + 3.98e-02 * psq[58] + 2.20e-05 * psq[72] +  \
             2.70e-02 * psq[78] + -7.62e-03 * psq[80] + 5.40e-02 * psq[82] +  \
             -1.04e-05 * psq[85] + 8.60e-03 * psq[86] + 5.46e-02 * psq[88] +  \
             -5.76e-05 * psq[91]

        """
        e2 17 coeffs,
        std lasso 9.10e-03 std full 9.02e-03
        [None, None]:    +0.00e+00
        2 ['r0', 'r0', 'r0']:    +4.55e-06
        4 ['r0', 'r0', 'z05']:    +7.57e-06
        5 ['r0', 'r0', 'z06']:    +1.77e-05
        8 ['r0', 'r0', 'z09']:    -6.87e-05
        13 ['r0', 'z04', 'z05']:    +2.75e-03
        27 ['r0', 'z05', 'z11']:    +1.28e-02
        37 ['r0', 'z07', 'z08']:    +6.48e-03
        39 ['r0', 'z07', 'z10']:    -4.96e-03
        43 ['r0', 'z08', 'z09']:    +4.04e-03
        57 ['z04', 'z05']:    +4.10e-02
        65 ['z05', 'z05']:    +7.61e-05
        71 ['z05', 'z11']:    +2.77e-02
        81 ['z07', 'z08']:    +1.53e-02
        83 ['z07', 'z10']:    -5.36e-02
        87 ['z08', 'z09']:    +5.54e-02
        92 ['z09', 'z10']:    +7.39e-05
        97 ['z11']:    +4.23e-05
        """

        e2 = 4.55e-06 * psq[2] + 7.57e-06 * psq[4] + 1.77e-05 * psq[5] + -6.87e-05 * psq[8] +  \
             2.75e-03 * psq[13] + 1.28e-02 * psq[27] + 6.48e-03 * psq[37] +  \
             -4.96e-03 * psq[39] + 4.04e-03 * psq[43] + 4.10e-02 * psq[57] +  \
             7.61e-05 * psq[65] + 2.77e-02 * psq[71] +  \
             1.53e-02 * psq[81] + -5.36e-02 * psq[83] + 5.54e-02 * psq[87] +  \
             7.39e-05 * psq[92] + 4.23e-05 * psq[97]

        shapes = np.vstack((e0, e1, e2)).T

        return shapes

    def _analytic_stars_to_shapes(self, stars, logger=None):

        # get zernikes and coordinates
        zernikes, u, v, fx, fy = self._stars_to_parameters(stars, logger=logger)

        # get misalignment from fitter_kwargs
        vals = []
        for key in self.keys:
            if key == 'g1' or key == 'g2':
                continue
            vals.append(self.fitter_kwargs[key])

        # apply misalignments
        params = self._analytic_misalign_zernikes(zernikes, fx, fy, *vals)

        # psq
        psq, pindx = self._analytic_parameterization(params)

        # shapes
        model_shapes = self._analytic_params_to_shapes(psq)

        return model_shapes


    def _analytic_chi2(self, vals_in, stars=None, logger=None):
        if not stars:
            stars = self._fit_stars
            actual_shapes = self._shapes
            errors = self._errors
        else:
            actual_shapes = self._measure_shapes(stars, logger=logger)
            errors = self._measure_shape_errors(stars, logger=logger)

        # based on fitter_kwargs translate vals_in to the misalignments
        vals = []
        key_i = 0
        for key in self.keys:
            if key == 'g1' or key == 'g2':
                continue
            if not self.fitter_kwargs['fix_' + key]:
                vals.append(vals_in[key_i])
                key_i += 1
            else:
                vals.append(self.fitter_kwargs[key])

        # get coordinates
        zernikes, u, v, fx, fy = self._stars_to_parameters(stars, logger=logger)

        # apply misalignments
        params = self._analytic_misalign_zernikes(zernikes, fx, fy, *vals)

        # psq
        psq, pindx = self._analytic_parameterization(params)

        # shapes
        model_shapes = self._analytic_params_to_shapes(psq)

        # chi2 of each star and shape
        chi2_long = np.sum((model_shapes - actual_shapes) ** 2 / errors ** 2, axis=0)

        # total chi2
        chi_squared = np.sum(self.weights * chi2_long) / (3 * len(stars) - len(vals_in)) / np.sum(self.weights)

        return chi_squared

    def _analytic_fit(self, logger=None):
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
            if key == 'g1' or key == 'g2':
                continue
            if not self.fitter_kwargs['fix_' + key]:
                val = self.fitter_kwargs[key]
                p0.append(val)
                if logger:
                    logger.debug('{0}:\t{1:+.2e}'.format(key, val))

        # minimize chi2
        if logger:
            logger.info('Starting analytic guess. Initial chi2 = {0:.2e}'.format(self._analytic_chi2(p0)))
        res = minimize(self._analytic_chi2, p0, args=(None, logger,))
        if logger:
            logger.warn('Analytic finished. Final chi2 = {0:.2e}'.format(self._analytic_chi2(res.x)))

        # update fitter_kwargs
        if logger:
            logger.info('Analytic guess parameters:')
        key_i = 0
        for key in self.keys:
            if key == 'g1' or key == 'g2':
                continue
            if not self.fitter_kwargs['fix_' + key]:
                val = res.x[key_i]
                key_i += 1
                self.fitter_kwargs[key] = val
                if logger:
                    logger.info('{0}:\t{1:.2e}'.format(key, val))

        # update fit parameters based on fitter_kwargs
        self.update_psf_params(logger=logger, **self.fitter_kwargs)

    def _scipy_gradient(self, vals, logger=None):

        # update fitter_kwargs with vals
        key_i = 0
        params = {}
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                params[key] = vals[key_i]
                key_i += 1
        gradients, gradients_l, stencils, fx, fy = self.chi2_gradient(
            stars=self._fit_stars, logger=logger, **params)

        return gradients

    def _scipy_chi2(self, vals, logger=None):

        # update fitter_kwargs with vals
        key_i = 0
        params = {}
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                params[key] = vals[key_i]
                key_i += 1

        # keep track in fitter_kwargs
        self.fitter_kwargs.update(params)
        # update psf params
        self.update_psf_params(logger=logger, **params)

        # get chi2
        reduced_chi2 = self.chi2(self._fit_stars, logger=logger)

        return reduced_chi2

    def _scipy_fit(self, logger=None):
        # TODO: add max_iterations
        """Fit interpolated PSF model to star data using minimize from scipy

        :param logger:          A logger object for logging debug info. [default: None]
        """
        from scipy.optimize import minimize

        # create x0
        x0 = []
        bounds = []
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                val = self.fitter_kwargs[key]
                x0.append(val)
                if 'limit_' + key in self.fitter_kwargs:
                    bounds.append(self.fitter_kwargs['limit_' + key])
                else:
                    bounds.append((None, None))

        if self.kwargs['use_gradient']:
            jac = self._scipy_gradient
        else:
            jac = None

        # minimize chi2
        if logger:
            logger.info("Start fitting Optical fit using scipy. Initial chi2 = {0:.2e}".format(self._scipy_chi2(x0)))
        res = minimize(self._scipy_chi2, x0, args=(logger,), jac=jac, bounds=bounds, options={'maxiter': self.kwargs['max_iterations']})

        if logger:
            logger.warn('Optical fit with scipy finished. Final chi2 = {0:.2e}. Success is {1} and message is {2}'.format(self._scipy_chi2(res.x), res.success, res.message))

        # update fitter_kwargs
        if logger:
            logger.info('Optical fit from scipy parameters:')
        key_i = 0
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                val = res.x[key_i]
                self.fitter_kwargs[key] = val
                if logger:
                    logger.info('{0}:\t{1:.2e}'.format(key, val))

                # TODO: try putting in errors based on jacobian?

                key_i += 1

        # update fit parameters based on fitter_kwargs
        self.update_psf_params(logger=logger, **self.fitter_kwargs)

    def _minuit_fit(self, logger=None):
        """Fit interpolated PSF model to star data using iminuit implementation of Minuit

        :param logger:          A logger object for logging debug info. [default: None]
        """
        from iminuit import Minuit
        if logger:
            logger.info("Start fitting Optical fit using minuit")
        if self.kwargs['use_gradient']:
            grad_fcn = self._minuit_gradient
        else:
            grad_fcn = None
        self._minuit = Minuit(self._minuit_chi2, grad_fcn=grad_fcn, forced_parameters=self.keys, **self.fitter_kwargs)
        # run the fit and solve! This will update the interior parameters
        if self.kwargs['max_iterations'] > 0:
            if logger:
                logger.info("Running migrad for {0} steps!".format(self.kwargs['max_iterations']))
            self._minuit.migrad(ncall=self.kwargs['max_iterations'])
            # self._hesse = self._minuit.hesse()
            # self._minos = self._minuit.minos()
            # these are the best fit parameters

            # save params and errors to the kwargs
            self.fitter_kwargs.update(self._minuit.fitarg)
            if logger:
                logger.info('Optical fit from minuit parameters:')
                for key in self.keys:
                    if not self.fitter_kwargs['fix_' + key]:
                        val = self.fitter_kwargs[key]
                        err = self.fitter_kwargs['error_' + key]
                        logstring = '{0}:\t{1:+.2e}\t{2:+.2e}'.format(key, val, err)
                        logger.info(logstring)
            # update fit parameters based on fitter_kwargs
            self.update_psf_params(logger=logger, **self.fitter_kwargs)
        else:
            if logger:
                logger.info('User specified {0} steps, so moving on without running migrad'.format(self.kwargs['max_iterations']))

    def _lmfit_resid(self, lmparams, logger=None):

        # convert lmparams instance
        params = lmparams.valuesdict()

        # keep track in fitter_kwargs
        self.fitter_kwargs.update(params)
        # update psf params
        self.update_psf_params(logger=logger, **params)
        # get chi (not chi2s!)
        reduced_chi2, dof, chi2, indx, chi_l = self.chi2(self._fit_stars, full=True, logger=logger)

        # chi_l is shaped (Nstar, Nshapes)
        # using indx, remove Nans, and multiply by sqrt(weight) since usually weight * chi2
        chi_flat = (np.sqrt(self.weights[None]) * chi_l[indx]).flatten()
        return chi_flat

    def _lmfit_fit(self, logger=None):
        """Fit interpolated PSF model to star data using lmfit implementation of Levenberg-Marquardt minimization

        :param logger:          A logger object for logging debug info. [default: None]
        """
        import lmfit
        if logger:
            logger.info("Start fitting Optical fit using lmfit")

        # create lmparameters instance
        params = lmfit.Parameters()
        # Order of params is important!
        # step through keys
        for key in self.keys:
            value = self.fitter_kwargs[key]
            vary = not self.fitter_kwargs['fix_' + key]
            if 'limit_' + key in self.fitter_kwargs:
                min, max = self.fitter_kwargs['limit_' + key]
            else:
                min, max = (None, None)
            params.add(key, value=value, vary=vary, min=min, max=max)

        # fit params
        results = lmfit.minimize(self._lmfit_resid, params, args=(logger,), maxfev=self.kwargs['max_iterations'])

        # update fitter_kwargs
        if logger:
            logger.info('Optical fit from lmfit parameters:')
        key_i = 0
        for key in self.keys:
            if not self.fitter_kwargs['fix_' + key]:
                val = results.params.valuesdict()[key]
                self.fitter_kwargs[key] = val
                if logger:
                    logstring = '{0}:\t{1:+.2e}'.format(key, val)

                if results.errorbars:
                    err = results.params[key].stderr
                    self.fitter_kwargs['error_' + key] = err
                    logstring += '\t{0:.2e}'.format(err)

                if logger:
                    logger.info(logstring)

                key_i += 1



    def _update_psf_params(self, **kwargs):
        # backward compatability
        return self.update_psf_params(**kwargs)

    def update_psf_params(self,
                          r0=np.nan, g1=np.nan, g2=np.nan,
                          z04d=np.nan, z04x=np.nan, z04y=np.nan,
                          z05d=np.nan, z05x=np.nan, z05y=np.nan,
                          z06d=np.nan, z06x=np.nan, z06y=np.nan,
                          z07d=np.nan, z07x=np.nan, z07y=np.nan,
                          z08d=np.nan, z08x=np.nan, z08y=np.nan,
                          z09d=np.nan, z09x=np.nan, z09y=np.nan,
                          z10d=np.nan, z10x=np.nan, z10y=np.nan,
                          z11d=np.nan, z11x=np.nan, z11y=np.nan,
                          logger=None, **kwargs):
        # update model
        old_r0 = self.model.kolmogorov_kwargs['r0']
        old_g1 = self.model.g1
        old_g2 = self.model.g2
        if r0 == r0:
            self.model.kolmogorov_kwargs['r0'] = r0
            self.model.kwargs['r0'] = r0
        if g1 == g1:
            self.model.g1 = g1
        if g2 == g2:
            self.model.g2 = g2
        # update the misalignment
        misalignment = np.array([
                  [z04d, z04x, z04y],
                  [z05d, z05x, z05y],
                  [z06d, z06x, z06y],
                  [z07d, z07x, z07y],
                  [z08d, z08x, z08y],
                  [z09d, z09x, z09y],
                  [z10d, z10x, z10y],
                  [z11d, z11x, z11y],
                  ])
        old_misalignment = self.interp.misalignment
        # make sure we don't update the fixed parameters
        misalignment = np.where(misalignment == misalignment, misalignment, old_misalignment)
        # now we can update the interp
        self.interp.misalignment = misalignment

        if logger:
            misalignment_print = np.array([
                      [r0, g1, g2],
                      [z04d, z04x, z04y],
                      [z05d, z05x, z05y],
                      [z06d, z06x, z06y],
                      [z07d, z07x, z07y],
                      [z08d, z08x, z08y],
                      [z09d, z09x, z09y],
                      [z10d, z10x, z10y],
                      [z11d, z11x, z11y],
                      ])
            misalignment_print = np.where(misalignment_print == misalignment_print, misalignment_print, 0)
            try:
                old_misalignment_print = np.vstack((
                    np.array([[old_r0, old_g1, old_g2]]),
                    old_misalignment))
                old_misalignment_print = np.where(old_misalignment_print == old_misalignment_print, old_misalignment_print, 0)
                logger.debug('New - Old misalignment is \n{0}'.format(misalignment_print - old_misalignment_print))
                logger.debug('New misalignment is \n{0}'.format(misalignment_print))
            except:
                # old misalignment could be None on first iteration
                logger.debug('New misalignment is \n{0}'.format(misalignment_print))

    def chi2(self, stars=None, full=False, logger=None):
        if not stars:
            stars = self._fit_stars
            actual_shapes = self._shapes
            errors = self._errors
        else:
            actual_shapes = self._measure_shapes(stars, logger=logger)
            errors = self._measure_shape_errors(stars, logger=logger)

        shapes = self._measure_shapes(self.drawStarList(stars), logger=logger)

        # calculate chisq
        chi_l = (shapes - actual_shapes) / (self.kwargs['error_estimate'] * errors)
        chi2_l = np.square(chi_l)
        indx = ~np.any(chi2_l != chi2_l, axis=1)
        chi2 = np.sum(chi2_l[indx], axis=0)
        dof = sum(indx)
        reduced_chi2 = np.sum(self.weights * chi2) * 1. / dof / np.sum(self.weights)

        if logger:
            if sum(indx) != len(indx):
                logger.info('Warning! We are using {0} stars out of {1} stars'.format(sum(indx), len(indx)))
            logger.debug('chi2 array:')
            logger.debug('chi2 summed: {0}'.format(reduced_chi2))
            logger.debug('chi2 in each shape: {0}'.format(chi2))
            logger.debug('chi2 with weights: {0}'.format(self.weights * chi2))
            logger.debug('sum of weights times dof: {0}'.format(dof * np.sum(self.weights)))
            logger.debug('chi2 for each star:\n{0}'.format(chi2_l))

            # TODO: extract from psf parameters
            log = ['\n',
                    '*******************************************************************************\n',
                    '* time \t|\t {0:.3e} \t|\t ncalls  \t|\t {1:04d} \t      *\n'.format(time() - self._time, self._n_iter),
                    '*******************************************************************************\n',
                    '*  \t|\t d \t\t|\t x \t\t|\t y \t      *\n',
                    '*******************************************************************************\n',
                    '* size \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(
                        self.model.kwargs['r0'], self.model.g1, self.model.g2),]
            for i in range(0, 8):
                log.append('* z{0}   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(
                    i + 4, self.interp.misalignment[i][0], self.interp.misalignment[i][1],
                    self.interp.misalignment[i][2]))
            log += [
                    '*******************************************************************************\n',
                    '* chi2 \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(*(chi2 / dof)),
                    '*******************************************************************************',
                    ]
            if self._n_iter % 50 == 0:
                logger.info(''.join(log))
            else:
                logger.debug(''.join(log))
        self._n_iter += 1

        if full:
            return reduced_chi2, dof, chi2, indx, chi_l
        else:
            return reduced_chi2

    # TODO: stars, logger, option for doing all, fix, step_sizes, keys
    # would be nice to make this a static function
    def chi2_gradient(self, stars,
                 r0, g1, g2,
                 z04d, z04x, z04y,
                 z05d, z05x, z05y,
                 z06d, z06x, z06y,
                 z07d, z07x, z07y,
                 z08d, z08x, z08y,
                 z09d, z09x, z09y,
                 z10d, z10x, z10y,
                 z11d, z11x, z11y,
                 logger=None, calculate_all=False):
        """Let's be smarter than is wise.
        Fundamentally this is what we are testing:
            d e_i / d delta = d e_i / d z d z / d delta
            d e_i / d theta = d e_i / d z d z / d theta

        but z = z0 + delta + theta x
        so d z / d delta = 1, d z / d theta = x

        therefore, for the delta terms, all I have to do is calculate d ei / d delta and I get the x and y terms for free!

        calculate_all will manually calculate gradients of "z*x" and "z*y" even though they are proportional to delta.
        """

        gradients_l = []
        stencils = []

        # extract focal coordinates
        fx = []
        fy = []
        for star in self._fit_stars:
            fx.append(star.data.properties['focal_x'])
            fy.append(star.data.properties['focal_y'])
        fx = np.array(fx)
        fy = np.array(fy)

        # step through each parameter
        for key, step_size in zip(self.keys, self.step_sizes):

            if calculate_all:
                stencil_gradient = True

            elif self.fitter_kwargs['fix_{0}'.format(key)]:
                # if a parameter is fixed, skip calculating it!
                gradients_l.append(np.zeros((len(self._fit_stars), 3)))
                stencil_gradient = False

            elif key[0] == 'z' and (key[-1] == 'x' or key[-1] == 'y'):
                # TODO: can go bad if we fixed delta but not x and y
                if key[-1] == 'x':
                    gradients_l.append(gradients_l[-1] * fy[:, None])
                elif key[-1] == 'y':
                    gradients_l.append(gradients_l[-2] * fx[:, None])
                stencil_gradient = False

            else:
                stencil_gradient = True

            if stencil_gradient:
                stencil_chi2_l = []

                # step through +- stencil stencil
                for term in [1, -1]:
                    params = dict(r0=r0, g1=g1, g2=g2,
                                  z04d=z04d, z04x=z04x, z04y=z04y,
                                  z05d=z05d, z05x=z05x, z05y=z05y,
                                  z06d=z06d, z06x=z06x, z06y=z06y,
                                  z07d=z07d, z07x=z07x, z07y=z07y,
                                  z08d=z08d, z08x=z08x, z08y=z08y,
                                  z09d=z09d, z09x=z09x, z09y=z09y,
                                  z10d=z10d, z10x=z10x, z10y=z10y,
                                  z11d=z11d, z11x=z11x, z11y=z11y,
                                  )
                    params[key] = params[key] + term * step_size

                    # update psf
                    self.update_psf_params(**params)
                    reduced_chi2, dof, chi2, indx, chi2_l = self.chi2(self._fit_stars, full=True)
                    # discount these calls for the purposes of displaying _n_iter
                    self._n_iter -= 1

                    stencil_chi2_l.append(chi2_l)

                # get slope
                # fx = [f(x + h) - f(x - h)] / (2h)
                stencil_values = np.array([0.5, -0.5])
                stencil_chi2_l = np.array(stencil_chi2_l)  # (n_stencil, Nstar, Nshapes)
                stencils.append(stencil_chi2_l)
                gradients_l.append(np.sum(stencil_values[:, None, None] * stencil_chi2_l, axis=0) / step_size)

        stencils = np.array(stencils)
        gradients_l = np.array(gradients_l)
        # convert gradients_l into gradients
        # recall that reduced_chi2 = np.sum(self.weights * np.sum(chi2_l[indx], axis=0)) * 1. / sum(indx) / np.sum(self.weights)
        gradients = np.sum(self.weights[None] * np.nansum(gradients_l, axis=1), axis=1) * 1. / len(self._fit_stars) / np.sum(self.weights)

        # print gradient log info
        if logger:
            log = ['\n',
                    '**************************************************************************************\n',
                    '* time        \t|\t {0:.3e} \t|\t ncalls  \t|\t {1:04d} \t      *\n'.format(time() - self._time, self._n_iter),
                    '**************************************************************************************\n',
                    '*         \t|\t d \t\t|\t x \t\t|\t y \t      *\n',
                    '**************************************************************************************\n',
                    '* d chi2 d size \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[0], gradients[1], gradients[2]),
                    '* d chi2 d z4   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[3], gradients[4], gradients[5]),
                    '* d chi2 d z5   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[6], gradients[7], gradients[8]),
                    '* d chi2 d z6   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[9], gradients[10], gradients[11]),
                    '* d chi2 d z7   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[12], gradients[13], gradients[14]),
                    '* d chi2 d z8   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[15], gradients[16], gradients[17]),
                    '* d chi2 d z9   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[18], gradients[19], gradients[20]),
                    '* d chi2 d z10  \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[21], gradients[22], gradients[23]),
                    '* d chi2 d z11  \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(gradients[24], gradients[25], gradients[26]),
                    '**************************************************************************************\n',
                    ]
            logger.debug(''.join(log))

        return gradients, gradients_l, stencils, fx, fy

    def _minuit_gradient(self,
                         r0, g1, g2,
                         z04d, z04x, z04y,
                         z05d, z05x, z05y,
                         z06d, z06x, z06y,
                         z07d, z07x, z07y,
                         z08d, z08x, z08y,
                         z09d, z09x, z09y,
                         z10d, z10x, z10y,
                         z11d, z11x, z11y,
                         ):

        gradients, gradients_l, stencils, fx, fy = self.chi2_gradient(
            self._fit_stars,
            r0, g1, g2,
            z04d, z04x, z04y,
            z05d, z05x, z05y,
            z06d, z06x, z06y,
            z07d, z07x, z07y,
            z08d, z08x, z08y,
            z09d, z09x, z09y,
            z10d, z10x, z10y,
            z11d, z11x, z11y,
            logger=self._logger)

        return gradients

    def _minuit_chi2(self,
                     r0, g1, g2,
                     z04d, z04x, z04y,
                     z05d, z05x, z05y,
                     z06d, z06x, z06y,
                     z07d, z07x, z07y,
                     z08d, z08x, z08y,
                     z09d, z09x, z09y,
                     z10d, z10x, z10y,
                     z11d, z11x, z11y,
                     ):

        # update fitter_kwargs and pass along
        params = dict(r0=r0, g1=g1, g2=g2,
                      z04d=z04d, z04x=z04x, z04y=z04y,
                      z05d=z05d, z05x=z05x, z05y=z05y,
                      z06d=z06d, z06x=z06x, z06y=z06y,
                      z07d=z07d, z07x=z07x, z07y=z07y,
                      z08d=z08d, z08x=z08x, z08y=z08y,
                      z09d=z09d, z09x=z09x, z09y=z09y,
                      z10d=z10d, z10x=z10x, z10y=z10y,
                      z11d=z11d, z11x=z11x, z11y=z11y,
                      )
        # keep track in fitter_kwargs
        self.fitter_kwargs.update(params)
        # update psf params
        self.update_psf_params(logger=self._logger, **params)

        reduced_chi2 = self.chi2(self._fit_stars, logger=self._logger)

        return reduced_chi2

    def drawStarList(self, stars):
        """Generate PSF images for given stars.

        :param stars:       List of Star instances holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           List of Star instances with its image filled with rendered PSF
        """
        # put in the focal coordinates
        stars = self.decaminfo.pixel_to_focalList(stars)
        # Interpolate parameters to this position/properties:
        stars = self.interp.interpolateList(stars)
        # Render the image
        stars = [self.model.draw(star) for star in stars]

        return stars

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        # put in the focal coordinates
        star = self.decaminfo.pixel_to_focal(star)
        # Interpolate parameters to this position/properties:
        star = self.interp.interpolate(star)
        # Render the image
        star = self.model.draw(star)
        return star

    def getProfile(self, star):
        """Get galsim profile for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Galsim profile
        """
        # put in the focal coordinates
        star = self.decaminfo.pixel_to_focal(star)
        # Interpolate parameters to this position/properties:
        star = self.interp.interpolate(star)
        # get the profile
        prof = self.model.getProfile(star.fit.params).shift(star.fit.center) * star.fit.flux
        return prof

    def getParams(self, star):
        """Get params for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Params
        """
        # put in the focal coordinates
        star = self.decaminfo.pixel_to_focal(star)
        # Interpolate parameters to this position/properties:
        star = self.interp.interpolate(star)
        return star.fit.params

    def _finish_write(self, fits, extname, logger):
        """Save the misalignment parameters

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension to write to.
        :param logger:      A logger object for logging debug info.
        """
        # save relevant entries in fitter_kwargs -- value and errors
        keys = ['r0', 'g1', 'g2']
        for zi in range(4, 12):
            for dxy in ['d', 'x', 'y']:
                zkey = 'z{0:02d}{1}'.format(zi, dxy)
                keys.append(zkey)
        prefixes = ['', 'fix_', 'error_']
        data_types = [float, bool, float]

        # create dtype
        dtypes = [('weights', '3f4')]
        for key in keys:
            for prefix, data_type in zip(prefixes, data_types):
                combined_key = '{0}{1}'.format(prefix, key)
                dtypes.append((combined_key, data_type))

        data = np.zeros(1, dtype=dtypes)
        data['weights'][0] = self.weights
        for key in keys:
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
        keys = ['r0', 'g1', 'g2']
        for zi in range(4, 12):
            for dxy in ['d', 'x', 'y']:
                zkey = 'z{0:02d}{1}'.format(zi, dxy)
                keys.append(zkey)
        prefixes = ['', 'fix_', 'error_']
        for key in keys:
            for prefix in prefixes:
                combined_key = '{0}{1}'.format(prefix, key)
                self.fitter_kwargs[combined_key] = data[combined_key][0]

        # do the awkward misalignment stuff
        self.update_psf_params(logger=logger, **self.fitter_kwargs)

        self._time = time()
        self._n_iter = 0

        # load model and interp
        self.model = Model.read(fits, extname + '_model', logger)
        self.interp = Interp.read(fits, extname + '_interp', logger)
