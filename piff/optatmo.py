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
        profiles = [self.optpsf.getProfile(star) for star in self.stars]
        # try cleaning out starfits
        self.stars = [Star(star.data, None) for star in self.stars]

        # fit AtmoPSF
        if logger:
            logger.info("Fitting AtmospherePSF")
        # TODO: want to make sure when we draw stars in atmopsf that we draw the same with self.drawStar
        self.atmopsf.fit(self.stars, wcs, pointing, profiles=profiles, logger=logger)

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
    def __init__(self, knn_file_name, knn_extname, max_iterations=300, n_fit_stars=0, error_estimate=0.001, pupil_plane_im=None,  extra_interp_properties=None, weights=np.array([0.5, 1, 1]), max_shapes=np.array([1e10, 1e10, 1e10]), fitter_kwargs={}, interp_kwargs={}, model_kwargs={}, engine='galsim_fast', template='des', fitter_algorithm='minuit', cut_ccds=[], guess_start=False, logger=None):
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
        :param cut_ccds:                    list of ccds to remove from decaminfo
        :param guess_start:                 if True, will adjust fitter kwargs for best guess
        """

        # TODO: trying with distance weighting to 40 nearest neighbors
        self.interp_kwargs = {'n_neighbors': 40, 'weights': 'distance', 'algorithm': 'auto'}
        self.interp_kwargs.update(interp_kwargs)

        # it turns out this part can also be slow!
        if logger:
            logger.debug("Making interp")
        self.interp = DECamWavefront(knn_file_name, knn_extname, logger=logger, **self.interp_kwargs)

        if logger:
            logger.debug("Making DECamInfo")
        self.decaminfo = DECamInfo(cut_ccds=cut_ccds)

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
            }

        # load up the model after kwargs are set
        if logger:
            logger.debug("Loading optical engine")
        self._engines = ['galsim', 'galsim_fast']
        self._model(template=template, engine=engine, **model_kwargs)


        # put in the variable names and initial values
        self.fitter_kwargs = {
            # note: r0 is in meters. Use 0.976 * lam / r0 = fwhm. This is odd, but lam is in nm, r0 is in m, and fwhm is in pixels?
            'r0': 0.15, 'fix_r0': False,   'limit_r0': (0.01, 0.25), 'error_r0': 1e-2,
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
                # limit
                self.fitter_kwargs['limit_' + zkey] = (-2, 2)
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
        else:
            raise NotImplementedError('fitter {0} not implemented.'.format(self.kwargs['fitter_algorithm']))

        # update with user kwargs
        self.fitter_kwargs.update(fitter_kwargs)

        self._update_psf_params(logger=logger, **self.fitter_kwargs)

        self._time = time()

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
        self._update_psf_params(r0=None, g1=None, g2=None, logger=logger)
        # double plus sure we disable it
        self.model.kwargs['r0'] = None
        self.model.kolmogorov_kwargs['r0'] = None
        self.model.g1 = None
        self.model.g2 = None

    def enable_atmosphere(self, r0, g1, g2, logger=None):
        """Turn the atmosphere back on.

        :param r0: seeing parameter
        :param g1,2: ellipticity
        :param logger: logger

        Note: r0, g1, g2 might still be fixed.
        """
        self.model.kwargs['r0'] = r0
        self.model.kolmogorov_kwargs['r0'] = r0
        self.model.g1 = g1
        self.model.g2 = g2

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
                logger.debug("Measuring error of star {0}".format(star_i))
            try:
                sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2 = hsm_error(star, logger=logger)
                errors.append([sigma_e0, sigma_e1, sigma_e2])
            except (ValueError, ModelFitError, RuntimeError):
                # hsm nan'd out
                if logger:
                    logger.debug("Star failed moment parameter; setting errors to nan!")
                errors.append([np.nan, np.nan, np.nan])
        errors = np.array(errors)
        return errors

    @staticmethod
    def _measure_shapes(stars, logger=None):
        """Measure shapes using piff.util.hsm
        """
        shapes = []
        for star_i, star in enumerate(stars):
            if logger:
                logger.debug("Measuring shape of star {0}".format(star_i))
            flux, u0, v0, sigma, g1, g2, flag = hsm(star, logger=logger)
            if flag != 0:
                # failed fit!
                if logger:
                    logger.debug("Star failed moment parameter; setting shapes to nan!")
                shapes.append([np.nan, np.nan, np.nan])
            else:
                # we want sigma^2, not sigma for size, and normalized ellipticity
                shapes.append([sigma ** 2, sigma ** 2 * g1, sigma ** 2 * g2])
        shapes = np.array(shapes)

        return shapes

    def _fit_init(self, stars, wcs, pointing, profiles=[], logger=None):
        """Sets up all the tasks you would need to start the fit. Useful for when you want to test PSF functionality but don't want to run the expensive fit.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param profiles:        List of Galsim profiles to convolve with model during fit. [default: []]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.warning("Start fitting OpticalWavefrontPSF using %s stars", len(stars))
        self.wcs = wcs
        self.pointing = pointing
        self.stars = stars

        # TODO: deal with convolve profiles in practice and using _fit_profiles
        convolve_profiles = len(profiles) and getattr(self.model, "getProfile", False)

        self._fit_stars = stars
        choice = np.arange(len(stars))
        if self.kwargs['n_fit_stars'] and self.kwargs['n_fit_stars'] < len(self._fit_stars):
            choice = np.sort(np.random.choice(len(self._fit_stars), self.kwargs['n_fit_stars'], replace=False))
            self._fit_stars = [stars[i] for i in choice]
            if convolve_profiles:
                self._fit_profiles = [profiles[i] for i in choice]
            else:
                self._fit_profiles = None
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
        if convolve_profiles:
            self._fit_profiles = [profile for profile, ind in zip(self._fit_profiles, indx) if ind]
        else:
            self._fit_profiles = None
        # make use_stars
        use_stars = np.in1d(np.arange(len(stars)), choice[indx], assume_unique=True)

        # modify stars to include whether used in fit.
        # This modifies both self.stars and stars
        for star, use_star in zip(self.stars, use_stars):
            star.data.properties['used_in_optical_wavefront'] = use_star

        if self.kwargs['guess_start']:
            r0_guess = (np.mean(self._shapes[:, 0]) / 0.004) ** -0.5
            if logger:
                logger.info('Adjusting r0 to best fit guess of {0} from average size of {1}'.format(r0_guess, np.mean(self._shapes[:, 0])))
            self.fitter_kwargs['r0'] = r0_guess
            if logger:
                logger.info('Starting analytic guesswork for fit.')
            self._analytic_fit(logger)

        self._logger = logger

    def fit(self, stars, wcs, pointing,
            profiles=[], logger=None):
        """Fit interpolated PSF model to star data using standard sequence of operations.

        :param stars:           A list of Star instances.
        :param wcs:             A dict of WCS solutions indexed by chipnum.
        :param pointing:        A galsim.CelestialCoord object giving the telescope pointing.
                                [Note: pointing should be None if the WCS is not a CelestialWCS]
        :param profiles:        List of Galsim profiles to convolve with model during fit. [default: []]
        :param logger:          A logger object for logging debug info. [default: None]
        """
        self._fit_init(stars, wcs, pointing, profiles=profiles, logger=logger)

        # based on fitter, do _minuit_fit or (TODO) _lmfit_fit
        if self.kwargs['fitter_algorithm'] == 'minuit':
            self._minuit_fit(logger=logger)
        else:
            raise NotImplementedError('fitter {0} not yet implmented!'.format(self.kwargs['fitter_algorithm']))

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
        std lasso 6.71e-03 std full 5.91e-03
        [None, None]:    +0.00e+00
        0 ['r0']:    +4.23e-01
        1 ['r0', 'r0']:    -3.71e-03
        2 ['r0', 'r0', 'r0']:    -3.78e-04
        19 ['r0', 'z04', 'z11']:    +1.24e-02
        54 ['r0', 'z11', 'z11']:    +1.17e-02
        56 ['z04', 'z04']:    +1.57e-02
        63 ['z04', 'z11']:    +1.05e-02
        65 ['z05', 'z05']:    +9.37e-03
        73 ['z06', 'z06']:    +9.41e-03
        80 ['z07', 'z07']:    +2.08e-02
        86 ['z08', 'z08']:    +2.06e-02
        91 ['z09', 'z09']:    +1.67e-02
        95 ['z10', 'z10']:    +1.69e-02
        98 ['z11', 'z11']:    +4.13e-02
        """
        e0 = psq[0] * 4.23e-1 + psq[1] * -3.71e-3 + psq[2] * -3.78e-4 + psq[19] * 1.24e-2 + psq[54] * 1.17e-2 + \
             psq[56] * 1.57e-2 + psq[63] * 1.05e-2 + psq[65] * 9.37e-3 + psq[73] * 9.41e-3 + \
             psq[80] * 2.08e-2 + psq[86] * 2.06e-2 + psq[91] * 1.67e-2 + psq[95] * 1.69e-2 + psq[98] * 4.13e-2

        """
        e1 14 coeffs,
        std lasso 2.14e-03 std full 2.12e-03
        [None, None]:    +0.00e+00
        4 ['r0', 'r0', 'z05']:    +1.02e-06
        6 ['r0', 'r0', 'z07']:    +8.45e-07
        14 ['r0', 'z04', 'z06']:    +6.00e-04
        34 ['r0', 'z06', 'z11']:    +3.31e-03
        36 ['r0', 'z07', 'z07']:    -8.06e-04
        38 ['r0', 'z07', 'z09']:    +1.22e-03
        42 ['r0', 'z08', 'z08']:    +7.99e-04
        44 ['r0', 'z08', 'z10']:    +1.14e-03
        58 ['z04', 'z06']:    +9.91e-03
        78 ['z06', 'z11']:    +6.72e-03
        80 ['z07', 'z07']:    -1.72e-03
        82 ['z07', 'z09']:    +1.30e-02
        86 ['z08', 'z08']:    +1.74e-03
        88 ['z08', 'z10']:    +1.33e-02
        """
        e1 = psq[4] * 1.02e-6 + psq[6] * 8.45e-7 + psq[14] * 6.0e-04 + psq[34] * 3.31e-3 + psq[36] * -8.06e-4 + \
             psq[38] * 1.22e-3 + psq[42] * 7.99e-4 + psq[44] * 1.14e-3 + psq[58] * 9.91e-3 + psq[78] * 6.72e-3 + \
             psq[80] * -1.72e-3 + psq[82] * 1.3e-2 + psq[86] * 1.74e-3 + psq[88] * 1.33e-2

        """
        e2 13 coeffs,
        std lasso 2.25e-03 std full 2.23e-03
        [None, None]:    +0.00e+00
        2 ['r0', 'r0', 'r0']:    +3.30e-07
        6 ['r0', 'r0', 'z07']:    +1.23e-05
        9 ['r0', 'r0', 'z10']:    +3.79e-06
        13 ['r0', 'z04', 'z05']:    +7.40e-04
        27 ['r0', 'z05', 'z11']:    +3.09e-03
        37 ['r0', 'z07', 'z08']:    +1.60e-03
        39 ['r0', 'z07', 'z10']:    -1.17e-03
        43 ['r0', 'z08', 'z09']:    +1.21e-03
        57 ['z04', 'z05']:    +9.85e-03
        71 ['z05', 'z11']:    +6.97e-03
        81 ['z07', 'z08']:    +3.69e-03
        83 ['z07', 'z10']:    -1.34e-02
        87 ['z08', 'z09']:    +1.34e-02
        """
        e2 = psq[2] * 3.3e-7 + psq[6] * 1.23e-5 + psq[9] * 3.79e-6 + psq[13] * 7.4e-4 + psq[27] * 3.09e-3 + \
             psq[37] * 1.6e-3 + psq[39] * -1.17e-3 + psq[43] * 1.21e-3 + psq[57] * 9.85e-3 + psq[71] * 6.97e-3 + \
             psq[81] * 3.69e-3 + psq[83] * -1.34e-2 + psq[87] * 1.34e-2

        shapes = np.vstack((e0, e1, e2)).T

        return shapes

    def _analytic_fit(self, logger=None):
        """Fit interpolated PSF model to star data by analytic relation. Should get us pretty close.

        :param logger:          A logger object for logging debug info. [default: None]

        :returns guess_values:  Reasonable first guesses for misalignment
        """
        from scipy.optimize import minimize

        zernikes, u, v, fx, fy = self._stars_to_parameters(self._fit_stars, logger=logger)

        # dynamically define function based on fixed parameters
        def chi2(vals_in):
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

            # apply misalignments
            params = self._analytic_misalign_zernikes(zernikes, fx, fy, *vals)

            # psq
            psq, pindx = self._analytic_parameterization(params)

            # shapes
            model_shapes = self._analytic_params_to_shapes(psq)

            # chi2 of each star and shape
            chi2_long = np.sum((model_shapes - self._shapes) ** 2 / self._errors ** 2, axis=0)

            # total chi2
            chi_squared = np.sum(self.weights * chi2_long) / (3 * len(self._fit_stars) - len(vals_in)) / np.sum(self.weights)

            return chi_squared

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
                    logger.debug('{0}:\t{1:.2e}'.format(key, val))

        # minimize chi2
        if logger:
            logger.info('Starting analytic guess. Initial chi2 = {0:.2e}'.format(chi2(p0)))
        res = minimize(chi2, p0)
        if logger:
            logger.warn('Analytic finished. Final chi2 = {0:.2e}'.format(chi2(res.x)))

        if logger:
            logger.info('Analytic guess parameters:')
        # update fitter_kwargs
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
        self._update_psf_params(logger=logger, **self.fitter_kwargs)

    def _minuit_fit(self, logger=None):
        """Fit interpolated PSF model to star data using iminuit implementation of Minuit

        :param logger:          A logger object for logging debug info. [default: None]
        """
        from iminuit import Minuit

        self._n_iter = 0

        if logger:
            logger.debug("Creating minuit object")
        self._minuit = Minuit(self._fit_func, grad_fcn=self._grad_func_minuit, forced_parameters=self.keys, **self.fitter_kwargs)
        # run the fit and solve! This will update the interior parameters
        if self.kwargs['max_iterations'] > 0:
            if logger:
                logger.info("Running migrad for {0} steps!".format(self.kwargs['max_iterations']))
            self._minuit.migrad(ncall=self.kwargs['max_iterations'])
            # self._hesse = self._minuit.hesse()
            # self._minos = self._minuit.minos()
            # these are the best fit parameters

            # update params to best values
            if logger:
                logger.debug("Minuit Fitargs:\n*****\n")
                for key in self._minuit.fitarg:
                    logger.debug("{0}: {1}\n".format(key, self._minuit.fitarg[key]))
                logger.debug("Minuit Values:\n*****\n")
                for key in self._minuit.values:
                    logger.debug("{0}: {1}\n".format(key, self._minuit.values[key]))
            self._update_psf_params(logger=logger, **self._minuit.values)

            # save params and errors to the kwargs
            self.fitter_kwargs.update(self._minuit.fitarg)
            if logger:
                logger.debug("Minuit kwargs are now:\n*****\n")
                for key in self.fitter_kwargs:
                    logger.debug("{0}: {1}\n".format(key, self.fitter_kwargs[key]))
        else:
            if logger:
                logger.info('User specified {0} steps, so moving on without running migrad'.format(self.kwargs['max_iterations']))

    def _lmfit_fit(self, stars, logger=None):
        """Fit interpolated PSF model to star data using lmfit implementation of Levenberg-Marquardt minimization

        :parma stars:           Stars used in fit
        :param logger:          A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.info("Start fitting Optical fit using lmfit and %s stars", len(stars))
        # import lmfit
        # results = lmfit.minimize(resid_func, params, args)
        raise NotImplementedError("lmfit not currently implemented")

    def _update_psf_params(self,
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
        # TODO: should these ONLY be r0 == r0 nan checks? I am worried these won't get reset when I load. On the other hand, we also save the model and interp
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

    def chi2(self, stars, full=False, logger=None):
        # using shapes, return chi2 of model
        shapes = self._measure_shapes(self.drawStarList(self._fit_stars), logger=logger)

        # calculate chisq
        # chi2 = np.sum(np.square((shapes - self._shapes) / self.kwargs['error_estimate']), axis=0)
        # dof = shapes.size
        chi2_l = np.square((shapes - self._shapes) / (self.kwargs['error_estimate'] * self._errors))
        indx = ~np.any(chi2_l != chi2_l, axis=1)
        chi2 = np.sum(chi2_l[indx], axis=0)
        dof = sum(indx)
        chi2_sum = np.sum(self.weights * chi2) * 1. / dof / np.sum(self.weights)

        if logger:
            if sum(indx) != len(indx):
                logger.info('Warning! We are using {0} stars out of {1} stars at step {2} of fit'.format(sum(indx), len(indx), self._n_iter))
            logger.debug('chi2 array:')
            logger.debug('chi2 summed: {0}'.format(chi2_sum))
            logger.debug('chi2 in each shape: {0}'.format(chi2))
            logger.debug('chi2 with weights: {0}'.format(self.weights * chi2))
            logger.debug('sum of weights times dof: {0}'.format(dof * np.sum(self.weights)))
            logger.debug('chi2 for each star:\n{0}'.format(chi2_l))

        if full:
            return chi2_sum, dof, chi2, indx, chi2_l
        else:
            return chi2_sum

    # TODO: stars, logger, option for doing all, fix, step_sizes, keys
    # would be nice to make this a static function
    def gradient(self, stars,
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
                                  logger=logger)
                    params[key] = params[key] + term * step_size

                    # update psf
                    self._update_psf_params(logger=logger, **params)
                    chi2_sum, dof, chi2, indx, chi2_l = self.chi2(self._fit_stars, full=True, logger=logger)

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
        # recall that chi2_sum = np.sum(self.weights * np.sum(chi2_l[indx], axis=0)) * 1. / sum(indx) / np.sum(self.weights)
        gradients = np.sum(self.weights[None] * np.nansum(gradients_l, axis=1), axis=1) * 1. / len(self._fit_stars) / np.sum(self.weights)

        return gradients, gradients_l, stencils, fx, fy

    def _grad_func_minuit(self,
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

        gradients, gradients_l, stencils, fx, fy = self.gradient(
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
            self._logger)

        if self._logger:
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
            self._logger.info(''.join(log))

        return gradients

    def _fit_func(self,
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

        logger = self._logger
        # update psf
        self._update_psf_params(r0, g1, g2,
                                z04d, z04x, z04y,
                                z05d, z05x, z05y,
                                z06d, z06x, z06y,
                                z07d, z07x, z07y,
                                z08d, z08x, z08y,
                                z09d, z09x, z09y,
                                z10d, z10x, z10y,
                                z11d, z11x, z11y,
                                logger=logger,
                                )

        chi2_sum, dof, chi2, indx, chi2_l = self.chi2(self._fit_stars, full=True, logger=logger)

        if logger:
            log = ['\n',
                    '*******************************************************************************\n',
                    '* time \t|\t {0:.3e} \t|\t ncalls  \t|\t {1:04d} \t      *\n'.format(time() - self._time, self._n_iter),
                    '*******************************************************************************\n',
                    '*  \t|\t d \t\t|\t x \t\t|\t y \t      *\n',
                    '*******************************************************************************\n',
                    '* size \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(r0, g1, g2),
                    '* z4   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z04d, z04x, z04y),
                    '* z5   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z05d, z05x, z05y),
                    '* z6   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z06d, z06x, z06y),
                    '* z7   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z07d, z07x, z07y),
                    '* z8   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z08d, z08x, z08y),
                    '* z9   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z09d, z09x, z09y),
                    '* z10  \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z10d, z10x, z10y),
                    '* z11  \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(z11d, z11x, z11y),
                    '*******************************************************************************\n',
                    '* chi2 \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e}   *\n'.format(*(chi2 / dof)),
                    '*******************************************************************************',
                    ]
            if self._n_iter % 50 == 0:
                logger.warning(''.join(log))
            else:
                logger.info(''.join(log))
        self._n_iter += 1
        return chi2_sum

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
            logger.debug("Wrote the PSF model to extension %s",extname + '_model')

        self.interp.write(fits, extname + '_interp', logger)
        if logger:
            logger.debug("Wrote the PSF interp to extension %s",extname + '_interp')

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
        self._update_psf_params(logger=logger, **self.fitter_kwargs)

        self._time = time()

        # load model and interp
        self.model = Model.read(fits, extname + '_model', logger)
        self.interp = Interp.read(fits, extname + '_interp', logger)
