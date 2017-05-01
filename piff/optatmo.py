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
.. module:: decam_wavefront_psf
"""

from __future__ import print_function

import numpy as np

import galsim

from .psf import PSF

from .gsobject_model import Gaussian
from .optical_model import Optical
from .star import Star, StarFit, StarData
# from .util import write_kwargs, read_kwargs, make_dtype, adjust_value

from .des.decam_wavefront import DECamWavefront
from .des.decaminfo import DECamInfo

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
            logger.debug("Starting OpticalWavefrontPSF fit")
        self.optpsf.fit(self.stars, wcs, pointing, logger=logger)

        # update stars from outlier rejection
        # TODO: mark stars probably changes this
        if self.optpsf.kwargs['n_fit_stars']:
            nremoved = self.optpsf.kwargs['n_fit_stars'] - len(self.optpsf._fit_stars)
        else:
            nremoved = len(self._fit_stars) - len(self.optpsf._fit_stars)
        if nremoved > 0:
            logger.warning("Removed {0} stars in OpticalWavefrontPSF fit".format(nremoved))
        else:
            logger.debug("Removed no stars in OpticalWavefrontPSF fit")

        # disable r0,g1,g2 from OpticalWavefrontPSF, since AtmoPSF deals with those bits.
        if logger:
            logger.debug("Disabling atmosphere in OpticalWavefrontPSF")
        self.optpsf._update_psf_params(r0=None, g1=None, g2=None)
        self.optpsf.fitter_kwargs['fix_r0'] = True
        self.optpsf.fitter_kwargs['fix_g1'] = True
        self.optpsf.fitter_kwargs['fix_g2'] = True
        # double plus sure we disable it
        self.optpsf.model.kolmogorov_kwargs = {}
        self.optpsf.model.g1 = None
        self.optpsf.model.g2 = None

        # extract profiles for AtmoPSF
        if logger:
            logger.debug("Extracting OpticalWavefrontPSF profiles")
        profiles = [self.optpsf.getProfile(star) for star in self.stars]

        # fit AtmoPSF
        if logger:
            logger.debug("Fitting AtmospherePSF")
        self.atmopsf.fit(self.stars, wcs, pointing, profiles=profiles, logger=logger)

        # update stars from outlier rejection
        nremoved = len(self.stars) - len(self.atmopsf.stars)
        if nremoved > 0:
            logger.warning("Removed {0} stars in AtmospherePSF fit".format(nremoved))
        else:
            logger.debug("Removed no stars in AtmospherePSF fit")

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

    def drawStar(self, star):
        """Generate PSF image for a given star.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.

        :returns:           Star instance with its image filled with rendered PSF
        """
        import galsim
        params = []
        profs = []
        for psf_i in self.psfs:
            profs.append(psf_i.getProfile(star))
            params.append(psf_i.getParams(star))
        params = np.hstack(params)

        # draw star
        prof = galsim.Convolve(profs)
        image = star.image.copy()
        prof.drawImage(image, method='auto', offset=(star.image_pos-image.trueCenter()))
        data = StarData(image, star.image_pos, star.weight, star.data.pointing)
        return Star(data, StarFit(params))

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


class OpticalWavefrontPSF(PSF):
    """A PSF class that uses the wavefront to model the PSF

    We need to fit the following variables:
        Constant optical sigma or kolmogorov, g1, g2 in model
        misalignments in interpolant: these are the interp.misalignment terms
    """
    def __init__(self, knn_file_name, knn_extname, max_iterations=300, n_fit_stars=0, error_estimate=0.001, pupil_plane_im=None,  extra_interp_properties=None, weights=np.array([0.5, 1, 1]), fitter_kwargs={}, interp_kwargs={}, model_kwargs={}, engine='galsim_fast', template='des', fitter_algorithm='minuit', logger=None):
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
        :param fitter_kwargs:               kwargs to pass to fitter
        :param fitter_algorithm:            fitter to use for measuring wavefront. Default is minuit but also can use lmfit
        """

        self.interp_kwargs = {'n_neighbors': 15, 'algorithm': 'auto'}
        self.interp_kwargs.update(interp_kwargs)

        # it turns out this part can also be slow!
        if logger:
            logger.debug("Making interp")
        self.interp = DECamWavefront(knn_file_name, knn_extname, logger=logger, **self.interp_kwargs)
        if logger:
            logger.debug("Making model_comparer Gaussian")
        self.model_comparer = Gaussian(fastfit=True, force_model_center=True, include_pixel=True, logger=logger)

        if logger:
            logger.debug("Making DECamInfo")
        self.decaminfo = DECamInfo()

        self.weights = np.array(weights)
        # normalize weights
        self.weights /= self.weights.sum()

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
            }

        # load up the model after kwargs are set
        if logger:
            logger.debug("Loading optical engine")
        self._engines = ['galsim', 'galsim_fast']
        self._model(template=template, engine=engine, **model_kwargs)


        # put in the variable names and initial values
        # TODO: This should be called from function
        self.fitter_kwargs = {
            'throw_nan': False,
            'pedantic': True,
            'print_level': 0,
            'errordef': 0.5,  # guesstimated

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
        # update with kwargs
        self.fitter_kwargs.update(fitter_kwargs)

        # initialize the _misalignment_fix array to False so we can set initial values
        self._misalignment_fix = np.array([[False] * 3] * (11 - 4 + 1))
        self._update_psf_params(**self.fitter_kwargs)

        # now set the _misalignment_fix array to the fixed terms based on the fitter_kwargs
        self._misalignment_fix = np.array([[self.fitter_kwargs['fix_z{0:02d}{1}'.format(zi, dxy)]
                                            for dxy in ['d', 'x', 'y']]
                                            for zi in range(4, 12)])

        self._time = time()

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

    def _measure_shapes(self, stars, logger=None):
        """Work around the gsobject to measure shapes. Returns array of shapes
        """
        stars_out = []
        # TODO: It would be nice if I could copy the stars so that the new list is not the same as the old...
        for star_i, star in enumerate(stars):
            if logger:
                logger.debug("Measuring shape of star {0}".format(star_i))
                logger.debug(star.data.properties)
            star.fit.params = None
            # # we also need to pop the hsm parameter because we are using the same stars, so it is getting confused with wrong parameters -- I htink especially flux (since input stars have fluxes like 100000 and our psfs have flux like 1)
            # if 'hsm' in star.data.properties:
            #     _ = star.data.properties.pop('hsm')
            try:
                star_out = self.model_comparer.fit(star, logger=logger)
                if logger:
                    logger.debug(star_out.fit.params)
                stars_out.append(star_out)
            except:
                if logger:
                    logger.debug("Star failed moment parameter; setting shapes to nan!")
                # put in a faux params
                star.fit.params = np.array([np.nan, np.nan, np.nan])
                stars_out.append(star)
                # # not supposed to be rejecting stars...
                # star_out = self.model_comparer.fit(star, logger=logger)
                # if logger:
                #     logger.debug(star_out.fit.params)
                # stars_out.append(star_out)

        shapes = np.array([star.fit.params for star in stars_out])
        # we want sigma^2, not sigma for size
        shapes[:, 0] = np.square(shapes[:, 0])
        # normalize ellipticity
        shapes[:, 1:] = shapes[:,0][:, None] * shapes[:, 1:]

        return shapes

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
        if logger:
            logger.warning("Start fitting OptAtmoPSF using %s stars", len(stars))
        self.wcs = wcs
        self.pointing = pointing
        self.stars = stars

        # TODO: deal with convolve profiles in practice and using _fit_profiles
        convolve_profiles = len(profiles) and getattr(self.model, "getProfile", False)

        self._fit_stars = stars
        if self.kwargs['n_fit_stars'] and self.kwargs['n_fit_stars'] < len(self._fit_stars):
            choice = np.random.choice(len(self._fit_stars), self.kwargs['n_fit_stars'])
            self._fit_stars = [stars[i] for i in choice]
            if convolve_profiles:
                self._fit_profiles = [profiles[i] for i in choice]
            if logger:
                logger.warning('Cutting from {0} to {1} stars'.format(len(self.stars), len(self._fit_stars)))
        # get the moments of the stars for comparison
        if logger:
            logger.warning("Start measuring the shapes")
        self._shapes = self._measure_shapes(self._fit_stars, logger=logger)
        # cut more stars if they fail shape measurement
        indx = ~np.any(self._shapes != self._shapes, axis=1)
        if logger:
            logger.warning("Cutting {0} stars out of {1}".format(sum(~indx), len(indx)))
        self._shapes = self._shapes[indx]
        self._fit_stars = [star for star, ind in zip(self._fit_stars, indx) if ind]
        if convolve_profiles:
            self._fit_profiles = [profile for profile, ind in zip(self._fit_profiles, indx) if ind]

        # based on fitter, do _minuit_fit or (TODO) _lmfit_fit
        if self.kwargs['fitter_algorithm'] == 'minuit':
            self._minuit_fit(self._fit_stars,
                             logger=logger)
        else:
            raise NotImplementedError('fitter {0} not yet implmented!'.format(self.kwargs['fitter_algorithm']))

    def _minuit_fit(self, stars, logger=None):
        """Fit interpolated PSF model to star data using iminuit implementation of Minuit

        :parma stars:           Stars used in fit
        :param logger:          A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.info("Start fitting Optical fit using minuit and %s stars", len(stars))
        from iminuit import Minuit

        self._n_iter = 0

        if logger:
            logger.debug("Creating minuit object")
            self._logger = logger
        else:
            self._logger = None
        self._minuit = Minuit(self._fit_func, **self.fitter_kwargs)
        # run the fit and solve! This will update the interior parameters
        if logger:
            logger.warning("Running migrad for {0} steps!".format(self.kwargs['max_iterations']))
        self._minuit.migrad(ncall=self.kwargs['max_iterations'])
        # self._hesse = self._minuit.hesse()
        # self._minos = self._minuit.minos()
        # these are the best fit parameters
        self._fitarg = self._minuit.fitarg

        # update params to best values
        if logger:
            logger.debug("Minuit Fitargs:\n*****\n")
            for key in self._fitarg:
                logger.debug("{0}: {1}\n".format(key, self._fitarg[key]))
            logger.debug("Minuit Values:\n*****\n")
            for key in self._minuit.values:
                logger.debug("{0}: {1}\n".format(key, self._minuit.values[key]))
        self._update_psf_params(**self._minuit.values)

        # save params and errors to the kwargs
        self.fitter_kwargs.update(self._fitarg)
        if logger:
            logger.debug("Minuit kwargs are now:\n*****\n")
            for key in self.fitter_kwargs:
                logger.debug("{0}: {1}\n".format(key, self.fitter_kwargs[key]))

    def _lmfit_fit(self, stars, logger=None):
        """Fit interpolated PSF model to star data using lmfit implementation of Levenberg-Marquardt minimization

        :parma stars:           Stars used in fit
        :param logger:          A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.debug("Start fitting Optical fit using lmfit and %s stars", len(stars))
        import lmfit
        # results = lmfit.minimize(resid_func, params, args)
        raise NotImplementedError("Finish me!")

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
        if not self.fitter_kwargs['fix_r0'] and r0 == r0:
            self.model.kolmogorov_kwargs['r0'] = r0
        if not self.fitter_kwargs['fix_g1'] and g1 == g1:
            self.model.g1 = g1
        if not self.fitter_kwargs['fix_g2'] and g2 == g2:
            self.model.g2 = g2
        # update the misalignment
        misalignment_arr = np.array([
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
        misalignment = np.where(self._misalignment_fix, old_misalignment, misalignment_arr)
        misalignment = np.where(misalignment != misalignment, old_misalignment, misalignment)
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
            try:
                old_misalignment_print = np.vstack((
                    np.array([[old_r0, old_g1, old_g2]]),
                    old_misalignment))
                logger.debug('New - Old misalignment is \n{0}'.format(misalignment_print - old_misalignment_print))
                logger.debug('New misalignment is \n{0}'.format(misalignment_print))
            except:
                # old misalignment could be None on first iteration
                logger.debug('New misalignment is \n{0}'.format(misalignment_print))

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

        # get shapes

        shapes = self._measure_shapes(self.drawStarList(self._fit_stars),)# logger=logger)

        # calculate chisq
        # chi2 = np.sum(np.square((shapes - self._shapes) / self.kwargs['error_estimate']), axis=0)
        # dof = shapes.size
        chi2_l = np.square((shapes - self._shapes) / self.kwargs['error_estimate'])
        indx = ~np.any(chi2_l != chi2_l, axis=1)
        chi2 = np.sum(chi2_l[indx], axis=0)
        dof = sum(indx)
        if logger:
            log = ['\n',
                    '*******************************************************************************\n',
                    '* time \t|\t {0:.3e} \t|\t ncalls  \t|\t {1} \n'.format(time() - self._time, self._n_iter),
                    '*******************************************************************************\n',
                    '*  \t|\t d \t\t|\t x \t\t|\t y \t\t*\n',
                    '*******************************************************************************\n',
                    '* size \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(r0, g1, g2),
                    '* z4   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z04d, z04x, z04y),
                    '* z5   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z05d, z05x, z05y),
                    '* z6   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z06d, z06x, z06y),
                    '* z7   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z07d, z07x, z07y),
                    '* z8   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z08d, z08x, z08y),
                    '* z9   \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z09d, z09x, z09y),
                    '* z10  \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z10d, z10x, z10y),
                    '* z11  \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(z11d, z11x, z11y),
                    '*******************************************************************************\n',
                    '* chi2 \t|\t {0:+.3e} \t|\t {1:+.3e} \t|\t {2:+.3e} *\n'.format(*(chi2 / dof)),
                    '*******************************************************************************',
                    ]
            if self._n_iter % 50 == 0:
                logger.warning(''.join(log))
            else:
                logger.debug(''.join(log))
        self._n_iter += 1
        chi2_sum = np.sum(self.weights * chi2) * 1. / dof / np.sum(self.weights)
        if logger:
            if sum(indx) != len(indx):
                logger.debug('Warning! We are using {0} stars out of {1} stars'.format(sum(indx), len(indx)))
            logger.debug('chi2 array:')
            logger.debug(chi2_sum)
            logger.debug(chi2)
            logger.debug(self.weights * chi2)
            logger.debug(dof * np.sum(self.weights))
            logger.debug(chi2_l)
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
