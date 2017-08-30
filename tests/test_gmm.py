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

from __future__ import print_function
import galsim
import piff
import numpy as np
import os
import fitsio

from piff_test_helper import timer

def make_data(params, n_gaussian=5, force_model_center=True, flux=1, noise=0., pix_scale=1.,
              fpu=0., fpv=0., nside=32, nom_u0=0., nom_v0=0., rng=None):
    """Make a Star instance filled with a Gaussian Mixture Model

    :param params:      Parameters to use in GMM
    :param n_gaussian:  Number of gaussians to use
    :param force_model_center:  If True, centroid is fixed at origin [default: True]
    :param flux:        The flux of the star
    :param noise:       RMS Gaussian noise to be added to each pixel [default: 0]
    :param pix_scale:   pixel size in "wcs" units [default: 1.]
    :param fpu,fpv:     position of this cutout in some larger focal plane [default: 0,0]
    :param nside:       The size of the array [default: 32]
    :param nom_u0, nom_v0:  The nominal u0,v0 in the StarData [default: 0,0]
    :param rng:         If adding noise, the galsim deviate to use for the random numbers
                        [default: None]
    """
    if noise == 0.:
        var = 0.1
    else:
        var = noise
    star = piff.Star.makeTarget(x=nside/2+nom_u0/pix_scale, y=nside/2+nom_v0/pix_scale,
                                u=fpu, v=fpv, scale=pix_scale, stamp_size=nside)
    star.image.setOrigin(0,0)

    gmm = piff.GaussianMixtureModel(n_gaussian=n_gaussian, force_model_center=force_model_center)

    star.fit = star.fit.newParams(params)
    star = gmm.draw(star)
    star.data.weight = star.image.copy()
    star.weight.fill(1./var/var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star

@timer
def test_single():
    """Make a one gaussian mixture model. Should just be a gaussian
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=4)
    else:
        logger = piff.config.setup_logger(verbose=1)
    # make single gaussian centered on 0
    params = np.array([.5, 0.8, 0.02, -0.02])
    star = make_data(params, n_gaussian=1, force_model_center=True)
    model = piff.GaussianMixtureModel(n_gaussian=1, force_model_center=True)
    # fit it
    star_model = model.fit(star, logger=logger)
    # check params
    np.testing.assert_allclose(star_model.fit.params, params, atol=1e-4)

    # make single gaussian with offset center
    params = np.array([2.5, 0.5, -0.3, 0.8, 0.02, -0.02])
    star = make_data(params, n_gaussian=1, force_model_center=False)
    model = piff.GaussianMixtureModel(n_gaussian=1, force_model_center=False)
    # fit it
    star_model = model.fit(star, logger=logger)
    # check params
    np.testing.assert_allclose(star_model.fit.params, params, atol=1e-4)

    # test getProfile
    prof_model = model.getProfile(star.fit)
    weight, mu_u, mu_v, sigma, g1, g2 = params
    prof = galsim.Gaussian(sigma=1.0).dilate(sigma).shear(g1=g1, g2=g2).shift(mu_u, mu_v) * weight
    assert prof==prof_model,"getProfile produces incorrect profile"

    # test draw
    star_drawn = model.draw(star_model)
    np.testing.assert_allclose(star.image.array, star_drawn.image.array, atol=1e-4)

@timer
def test_complex():
    """Same as single, only put in a complicated model
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = piff.config.setup_logger(verbose=1)
    np.random.seed(123)
    n_gaussian = 2

    weights = [1., 2]
    sigmas = [1.2, 0.8]
    g1s = [0.2, 0]
    g2s = [0, 0.2]
    dus = [5., 0]
    dvs = [0, -5]

    model = piff.GaussianMixtureModel(n_gaussian=n_gaussian, force_model_center=False, maxiter=10000, tol=1e-5)
    params = []
    for indx in range(n_gaussian):
        params += [weights[indx], dus[indx], dvs[indx], sigmas[indx], g1s[indx], g2s[indx]]
    params = np.array(params)
    # order params by weight
    params = model._reorder_params(params, logger=logger)

    logger.debug('Input GMM of star. Values for each gaussian are:')
    logger.debug('N, weight, mean_u, mean_v, sigma, g1, g2')
    for indx in range(n_gaussian):
        logger.debug('{0}, {1:.2e}, {2:+.2e}, {3:+.2e}, {4:.2e}, {5:+.2e}, {6:+.2e}'.format(indx, *params[model._niter * indx: model._niter * (indx + 1)]))

    star = make_data(params, n_gaussian=n_gaussian, force_model_center=False)
    # fit it
    star_model = model.fit(star, logger=logger)
    # check params
    np.testing.assert_allclose(star_model.fit.params, params, atol=1e-4)

    # test getProfile
    prof_model = model.getProfile(star.fit)
    for indx in range(n_gaussian):
        weight, mu_u, mu_v, sigma, g1, g2 = params[model._niter * indx: model._niter * (indx + 1)]
        prof_i = galsim.Gaussian(sigma=1.0).dilate(sigma).shear(g1=g1, g2=g2).shift(mu_u, mu_v) * weight
        if indx == 0:
            prof = prof_i
        else:
            prof = prof + prof_i
    assert prof==prof_model,"getProfile produces incorrect profile"

    # test draw
    star_drawn = model.draw(star_model)
    np.testing.assert_allclose(star.image.array, star_drawn.image.array, atol=1e-4)

@timer
def test_disk():
    """Test saving and loading of the GMM
    """

    for maxiter, tol in zip([10, 1000], [1e-3, 1e-5]):
        for force_model_center in [True, False]:
            # make model
            if force_model_center:
                n_gaussian = 6
            else:
                n_gaussian = 4
            for order_by in ['weight', 'size']:
                model = piff.GaussianMixtureModel(n_gaussian=n_gaussian, force_model_center=force_model_center, order_by=order_by, maxiter=maxiter, tol=tol)
                # Also need to test ability to serialize
                outfile = os.path.join('output', 'gmm_disk_test.fits')
                with fitsio.FITS(outfile, 'rw', clobber=True) as f:
                    model.write(f, 'psf_model')
                with fitsio.FITS(outfile, 'r') as f:
                    roundtrip_model = piff.GaussianMixtureModel.read(f, 'psf_model')
                assert model.__dict__ == roundtrip_model.__dict__

                assert model.kwargs['force_model_center'] == force_model_center
                assert model.kwargs['maxiter'] == maxiter
                assert model.kwargs['tol'] == tol
                assert model.kwargs['n_gaussian'] == n_gaussian
                assert model.kwargs['order_by'] == order_by
                assert model._ngauss == n_gaussian
                assert model._niter == [6, 4][force_model_center]
                assert model._nparams == n_gaussian * [6, 4][force_model_center]

                assert roundtrip_model.kwargs['force_model_center'] == force_model_center
                assert roundtrip_model.kwargs['maxiter'] == maxiter
                assert roundtrip_model.kwargs['tol'] == tol
                assert roundtrip_model.kwargs['n_gaussian'] == n_gaussian
                assert roundtrip_model.kwargs['order_by'] == order_by
                assert roundtrip_model._ngauss == n_gaussian
                assert roundtrip_model._niter == [6, 4][force_model_center]
                assert roundtrip_model._nparams == n_gaussian * [6, 4][force_model_center]

@timer
def test_reorder():
    """Test reordering by weight and by sigma works
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=4)
    else:
        logger = piff.config.setup_logger(verbose=1)

    # make params: have it be 4 x 6 (so that we can use the same params for testing reordering with force_model_center on and off
    np.random.seed(123)
    params = np.random.random(4 * 6)
    for force_model_center in [True, False]:
        # make model
        if force_model_center:
            n_gaussian = 6
            niter = 4
        else:
            n_gaussian = 4
            niter = 6
        model = piff.GaussianMixtureModel(n_gaussian=n_gaussian, force_model_center=force_model_center)
        for order_by in ['weight', 'size', None]:
            if order_by == None:
                order_by = model.kwargs['order_by']

            params_reordered = model._reorder_params(params, order_by=order_by, logger=logger)

            # assert that the difference in the weight or size key is indeed always growing
            if order_by == 'weight':
                order_indx = 0
            elif order_by == 'size':
                if force_model_center:
                    order_indx = 1
                else:
                    order_indx = 3

            values = params_reordered[order_indx::niter]
            diff = np.diff(values)
            np.testing.assert_array_equal(diff > 0, np.ones(len(diff), dtype=bool))


@timer
def test_convert_shapes_covars():
    """test that we get the conversion to sigma, g1, g2 correct from covar
    """
    # covar vlues
    covar = [[1, 0.5], [0.5, 1.5]]
    covars = np.array([covar])

    # shape values
    covar = covars[0]
    size = covar[0,0] + covar[1,1]
    e1 = (covar[0,0] - covar[1,1]) / size
    e2 = 2 * covar[0,1] / size
    sigma = np.sqrt(np.sqrt(covar[0,0] * covar[1,1] - covar[0,1] * covar[1,0]))
    shear = galsim.Shear(e1=e1, e2=e2)
    g1 = shear.getG1()
    g2 = shear.getG2()
    shapes = np.array([[sigma, g1, g2]])

    # test that converts work
    shapes_test = piff.GaussianMixtureModel._convert_covars_to_shapes(covars)
    covars_test = piff.GaussianMixtureModel._convert_shapes_to_covars(shapes)

    np.testing.assert_almost_equal(shapes, shapes_test)
    np.testing.assert_almost_equal(covars, covars_test)

if __name__ == '__main__':
    test_single()
    test_complex()
    test_reorder()
    test_disk()
    test_convert_shapes_covars()
