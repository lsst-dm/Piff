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
import numpy as np
import piff
import os
import yaml
import subprocess

import galsim
from piff_test_helper import get_script_name, timer

from time import time

def create_fit_dict():
    # code for returning everything we need for the fits

    piff_dict = {'verbose': 1}

    # input
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # TODO: update this! It is OK if we only use a single CCD for testing.
    piff_dict['input'] = {
        'dir': dir_path + '/input',
        'image_file_name': "DECam_00241238_??.fits.fz",
        'cat_file_name': {
            'type': 'Eval',
            'str': "image_file_name.replace('.fits.fz', '_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits')",
            'simage_file_name': '@input.image_file_name',
            },

        # What hdu is everything in?
        'image_hdu': 1,
        'badpix_hdu': 2,
        'weight_hdu': 3,
        'cat_hdu': 2,

        # What columns in the catalog have things we need?
        'x_col': 'XWIN_IMAGE',
        'y_col': 'YWIN_IMAGE',
        'ra': 'TELRA',
        'dec': 'TELDEC',
        'gain': 'GAINA',
        'sky_col': 'BACKGROUND',

        # How large should the postage stamp cutouts of the stars be?
        'stamp_size': 31,
        }

    # psf.
    piff_dict['psf'] = {

        'type': 'OptAtmo',

        'optpsf': {
            'type': 'OpticalWavefront',

            'knn_file_name': dir_path + '/input/Science-20121120s1-v20i2.fits',
            'knn_extname': 1,
            'max_iterations': 100,
            'n_fit_stars': 50,
            'error_estimate': 0.001,
            'engine': 'galsim_fast',
            'template': 'des',
            'fitter_algorithm': 'minuit',
            # by default only fit for constant size + shape and z04d
            'fitter_kwargs': {
                'fix_r0': False,
                'fix_g1': False,
                'fix_g2': False,
                'fix_z04d': False,
                'fix_z05d': True,
                'fix_z06d': True,
                'fix_z07d': True,
                'fix_z08d': True,
                'fix_z09d': True,
                'fix_z10d': True,
                'fix_z11d': True,
                'fix_z04x': True,
                'fix_z05x': True,
                'fix_z06x': True,
                'fix_z07x': True,
                'fix_z08x': True,
                'fix_z09x': True,
                'fix_z10x': True,
                'fix_z11x': True,
                'fix_z04y': True,
                'fix_z05y': True,
                'fix_z06y': True,
                'fix_z07y': True,
                'fix_z08y': True,
                'fix_z09y': True,
                'fix_z10y': True,
                'fix_z11y': True,
                }

            },

        'atmopsf': {
            'type': 'Simple',

            'model': {
                'type': 'Gaussian',
                'include_pixel': False,
                'fastfit': True,
                },

            'interp': {
                'type': 'Polynomial',
                'order': 0,
                },

            },
        }

    # output
    piff_dict['output'] = {

        'dir': 'output',
        'file_name': 'test_optatmo.piff',

        'stats': [
                {
                    'type': 'ShapeHistograms',
                    'file_name': 'test_optatmo_shapes.png',
                    'bins_size': 100,
                    'bins_shape': 100,
                },
                {
                    'type': 'Rho',
                    'file_name': 'test_optatmo_rho.png',
                    'min_sep': 0.5,
                    'max_sep': 300,
                    'sep_units': 'arcmin',
                    'bin_size': 0.5,
                },
                {
                    'type': 'TwoDHist',
                    'file_name': 'test_optatmo_twdhist.png',
                    'number_bins_u': 20,
                    'number_bins_v': 20,
                },
            ]
        }

    return piff_dict

def create_synthetic_optatmo(fit_dict, misalignment={'z04d': -0.5, 'r0': 0.1, 'g1': 0.02, 'g2': -0.05}, n_samples=10000, logger=None):
    """creates synthetic catalog with a misalignment and a constant ellipticity"""

    decaminfo = piff.des.DECamInfo()

    # generate star coordinates
    np_rng = np.random.RandomState(1234)
    # only fit small number of chips
    chipnums = np_rng.randint(1, 63, n_samples)

    star_list = []
    wcs_field = {}
    for chipnum in chipnums:
        # OK let us try makeTarget instead
        # wcs = galsim.AffineTransform(0, -0.26, -0.26, 0, world_origin=galsim.PositionD(0,0))
        wcs = decaminfo.get_nominal_wcs(chipnum)
        wcs_field[chipnum] = wcs
        icen = np_rng.randint(100, 2048) * 1.
        jcen = np_rng.randint(100, 4096) * 1.
        star = piff.Star.makeTarget(x=icen, y=jcen, wcs=wcs, stamp_size=64, properties={'chipnum': chipnum})

        star_list.append(star)

    # get the focal positions
    star_list = decaminfo.pixel_to_focalList(star_list)

    # create fake psf
    fit_dict = fit_dict.copy() # don't modify original fit_dict
    psf_dict = fit_dict['psf']
    psf_dict.pop('type')

    # fix terms which are not in misalignment and viceversa
    for kwarg in psf_dict['optpsf']['fitter_kwargs']:
        key = kwarg.split('fix_')[1]
        psf_dict['optpsf']['fitter_kwargs'][kwarg] = key not in misalignment

    import copy
    psf_dict_copy = copy.deepcopy(psf_dict)
    psf = piff.OptAtmoPSF(**piff.OptAtmoPSF.parseKwargs(psf_dict, logger))

    # set psf params

    # use separate optical psf to draw the stars
    optpsf = piff.OptAtmoPSF(**piff.OptAtmoPSF.parseKwargs(psf_dict_copy, logger)).optpsf
    # I think I can skip the atmpsf params? those would be fitted later...
    optpsf._update_psf_params(logger=logger, **misalignment)
    # draw the stars using only optpsf
    star_list = optpsf.drawStarList(star_list)
    # we also need to clean the star_list of fit params
    star_list_clean = [piff.Star(star.data, None) for star in star_list]

    pointing = None
    return star_list, star_list_clean, psf, wcs_field, pointing

@timer
def test_yaml():
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None
    print('test that fitting procedure works')
    # make dictionary
    fit_dict = create_fit_dict()
    psf_file = fit_dict['output']['file_name']

    # execute in piff
    piff.piffify(fit_dict, logger)
    psf = piff.read(psf_file)

    import ipdb; ipdb.set_trace()

    # tests

    # execute with piffify

    # read final file

@timer
def test_fit():
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = None

    print('test reading in yaml and executing')

    # make dictionary
    fit_dict = create_fit_dict()
    # test that we can generate stars
    misalignment = {'z04d': -0.5, 'r0': 0.1, 'g1': 0.02, 'g2': -0.05}
    stars, stars_clean, psf, wcs, pointing = create_synthetic_optatmo(fit_dict, misalignment=misalignment, n_samples=50, logger=logger)

    # run fit
    psf.fit(stars_clean, wcs, pointing, logger=logger)

    # check parameters

    import ipdb; ipdb.set_trace()

@timer
def test_optical_fit():
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=3)
    else:
        logger = None

    print('test reading in yaml and executing')

    # make dictionary
    fit_dict = create_fit_dict()
    # test that we can generate stars
    misalignment = {'z04d': -0.5, 'z05d': 0.1}#, 'r0': 0.1, 'g1': 0.02, 'g2': -0.05}
    stars, stars_clean, psf, wcs, pointing = create_synthetic_optatmo(fit_dict, misalignment=misalignment, n_samples=1000, logger=logger)
    optpsf = psf.optpsf

    # run fit
    optpsf.fit(stars_clean, wcs, pointing, logger=logger)

    # check parameters of fit
    fitargs = optpsf._fitarg
    # construct results
    misalignment_fit = {key: fitargs[key] for key in misalignment}
    misalignment_fit_errors = {key: fitargs['error_{0}'.format(key)] for key in misalignment}
    # one measure of success is if true value is within 1 sigma of each error
    # print(misalignment.keys())
    # print(misalignment)
    # print(misalignment_fit)
    # print(misalignment_fit_errors)
    for key in misalignment:
        assert abs(misalignment[key] - misalignment_fit[key]) < 2 * misalignment_fit_errors[key], '{0} Failed Optical Fit. Parameter: {1}, Fit: {2} +- {3}'.format(key, misalignment[key], misalignment_fit[key], misalignment_fit_errors[key])
    # TODO: what about if i fit a lot of params, and also what if the errors are messed up?

    indx = np.random.randint(len(stars))
    star_raw = stars[indx]
    star_clean = stars_clean[indx]
    star_drawn = optpsf.drawStar(star_clean)
    # related: make sure the params match
    np.testing.assert_allclose(star_raw.fit.params, star_drawn.fit.params, rtol=0, atol=1e-4)

    # finally, also make sure the shapes match
    shape_raw = optpsf._measure_shapes([star_raw, star_drawn, star_clean])
    np.testing.assert_allclose(shape_raw[0], shape_raw[1], rtol=0, atol=1e-3)
    np.testing.assert_allclose(shape_raw[0], shape_raw[2], rtol=0, atol=1e-3)


    # TEMPORARY: write code to make plots of u, v and make sure that works

@timer
def test_disk():
    # test saving and loading
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    # make dictionary
    fit_dict = create_fit_dict()
    # test that we can generate stars
    misalignment = {'z04d': -0.5, 'r0': 0.1, 'g1': 0.02, 'g2': -0.05}
    # only take one ncall
    fit_dict['psf']['optpsf']['max_iterations'] = 1
    stars, stars_clean, psf, wcs, pointing = create_synthetic_optatmo(fit_dict, misalignment=misalignment, n_samples=2, logger=logger)
    optpsf = psf.optpsf
    optpsf.fit(stars_clean, wcs, pointing, logger=logger)

    optpsf_file = os.path.join('output', 'opticalwavefront.piff')
    optpsf.write(optpsf_file)
    optpsf_disk = piff.read(optpsf_file)

    # fitter_kwargs
    for key in optpsf_disk.fitter_kwargs:
        assert optpsf_disk.fitter_kwargs[key] == optpsf.fitter_kwargs[key], "Failture on saving and loading optical wavefront psf key {0}".format(key)


    # interp.misalignment
    np.testing.assert_array_equal(optpsf.interp.misalignment, optpsf_disk.interp.misalignment, "DECam interp misalignment fails saving and loading")

    # model.kolmogorov_kwargs
    for key in optpsf_disk.model.kolmogorov_kwargs:
        assert optpsf_disk.model.kolmogorov_kwargs[key] == optpsf.model.kolmogorov_kwargs[key], "Failture on saving and loading optical wavefront optical key {0}".format(key)

    # model.g1, model.g2
    assert optpsf_disk.model.g1 == optpsf.model.g1, "Failture on saving and loading optical wavefront optical g1"
    assert optpsf_disk.model.g2 == optpsf.model.g2, "Failture on saving and loading optical wavefront optical g2"

    # check fit params
    # TODO: there is something messed up in the saving and loading of stars
    star = optpsf.stars[0]
    # first check decaminfo
    properties = optpsf.decaminfo.pixel_to_focal(star.copy()).data.properties
    properties_disk = optpsf_disk.decaminfo.pixel_to_focal(star.copy()).data.properties
    for key in ['focal_x', 'focal_y']:
        assert properties[key] == properties_disk[key], "{0} not close in saving and loading optical wavefront".format(key)

    # now check interpolation
    params = optpsf.interp.interpolate(star.copy()).fit.params
    params_disk = optpsf_disk.interp.interpolate(star.copy()).fit.params
    np.testing.assert_array_equal(params, params_disk, "Interpolation failed for saving and loading optical wavefront")

    # repeat task for List
    properties = optpsf.decaminfo.pixel_to_focalList([star.copy()])[0].data.properties
    properties_disk = optpsf_disk.decaminfo.pixel_to_focalList([star.copy()])[0].data.properties
    for key in ['focal_x', 'focal_y']:
        assert properties[key] == properties_disk[key], "{0} not close in saving and loading optical wavefront".format(key)
    params = optpsf.interp.interpolateList([star.copy()])[0].fit.params
    params_disk = optpsf_disk.interp.interpolateList([star.copy()])[0].fit.params
    np.testing.assert_array_equal(params, params_disk, "Interpolation failed for saving and loading optical wavefront")

    # and for getParams
    params = optpsf.getParams(star.copy())
    params_disk = optpsf_disk.getParams(star.copy())
    np.testing.assert_array_equal(params, params_disk, "Interpolation failed for saving and loading optical wavefront")

    # check getProfile
    profile = optpsf.getProfile(star.copy())
    profile_disk = optpsf_disk.getProfile(star.copy())
    assert profile == profile_disk, "failure matching profiles when saving and loading"

    # check shapes
    star_drawn = optpsf.drawStar(star.copy())
    # use the same star but different psf
    star_iodrawn = optpsf_disk.drawStar(star.copy())
    # compare shape
    shape = optpsf._measure_shapes([star])[0]
    shape_drawn = optpsf._measure_shapes([star_drawn])[0]
    ioshape = optpsf_disk._measure_shapes([star])[0]
    ioshape_drawn = optpsf_disk._measure_shapes([star_drawn])[0]
    shape_iodrawn = optpsf._measure_shapes([star_iodrawn])[0]
    ioshape_drawn = optpsf_disk._measure_shapes([star_drawn])[0]
    ioshape_iodrawn = optpsf_disk._measure_shapes([star_iodrawn])[0]
    for s in [shape, ioshape, shape_drawn, ioshape_drawn, shape_iodrawn, ioshape_drawn, ioshape_iodrawn]:
        print(s)
    np.testing.assert_array_equal(shape_drawn, ioshape_drawn, "Failure on comparing the measured shape before and after IO of optical psf")

    # compare array
    assert star_drawn.image == star_iodrawn.image, "Failure of drawn images"

    star_disk = optpsf_disk.stars[0].copy()
    star_disk_drawn = optpsf.drawStar(star_disk.copy())
    star_disk_iodrawn = optpsf.drawStar(star_disk.copy())
    shape_disk_drawn = optpsf._measure_shapes([star_disk_drawn])[0]
    shape_disk_iodrawn = optpsf._measure_shapes([star_disk_iodrawn])[0]

    for s in [shape_disk_drawn, shape_disk_iodrawn]:
        print(s)
    import ipdb; ipdb.set_trace()
    # TODO: requires making psf images
    # repeat for psf that has been fitted

    # save

    # load

    # check stuff

    # make sure the optpsf portion has turned off the atmosphere

if __name__ == '__main__':
    test_disk()
    test_optical_fit()
    # test_fit()
    # test_yaml()
