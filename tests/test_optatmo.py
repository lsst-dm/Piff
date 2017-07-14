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

    piff_dict = {'verbose': 2}

    # input
    dir_path = os.path.dirname(os.path.realpath(__file__))
    piff_dict['input'] = {
        'dir': dir_path + '/y1_test',
        'images': "DECam_00241238_%02d.fits.fz",
        'cats': "DECam_00241238_%02d_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits",
        'chipnums': "[ c for c in range(1,63) if c is not 61 and c is not 31 ]",

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

            'knn_file_name': dir_path + '/wavefront_test/Science-20121120s1-v20i2_noextname.fits',
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
            'type': 'SingleChip',

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
        wcs = decaminfo.get_nominal_wcs(chipnum)
        wcs_field[chipnum] = wcs
        icen = np_rng.randint(100, 2048) * 1.
        jcen = np_rng.randint(100, 4096) * 1.
        image_pos = galsim.PositionD(icen, jcen)
        image = galsim.Image(64,64, wcs=wcs)
        image.setCenter(33, 33)

        stardata = piff.StarData(image, image_pos, properties={'chipnum': chipnum})
        star = piff.Star(stardata, None)

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
    stars, stars_clean, psf, wcs, pointing = create_synthetic_optatmo(fit_dict, misalignment=misalignment, n_samples=100, logger=logger)
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


if __name__ == '__main__':
    test_optical_fit()
    test_fit()
    test_yaml()
