"""Useful things for testing plot routines. Probably will delete before this goes into PR
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

import piff
import galsim
import copy

def return_config(expid='synthetic', psf_dir='/nfs/slac/g/ki/ki19/des/cpd/piff_y1_testbed', name='optatmo', **kwargs):
    config = {
        'verbose': 2,
        'psf': {
            'type': 'OptAtmo',
            'optpsf': {
                'type': 'OpticalWavefront',
                'knn_file_name': '/nfs/slac/g/ki/ki18/cpd/Projects/DES/Piff/tests/input/Science-20121120s1-v20i2.fits',
                'knn_extname': 1,
                'template': 'des',
                'engine': 'galsim_fast',
                'error_estimate': 1,
                'n_fit_stars': 300,
                'guess_start': True,
                'fitter_kwargs': {
                    'fix_r0': False,
                    'fix_g1': False,
                    'fix_g2': False,
                    'fix_z04d': False,
                    'fix_z05d': False,
                    'fix_z06d': False,
                    'fix_z07d': False,
                    'fix_z08d': False,
                    'fix_z09d': True,
                    'fix_z10d': True,
                    'fix_z11d': True,
                    'fix_z04x': True,
                    'fix_z05x': False,
                    'fix_z06x': False,
                    'fix_z07x': False,
                    'fix_z08x': False,
                    'fix_z09x': True,
                    'fix_z10x': True,
                    'fix_z11x': True,
                    'fix_z04y': True,
                    'fix_z05y': False,
                    'fix_z06y': False,
                    'fix_z07y': False,
                    'fix_z08y': False,
                    'fix_z09y': True,
                    'fix_z10y': True,
                    'fix_z11y': True,
                },
            },
            'atmopsf': {
                'type': 'Simple',
                'model': {
                    'type': 'Kolmogorov',
                    'include_pixel': True,
                    'fastfit': True,
                },
                'interp': {
                    'type': 'Polynomial',
                    'order': 0,
                },
            },
        },
    }

    config['output'] = {
        'dir': '{0}/{1}/'.format(psf_dir, expid),
        'file_name': 'DECam_{0}_{1}.piff'.format(expid, name),
        'stats': [
            {
                'type': 'ShapeHistograms',
                'file_name': 'DECam_{0}_{1}_shapes.png'.format(expid, name),
                'bins_size': 100,
                'bins_shape': 100,
            },
            {
                'type': 'Rho',
                'file_name': 'DECam_{0}_{1}_rho.png'.format(expid, name),
                'min_sep': 0.5,
                'max_sep': 300,
                'sep_units': 'arcmin',
                'bin_size': 0.5,
            },
            {
                'type': 'TwoDHist',
                'file_name': 'DECam_{0}_{1}_twodhist_10.png'.format(expid, name),
                'number_bins_u': 10,
                'number_bins_v': 10,
            },
            {
                'type': 'TwoDHist',
                'file_name': 'DECam_{0}_{1}_twodhist_30.png'.format(expid, name),
                'number_bins_u': 30,
                'number_bins_v': 30,
            },
            {
                'type': 'TwoDHist',
                'file_name': 'DECam_{0}_{1}_twodhist_50.png'.format(expid, name),
                'number_bins_u': 50,
                'number_bins_v': 50,
            },
        ]
    }

    config['input'] = {}

    config['input']['image_file_name'] = '{0}/{1}/DECam_{1}_??.fits.fz'.format(psf_dir, expid)
    config['input']['cat_file_name'] = {'type': 'Eval', 'simage_file_name': '@input.image_file_name'}
    config['input']['cat_file_name']['str'] = "image_file_name.replace('.fits.fz', '_findstars.fits')"
    config['image_hdu'] = 1
    config['badpix_hdu'] = 2
    config['weight_hdu'] = 3
    config['input']['cat_hdu'] = 1
    config['input']['x_col'] = 'x'
    config['input']['y_col'] = 'y'
    config['input']['sky_col'] = 'sky'
    config['input']['use_col'] = 'star_flag'
    config['input']['ra'] = 'TELRA'
    config['input']['dec'] = 'TELDEC'
    config['input']['stamp_size'] = 31

    #####
    # other input config options
    #####

    # number of stars per ccd; keep small for now
    config['input']['nstars'] = 200

    # minimum snr
    config['input']['min_snr'] = 20


    config['input'].update(kwargs)

    # modify psf to cut bad ccds
    if expid != 'synthetic':
        config['psf']['optpsf']['cut_ccds'] = [2, 31, 61]

    return config

def load_image(config, logger=None):
    logger = galsim.config.LoggerWrapper(logger)

    config = copy.deepcopy(config)
    # Import extra modules if requested
    if 'modules' in config:
        galsim.config.ImportModules(config)

    # read in the input images
    stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)

    return stars, wcs, pointing

def create_optatmo_psf(psf_dict, draw_mode='optatmo', logger=None):

    # create fake psf
    if 'type' in psf_dict:
        psf_dict.pop('type')

    # psf_clean has temporary atmosphere enabled!
    psf = piff.OptAtmoPSF(**piff.OptAtmoPSF.parseKwargs(psf_dict, logger))

    if draw_mode == 'opt':
        # also set the drawstar functions in psf to the optpsf ones
        psf.drawStar = psf.optpsf.drawStar
        psf.drawStarList = psf.optpsf.drawStarList
    elif draw_mode == 'atmo':
        psf.drawStar = psf.atmopsf.drawStar
        psf.drawStarList = psf.atmopsf.drawStarList

    return psf

def create_optatmo_data(fit_dict, n_samples=100, flux=1e4,
                draw_mode='optatmo',
                misalignment={'z04d': -0.5, 'r0': 0.1, 'g1': 0.02, 'g2': -0.05},
                coeffs=[np.array([[0.65]]), np.array([[-0.01]]), np.array([[-0.06]])],
                logger=None):
    """create psf and also clean psf"""

    psf_dict = copy.deepcopy(fit_dict['psf'])
    # fix terms which are not in misalignment and viceversa
    for kwarg in psf_dict['optpsf']['fitter_kwargs']:
        key = kwarg.split('fix_')[1]
        psf_dict['optpsf']['fitter_kwargs'][kwarg] = key not in misalignment

    psf = create_optatmo_psf(psf_dict, draw_mode, logger)
    psf_clean = create_optatmo_psf(psf_dict, draw_mode, logger)

    # set psf params
    psf.optpsf._update_psf_params(logger=logger, **misalignment)

    # if we have temporary atmosphere keys in misalignment, keep it in
    if 'r0' not in misalignment.keys():
        # otherwise disable atmosphere
        psf.optpsf.disable_atmosphere(logger=logger)
        psf_clean.optpsf.disable_atmosphere(logger=logger)

    # set atmopsf
    psf.atmopsf.interp.coeffs = coeffs
    # now deal with the stars
    decaminfo = piff.des.DECamInfo()
    # generate star coordinates
    np_rng = np.random.RandomState(1234)
    # only fit small number of chips
    chipnums = np_rng.randint(1, 63, n_samples)
    star_list = []
    wcs = {}
    pointing = None
    for chipnum in chipnums:
        # OK let us try makeTarget instead
        wcs_i = decaminfo.get_nominal_wcs(chipnum)
        wcs[chipnum] = wcs_i
        icen = np_rng.randint(100, 2048) * 1.
        jcen = np_rng.randint(100, 4096) * 1.
        star = piff.Star.makeTarget(x=icen, y=jcen, wcs=wcs_i, pix_scale=0.27,
                                    stamp_size=fit_dict['input']['stamp_size'],
                                    flux=flux, properties={'chipnum': chipnum})

        star_list.append(star)
    # get the focal positions
    star_list = decaminfo.pixel_to_focalList(star_list)

    star_list = psf.drawStarList(star_list)

    # add noise
    for star in star_list:
        star.weight.fill(1. / flux)
        gn = galsim.GaussianNoise(sigma=np.sqrt(flux), rng=None)
        star.image.addNoise(gn)

    # we also need to clean the star_list of fit params
    star_list_clean = [piff.Star(star.data, None) for star in star_list]

    # we also need to init our stars
    psf.optpsf._fit_init(star_list_clean, wcs, pointing, logger=logger)
    psf_clean.optpsf._fit_init(star_list_clean, wcs, pointing, logger=logger)

    return star_list, star_list_clean, wcs, pointing, psf, psf_clean

# plot field
def plot_2dhist(psf, stars=None, **kwargs):

    if stars:
        actual_shapes = psf.optpsf._measure_shapes(stars)
        errors = psf.optpsf._measure_shape_errors(stars)
    else:
        stars = psf.optpsf._fit_stars
        actual_shapes = psf.optpsf._shapes
        errors = psf.optpsf._errors
    star_models = psf.drawStarList(stars)
    model_shapes = psf.optpsf._measure_shapes(star_models)

    kwargs_vvals = {'vmin_e0': None, 'vmax_e0': None,
                    'vmax_e': None,
                    'vmax_de0': None, 'vmax_de': None,
                    'vmax_err': None, 'vmin_err': None,}
    for key in kwargs_vvals:
        if key in kwargs:
            kwargs_vvals[key] = kwargs.pop(key)
    # check for kwargs that are "None" and figure out from the measured shapes how to assign parameters
    if kwargs_vvals['vmin_e0'] == None:
        kwargs_vvals['vmin_e0'] = np.percentile([actual_shapes[:,0], model_shapes[:,0]],
                                                q=10)
    if kwargs_vvals['vmax_e0'] == None:
        kwargs_vvals['vmax_e0'] = np.percentile([actual_shapes[:,0], model_shapes[:,0]],
                                                q=90)
    if kwargs_vvals['vmax_e'] == None:
        kwargs_vvals['vmax_e'] = np.percentile(np.abs([actual_shapes[:,1:], model_shapes[:,1:]]),
                                                q=90)
    if kwargs_vvals['vmax_de0'] == None:
        kwargs_vvals['vmax_de0'] = np.percentile(np.abs([actual_shapes[:,0] - model_shapes[:,0]]),
                                                q=90)
    if kwargs_vvals['vmax_de'] == None:
        kwargs_vvals['vmax_de'] = np.percentile(np.abs([actual_shapes[:,1:] - model_shapes[:,1:]]),
                                                q=90)
    if kwargs_vvals['vmin_err'] == None:
        kwargs_vvals['vmin_err'] = np.percentile([errors[:, 0], errors[:, 1], errors[:, 2]],
                                                q=10)
    if kwargs_vvals['vmax_err'] == None:
        kwargs_vvals['vmax_err'] = np.percentile([errors[:, 0], errors[:, 1], errors[:, 2]],
                                                q=90)
    vvals = [[[kwargs_vvals['vmin_e0'], kwargs_vvals['vmax_e0']],
              [-kwargs_vvals['vmax_e'], kwargs_vvals['vmax_e']],
              [-kwargs_vvals['vmax_e'], kwargs_vvals['vmax_e']],  # star
             ],
             [[kwargs_vvals['vmin_e0'], kwargs_vvals['vmax_e0']],
              [-kwargs_vvals['vmax_e'], kwargs_vvals['vmax_e']],
              [-kwargs_vvals['vmax_e'], kwargs_vvals['vmax_e']],  # model
             ],
             [[-kwargs_vvals['vmax_de0'], kwargs_vvals['vmax_de0']],
              [-kwargs_vvals['vmax_de'], kwargs_vvals['vmax_de']],
              [-kwargs_vvals['vmax_de'], kwargs_vvals['vmax_de']],  # d
             ],
             [[kwargs_vvals['vmin_err'], kwargs_vvals['vmax_err']],
              [kwargs_vvals['vmin_err'], kwargs_vvals['vmax_err']],
              [kwargs_vvals['vmin_err'], kwargs_vvals['vmax_err']],  # err
             ],
             [[-5, 5], [-5, 5], [-5, 5]],  # chi2
             ]

    # coordinates
    u = []
    v = []
    for star in stars:
        u.append(star.data.properties['u'])
        v.append(star.data.properties['v'])
    u = np.array(u)
    v = np.array(v)

    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(4 * 3, 3 * 5))

    # left column gets the Y coordinate label
    axs[0, 0].set_ylabel('v')
    axs[1, 0].set_ylabel('v')
    axs[2, 0].set_ylabel('v')
    axs[3, 0].set_ylabel('v')
    axs[4, 0].set_ylabel('v')

    # bottom row gets the X coordinate label
    axs[-1, 0].set_xlabel('u')
    axs[-1, 1].set_xlabel('u')
    axs[-1, 2].set_xlabel('u')

    row_labels = [['', ' Data'], ['', ' Model'], ['d', ''], ['error ', ' Data'], ['chi2 ', '']]
    col_labels = ['e0', 'e1', 'e2']
    shapes = [actual_shapes, model_shapes, actual_shapes - model_shapes, errors, (actual_shapes - model_shapes) / errors]
    for ax_i in range(len(axs)):
        for ax_j in range(len(axs[0])):
            ax = axs[ax_i, ax_j]
            label = row_labels[ax_i][0] + col_labels[ax_j] + row_labels[ax_i][1]
            im = ax.hexbin(u, v, C=shapes[ax_i][:, ax_j],
                           vmin=vvals[ax_i][ax_j][0], vmax=vvals[ax_i][ax_j][1],
                           cmap=plt.cm.RdBu_r, **kwargs)
            fig.colorbar(im, ax=ax)
            ax.set_xlim(min(u), max(u))
            ax.set_ylim(min(v), max(v))
            ax.set_title(label)

    plt.tight_layout()

    return fig, axs, actual_shapes, errors, model_shapes

# plot 1d hist
def plot_1dhist(psf, stars, **kwargs):

    actual_shapes = psf.optpsf._measure_shapes(stars)
    errors = psf.optpsf._measure_shape_errors(stars)
    star_models = psf.drawStarList(stars)
    model_shapes = psf.optpsf._measure_shapes(star_models)

    # coordinates
    u = []
    v = []
    for star in stars:
        u.append(star.data.properties['u'])
        v.append(star.data.properties['v'])
    u = np.array(u)
    v = np.array(v)

    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(4 * 3, 3 * 5))

    row_labels = [['', ' Data'], ['', ' Model'], ['d', ''], ['error ', ' Data'], ['chi2 ', '']]
    col_labels = ['e0', 'e1', 'e2']
    shapes = [actual_shapes, model_shapes, actual_shapes - model_shapes, errors, (actual_shapes - model_shapes) / errors]
    for ax_i in range(len(axs)):
        for ax_j in range(len(axs[0])):
            ax = axs[ax_i, ax_j]
            label = row_labels[ax_i][0] + col_labels[ax_j] + row_labels[ax_i][1]
            ax.hist(shapes[ax_i][:, ax_j], histtype='step', **kwargs)
            ax.set_title(label)
            ax.set_xlabel(col_labels[ax_j])

    plt.tight_layout()

    return fig, axs

# widget to plot star and associated psf
def plot_stars(psf, stars, indx=0):
    # load star
    star = stars[indx].copy()

    # draw new star
    model_star = psf.drawStar(star)

    s_image = star.image.array
    s_image /= s_image.sum()
    m_image = model_star.image.array
    m_image /= m_image.sum()

    # plot star and drawn star
    fig, axs = plt.subplots(ncols=3, figsize=(16,3))

    ax = axs[0]
    ax.set_title('Star')
    im = ax.pcolor(s_image)
    fig.colorbar(im, ax=ax)
    ax = axs[1]
    ax.set_title('Model')
    im = ax.pcolor(model_star.image.array)
    fig.colorbar(im, ax=ax)
    ax = axs[2]
    ax.set_title('Star - Model')
    im = ax.pcolor(s_image - m_image)
    fig.colorbar(im, ax=ax)

    plt.tight_layout()

    # measure shapes and errors
    shapes = psf.optpsf._measure_shapes([star, model_star])
    errors = psf.optpsf._measure_shape_errors([star])

    # print shapes and errors. Also u, v
    string = ''
    string += 'u\t\tv: \n'
    string += '{0:+.02e}\t{1:+.02e}'.format(star.data.properties['u'], star.data.properties['v'])
    string += '\n\n'
    string += 'e0_star\t\te0_model\te0_err\t\ts-m\t\tchi: \n'
    string += '{0:+.02e}\t{1:+.02e}\t{2:+.02e}\t{3:+.02e}\t{4:+.02e}'.format(
        shapes[0,0], shapes[1,0], errors[0,0], shapes[0,0] - shapes[1,0],
        (shapes[0,0] - shapes[1,0]) / errors[0,0])
    string += '\n\n'
    string += 'e1_star\t\te1_model\te1_err\t\ts-m\t\tchi: \n'
    string += '{0:+.02e}\t{1:+.02e}\t{2:+.02e}\t{3:+.02e}\t{4:+.02e}'.format(
        shapes[0,1], shapes[1,1], errors[0,1], shapes[0,1] - shapes[1,1],
        (shapes[0,1] - shapes[1,1]) / errors[0,1])
    string += '\n\n'
    string += 'e2_star\t\te2_model\te2_err\t\ts-m\t\tchi: \n'
    string += '{0:+.02e}\t{1:+.02e}\t{2:+.02e}\t{3:+.02e}\t{4:+.02e}'.format(
        shapes[0,2], shapes[1,2], errors[0,2], shapes[0,2] - shapes[1,2],
        (shapes[0,2] - shapes[1,2]) / errors[0,2])
    string += '\n\n'
    string += 'norm_e1_star\tnorm_e1_model\ts-m: \n'
    string += '{0:+.02e}\t{1:+.02e}\t{2:+.02e}'.format(
        shapes[0,1] / shapes[0,0], shapes[1,1] / shapes[1,0],
        shapes[0,1] / shapes[0,0] - shapes[1,1] / shapes[1,0])
    string += '\n\n'
    string += 'norm_e2_star\tnorm_e2_model\ts-m: \n'
    string += '{0:+.02e}\t{1:+.02e}\t{2:+.02e}'.format(
        shapes[0,2] / shapes[0,0], shapes[1,2] / shapes[1,0],
        shapes[0,2] / shapes[0,0] - shapes[1,2] / shapes[1,0])

    print(string)

def guess_r0_from_e0(e0):
    pix_size = 0.00572
    r0_const = 0.004
    r0 = ((e0 - pix_size) / r0_const) ** -0.5
    return r0
