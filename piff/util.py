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
.. module:: util
"""

from __future__ import print_function
import numpy as np
import os
import galsim

# Courtesy of
# http://stackoverflow.com/questions/3862310/how-can-i-find-all-subclasses-of-a-given-class-in-python
def get_all_subclasses(cls):
    """Get all subclasses of an existing class.
    """
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses

def ensure_dir(target):
    """Ensure that the directory for a target output file exists.

    :param target:      The file that you want to write to.
    """
    d = os.path.dirname(target)
    if d != '' and  not os.path.exists(d):
        os.makedirs(d)

def make_dtype(key, value):
    """A helper function that makes a dtype appropriate for a given value

    :param key:     The key to use for the column name in the dtype.
    :param value:   The input value (just one item if using a column of multiple values)

    :returns: a numpy.dtype instance
    """
    def make_dt_tuple(key, t, size):
        # If size == 0, then it's not an array, so return a 2 element tuple.
        # Otherwise, the size is the third item in the tuple.
        if size == 0:
            return (key, t)
        else:
            return (key, t, size)

    try:
        # Note: this works for either arrays or strings
        size = len(value)
        t = type(value[0])
    except TypeError:
        size = 0
        t = type(value)
    dt = np.dtype(t) # just used to categorize the type into int, float, str

    if dt.kind in np.typecodes['AllInteger']:
        t = int
    elif dt.kind in np.typecodes['AllFloat']:
        t = float
    elif dt.kind in ['S','U'] and not isinstance(value, str):
        # catch lists of strings
        t = np.array(value).dtype.str
        t = t.replace('U','S')
    elif dt.kind in ['S','U']:
        t = bytes
    else:
        # Other objects should be manually serialized by the initializer or the finish_read and
        # finish_write functions.
        raise ValueError("Cannot serialize object of type %s"%t)
    dt = make_dt_tuple(key, t, size)

    return dt

def adjust_value(value, dtype):
    """Possibly adjust a value to match the type expected for the given dtype.

    e.g. change np.int16 -> int if dtype expects int.  Or vice versa.

    :param value:   The input value to possible adjust.

    :returns: the adjusted value to use for writing in a FITS table.
    """
    t = dtype[1]
    if len(dtype) == 2 or dtype[2] == 0:
        # dtype is either (key, t) or (key, t, size)
        # if no size or size == 0, then just use t as the type.
        return t(value)
    elif t == bytes:
        # Strings may need to be encoded.
        try:
            return value.encode()
        except AttributeError:
            return value
    else:
        try:
            # Arrays of strings may need to be encoded.
            return np.array([v.encode() for v in value])
        except AttributeError:
            # For other numpy arrays, we can use astype instead.
            return np.array(value).astype(t)

def write_kwargs(fits, extname, kwargs):
    """A helper function for writing a single row table into a fits file with the values
    and column names given by a kwargs dict.

    :param fits:        An open fitsio.FITS instance
    :param extname:     The extension to write to
    :param kwargs:      A kwargs dict to be written as a FITS binary table.
    """
    cols = []
    dtypes = []
    for key, value in kwargs.items():
        # Don't add values that are None to the table.
        if value is None:
            continue
        dt = make_dtype(key, value)
        value = adjust_value(value,dt)
        cols.append([value])
        dtypes.append(dt)
    data = np.array(list(zip(*cols)), dtype=dtypes)
    fits.write_table(data, extname=extname)

def read_kwargs(fits, extname):
    """A helper function for reading a single row table from a fits file returning the values
    and column names as a kwargs dict.

    :param fits:        An open fitsio.FITS instance
    :param extname:     The extension to read.

    :returns: A dict of the kwargs that were read from the file.
    """
    cols = fits[extname].get_colnames()
    data = fits[extname].read()
    assert len(data) == 1
    kwargs = dict([ (col, data[col][0]) for col in cols ])
    for key,value in kwargs.items():
        # Convert any byte strings to a regular str
        try:
            value = str(value.decode())
            kwargs[key] = value

        except AttributeError:
            # Also convert arrays of bytes into arrays of strings.
            try:
                value = np.array([str(v.decode()) for v in value])
                kwargs[key] = value
            except (AttributeError, TypeError):
                pass
    return kwargs

def hsm(star, logger=None):
    """ Use HSM to measure moments of star image.
    """
    image, weight, image_pos = star.data.getImage()
    # note that findadaptivemom only respects the weight function in a binary sense.  i.e., pixels
    # with non-zero weight will be included in the moment measurement, those with weight=0.0 will be
    # excluded.
    mom = image.FindAdaptiveMom(weight=weight, strict=False)

    sigma = mom.moments_sigma
    shape = mom.observed_shape
    # These are in pixel coordinates.  Need to convert to world coords.
    jac = image.wcs.jacobian(image_pos=image_pos)
    scale, shear, theta, flip = jac.getDecomposition()
    # Fix sigma
    sigma *= scale
    # Fix shear.  First the flip, if any.
    if flip:
        shape = galsim.Shear(g1 = -shape.g1, g2 = shape.g2)
    # Next the rotation
    shape = galsim.Shear(g = shape.g, beta = shape.beta + theta)
    # Finally the shear
    shape = shear + shape

    flux = mom.moments_amp

    localwcs = image.wcs.local(image_pos)
    center = localwcs.toWorld(mom.moments_centroid) - localwcs.toWorld(image_pos)
    flag = mom.moments_status

    if logger:
        logger.debug('Star hsm. Values of flux, u0, v0, size, g1, g2, flag are are:')
        logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e} {6:.2e}'.format(flux, center.x, center.y, sigma, shape.g1, shape.g2, flag))

    return flux, center.x, center.y, sigma, shape.g1, shape.g2, flag

def hsm_error(star, logger=None):
    """ Use python implementation of HSM to measure higher order moments of star image to get errors.

    Slow since it's python, not C, but we should only have to do this once per star.

    calculate the error on our e0,e1,e2

    $ e_0= \sum \left[ (x-x_0)^2 + (y-y_0)^2  \right] K(x,y) I(x,y) $

    where K(x,y) is the HSM kernel, I(x,y) is the image, s(x,y) is the shot noise per pixel
    so

    $ \sigma^2(e_0) = \sum \left\{ \left[ (x-x_0)^2 + (y-y_0)^2  \right] K(x,y) \right\}^2 s^2(x,y) $


    TODO: might be a factor of 2 missing still?
    TODO: what do the _v subscripts indicate? can I cut that and keep clarity?
    """
    import piff

    star = star.copy()
    # TODO: does it matter if include_pixel is True vs False?
    hsm_model = piff.Gaussian(fastfit=True, force_model_center=False, include_pixel=True)

    # need to calculate moments w/ HSM first
    star.fit.params = None   # need to clear fit params for the HSM algorithm
    star_fit = hsm_model.fit(star)  # fit to this star

    # get the model image (ie. the kernel)
    star_model = hsm_model.draw(star_fit)

    # get parameters
    u0, v0, e0_sqrt, e1_norm, e2_norm = star_fit.fit.params
    e0 = e0_sqrt ** 2
    e1 = e1_norm * e0
    e2 = e2_norm * e0

    flux_hsm, u0_hsm, v0_hsm, size_hsm, g1_hsm, g2_hsm, flag_hsm = hsm(star)


    # get vectors for data, weight and u, v
    data_v, weight_v, u_v, v_v = star.data.getDataVector(include_zero_weight=False)
    # also get the values for the HSM kernel, which is just the fitted hsm model
    kernel_v, wk_v, uk_v, vk_v = star_model.data.getDataVector(include_zero_weight=False)

    # if weight=0 don't use pixel, otherwise set this = 1.
    wuse_v = np.where(weight_v>0.0,1.0,0.0)

    # normalization for the various sums over pixels
    normalization = np.sum(data_v * wuse_v * kernel_v)
    normalization2 = normalization * normalization

    # use flux and calculate its error from star_model - noise-free Gaussian matched to the data
    flux_calc = star_model.flux

    star_model_wgt = star_model.weight.array
    star_model_sdata2 = 1. / star_model_wgt
    sigma_flux = np.sqrt(np.sum(star_model_sdata2))

    # calculate number of effective pixels
    npixeff_calc = np.power(np.sum(wuse_v * kernel_v), 2) / np.sum(np.power(wuse_v * kernel_v, 2))

    # for completeness calculate my own u0,v0
    u0_calc = np.sum(data_v * wuse_v * kernel_v * u_v) / normalization
    v0_calc = np.sum(data_v * wuse_v * kernel_v * v_v) / normalization

    # calculate moments
    # TODO: should I use u0_calc etc here?
    du_v = u_v - u0
    dv_v = v_v - v0
    Muu = np.sum(data_v * wuse_v * kernel_v * du_v * du_v) / normalization
    Mvv = np.sum(data_v * wuse_v * kernel_v * dv_v * dv_v) / normalization
    Muv = np.sum(data_v * wuse_v * kernel_v * du_v * dv_v) / normalization

    # now e0,e1,e2 (these are very close but not the same as the HSM values?)
    # also note that this defintion for e1 and e2 is /2 compared to previous definitions
    e0_calc = Muu + Mvv
    e1_calc = (Muu - Mvv) / 2.
    e2_calc = Muv

    # now calculate errors: ie. shot and read noise per pixel
    sdata_v = np.sqrt(1.0 / weight_v)

    sigma2_u0 = np.sum(np.power(sdata_v * wuse_v * kernel_v * u_v, 2)) / normalization2
    sigma2_v0 = np.sum(np.power(sdata_v * wuse_v * kernel_v * v_v, 2)) / normalization2
    sigma_u0 = np.sqrt(sigma2_u0)
    sigma_v0 = np.sqrt(sigma2_v0)

    # TODO: what is the scale of these different errors?

    # three terms: those proportional to: sdata_v, sigma_u0 and sigma_v0, and sigma_flux
    sigma2_e0 = np.sum(np.power(sdata_v * wuse_v * kernel_v * (du_v * du_v + dv_v * dv_v), 2)) / normalization2
    # TODO: should this have the /2?
    sigma2_e1 = np.sum(np.power(sdata_v * wuse_v * kernel_v * (du_v * du_v - dv_v * dv_v) / 2., 2)) / normalization2
    sigma2_e2 = np.sum(np.power(sdata_v * wuse_v * kernel_v * (du_v * dv_v), 2)) / normalization2

    # add sigma_u0, sigma_v0
    # TODO: should this have the *2?
    sigma2_e0 = sigma2_e0 + np.sum(np.power(2. * du_v * data_v * wuse_v * kernel_v * sigma_u0, 2) + np.power(2. * dv_v * data_v * wuse_v * kernel_v * sigma_v0, 2)) / normalization2
    sigma2_e1 = sigma2_e1 + np.sum(np.power(du_v * data_v * wuse_v * kernel_v * sigma_u0, 2) + np.power(dv_v * data_v * wuse_v * kernel_v * sigma_v0, 2)) / normalization2
    sigma2_e2 = sigma2_e2 + np.sum(np.power(dv_v * data_v * wuse_v * kernel_v * sigma_u0, 2) + np.power(du_v * data_v * wuse_v * kernel_v * sigma_v0, 2)) / normalization2

    # add sigma_flux
    sigma2_e0 = sigma2_e0 + np.power(e0_calc * sigma_flux / normalization, 2)
    sigma2_e1 = sigma2_e1 + np.power(e1_calc * sigma_flux / normalization, 2)
    sigma2_e2 = sigma2_e2 + np.power(e2_calc * sigma_flux / normalization, 2)

    sigma_e0 = np.sqrt(sigma2_e0)
    sigma_e1 = np.sqrt(sigma2_e1)
    sigma_e2 = np.sqrt(sigma2_e2)

    if logger:
        logger.debug('Star hsm_error. Value of flux, u0, v0, e0, e1, e2 are:')
        logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e}'.format(flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc))
        logger.debug('Star hsm_error. Value of Gaussian model flux, u0, v0, e0, e1, e2 are:')
        logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e}'.format(flux, u0, v0, e0, e1, e2))
        logger.debug('Star hsm_error. Value of regular hsm algorithm flux, u0, v0, e0, e1, e2 are:')
        logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e}'.format(flux_hsm, u0_hsm, v0_hsm, e0_hsm, e1_hsm, e2_hsm))
        logger.debug('Star hsm_error. Value of errors for flux, u0, v0, e0, e1, e2 are:')
        logger.debug('{0:.2e} {1:.2e} {2:.2e} {3:.2e} {4:.2e} {5:.2e}'.format(sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2))

    # TODO: when I feel confident about calculated values, do not return _calc
    return sigma_flux, sigma_u0, sigma_v0, sigma_e0, sigma_e1, sigma_e2, flux_calc, u0_calc, v0_calc, e0_calc, e1_calc, e2_calc
