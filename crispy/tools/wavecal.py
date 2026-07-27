from scipy.optimize import curve_fit
from photutils.centroids import centroid_com, centroid_2dg
from crispy.tools.imgtools import gen_bad_pix_mask
from scipy.interpolate import griddata
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder, find_peaks
from scipy import ndimage, interpolate
from scipy.spatial import cKDTree
import warnings
import glob
from shutil import copy2
from scipy.special import erf
from crispy.tools.reduction import calculateWaveList
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy import ndimage
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import re
import os
from crispy.tools.locate_psflets import locatePSFlets, PSFLets, transform, fine_transform
from crispy.tools.image import Image
import matplotlib as mpl
import numpy as np
from scipy import signal
from astropy.io import fits as fits
import pandas as pd

from crispy.tools.initLogger import getLogger
log = getLogger('crispy')


# from photutils import EPSFBuilder
# from astropy.nddata import NDData
# from astropy.stats import sigma_clipped_stats
# from astropy.table import Table
# from photutils import find_peaks
# from photutils.psf import extract_stars


warnings.filterwarnings("ignore")


def do_inspection(par, image, xpos, ypos, lam, display_plot=False):
    """
    Generate a plot of PSFlet positions overlaid on a calibration image.

    Args:
        par: Parameter object containing wavecalDir attribute
        image: 2D numpy array of the calibration image
        xpos: 2D numpy array of x-positions of PSFlets in image-space
        ypos: 2D numpy array of y-positions of PSFlets in image-space
        lam: Wavelength of the calibration image
        display_plot: Whether to display the plot interactively (default False)

    Saves a PNG image in the directory specified in the 'par' object showing 
    PSFlet positions as blue circles overlaid on the grayscale calibration image.
    
"""

    log.info(f'Generating inspection image for calibration image at {lam} nm')
    xg, yg = xpos.shape
    vals = np.array([(xpos[m, n], ypos[m, n])
                     for m in range(xg) for n in range(yg)])

    # Temporarily turn off interactive plotting until this function is complete
    if not display_plot:
        plt.ioff()

    fig, ax = plt.subplots(figsize=(15, 15))
    mean = np.mean(image)
    std = np.std(image)
    norm = mpl.colors.Normalize(vmin=mean, vmax=mean + 5 * std)
    ax.imshow(image, cmap='gray_r', norm=norm, interpolation='nearest', origin='lower')
    patches = [plt.Circle(val, 3) for val in vals]
    collection = PatchCollection(patches, color='blue', lw=1, alpha=0.5)
    ax.add_collection(collection)
    fig.savefig(os.path.join(par.outdir, 'inspection_%3d.png' % (lam)), dpi=600)

    if display_plot:
        plt.show(block=False)
    else:
        plt.ion()
    plt.close(fig)  # Ensure that the plot gets closed, no matter what. Since display_plot doesn't seem to work on all machines. 


def make_polychrome(lam1, lam2, hires_arrs, lam_arr, psftool, allcoef,
                    xindx, yindx, ydim, xdim, finexy=None, reflam=None, upsample=10, nlam=10,
                    prefiltered=False,
                    ):
    """
    TODO, make a numpy-style docstring. Include some details about how "the polychrome" image is made.

    prefiltered: bool
        If True, hires_arrs is assumed to have already been passed through
        ndimage.spline_filter (once, ahead of time, e.g. in buildcalibrations), so the
        in-loop spline_filter call below is skipped. Since spline_filter is linear and
        commutes with the affine interpolation between calibration wavelengths performed
        below, this is mathematically equivalent to filtering the interpolated array on
        every sub-wavelength, but avoids redundant repeated filtering of the same
        calibration arrays across sub-wavelengths and wavelength bins. Default False
        (preserves original behavior).
    """

    padding = 10
    image = np.zeros((ydim + 2 * padding, xdim + 2 * padding))
    x = np.arange(image.shape[1])   # width  -> x-axis
    y = np.arange(image.shape[0])   # height -> y-axis
    x, y = np.meshgrid(x, y)        # default 'xy' indexing -> both arrays shape (ny, nx)
    npix = hires_arrs[0].shape[2] // upsample

    dloglam = (np.log(lam2) - np.log(lam1)) / nlam
    loglam = np.log(lam1) + dloglam / 2. + np.arange(nlam) * dloglam

    for lam in np.exp(loglam):

        ################################################################
        # Build the appropriate average hires image by averaging over
        # the nearest wavelengths.  Then apply a spline filter to the
        # interpolated high resolution PSFlet images to avoid having
        # to do this later, saving a factor of a few in time.
        ################################################################

        hires = np.zeros((hires_arrs[0].shape))
        if lam <= np.amin(lam_arr):
            hires[:] = hires_arrs[0]
        elif lam >= np.amax(lam_arr):
            hires[:] = hires_arrs[-1]
        else:
            i1 = np.amax(np.arange(len(lam_arr))[np.where(lam > lam_arr)])
            i2 = i1 + 1
            hires = hires_arrs[i1] * \
                (lam - lam_arr[i1]) / (lam_arr[i2] - lam_arr[i1])
            hires += hires_arrs[i2] * \
                (lam_arr[i2] - lam) / (lam_arr[i2] - lam_arr[i1])

        if not prefiltered:
            for i in range(hires.shape[0]):
                for j in range(hires.shape[1]):
                    hires[i, j] = ndimage.spline_filter(hires[i, j])

        ################################################################
        # Run through lenslet centroids at this wavelength using the
        # fitted coefficients in psftool to get the centroids.  For
        # each centroid, compute the weights for the four nearest
        # regions on which the high-resolution PSFlets have been made.
        # Interpolate the high-resolution PSFlets and take their
        # weighted average, adding this to the image in the
        # appropriate place.
        ################################################################

#         if finexy is None:
#             xcen, ycen = psftool.return_locations(
#                 lam, allcoef, xindx, yindx)
#         else:
#             xcen, ycen = fine_transform(
#                 lam, xindx, yindx, reflam, finexy[0], finexy[1])
        xcen, ycen = psftool.return_locations(lam, allcoef, xindx, yindx)
        if finexy is not None:
            xcen += finexy[0]
            ycen += finexy[1]

        xcen += padding
        ycen += padding
        xcen = np.reshape(xcen, -1)
        ycen = np.reshape(ycen, -1)
        for i in range(xcen.shape[0]):
            if not (
                    xcen[i] > npix // 2 and xcen[i] < image.shape[1] - npix // 2 and
                    ycen[i] > npix // 2 and ycen[i] < image.shape[0] - npix // 2):
                continue

            # central pixel -> npix*upsample//2
            iy1 = int(ycen[i]) - npix // 2
            iy2 = iy1 + npix
            ix1 = int(xcen[i]) - npix // 2
            ix2 = ix1 + npix
            yinterp = (y[iy1:iy2, ix1:ix2] - ycen[i]) * \
                upsample + upsample * npix / 2
            xinterp = (x[iy1:iy2, ix1:ix2] - xcen[i]) * \
                upsample + upsample * npix / 2
            # Now find the closest high-resolution PSFs

            x_hires = xcen[i] * 1. / image.shape[1]
            y_hires = ycen[i] * 1. / image.shape[0]

            x_hires = x_hires * hires_arrs[0].shape[1] - 0.5
            y_hires = y_hires * hires_arrs[0].shape[0] - 0.5

            totweight = 0

            if x_hires <= 0:
                i1 = i2 = 0
            elif x_hires >= hires_arrs[0].shape[1] - 1:
                i1 = i2 = hires_arrs[0].shape[1] - 1
            else:
                i1 = int(x_hires)
                i2 = i1 + 1

            if y_hires < 0:
                j1 = j2 = 0
            elif y_hires >= hires_arrs[0].shape[0] - 1:
                j1 = j2 = hires_arrs[0].shape[0] - 1
            else:
                j1 = int(y_hires)
                j2 = j1 + 1

            ##############################################################
            # Bilinear interpolation by hand.  Do not extrapolate, but
            # instead use the nearest PSFlet near the edge of the
            # image.  The outer regions will therefore have slightly
            # less reliable PSFlet reconstructions.  Then take the
            # weighted average of the interpolated PSFlets.
            ##############################################################

            weight22 = max(0, (x_hires - i1) * (y_hires - j1))
            weight12 = max(0, (x_hires - i1) * (j2 - y_hires))
            weight21 = max(0, (i2 - x_hires) * (y_hires - j1))
            weight11 = max(0, (i2 - x_hires) * (j2 - y_hires))
            totweight = weight11 + weight21 + weight12 + weight22
            weight11 /= totweight * nlam
            weight12 /= totweight * nlam
            weight21 /= totweight * nlam
            weight22 /= totweight * nlam

            ##############################################################
            # map_coordinates(prefilter=False) is linear in its input array,
            # and all four calls below share identical [yinterp, xinterp]
            # coordinates, so the weighted sum of four interpolations equals
            # a single interpolation of the weighted sum of the four hires
            # regions. Combining first cuts the map_coordinates call count
            # by 4x (exactly, for nsubarr=1, since the four regions are then
            # identical).
            ##############################################################
            combined = (weight11 * hires[j1, i1] + weight12 * hires[j1, i2] +
                       weight21 * hires[j2, i1] + weight22 * hires[j2, i2])
            image[iy1:iy2, ix1:ix2] += ndimage.map_coordinates(
                combined, [yinterp, xinterp], prefilter=False)

    image = image[padding:-padding, padding:-padding]
    return image


def make_hires_polychrome(lam1, lam2, hires_arrs, lam_arr, psftool, allcoef,
                          xindx, yindx, ydim, xdim, upsample=10, nlam=10,
                          finexy=None, reflam=None, prefiltered=False):
    """
    prefiltered: bool
        If True, hires_arrs is assumed to have already been passed through
        ndimage.spline_filter (once, ahead of time), so the in-loop spline_filter call
        below is skipped. See make_polychrome's docstring for why this is exact.
        Default False (preserves original behavior).
    """

    padding = 10
    image = np.zeros((ydim + 2 * padding, xdim + 2 * padding))
    hiresimg = np.zeros((image.shape[0] * upsample, image.shape[1] * upsample))
    x = np.arange(hiresimg.shape[1])   # width  -> x-axis
    y = np.arange(hiresimg.shape[0])   # height -> y-axis
    x, y = np.meshgrid(x, y)           # default 'xy' indexing -> both arrays shape (ny, nx)
    npix = hires_arrs[0].shape[2]

    dloglam = (np.log(lam2) - np.log(lam1)) / nlam
    loglam = np.log(lam1) + dloglam / 2. + np.arange(nlam) * dloglam

    for lam in np.exp(loglam):

        ################################################################
        # Build the appropriate average hires image by averaging over
        # the nearest wavelengths.  Then apply a spline filter to the
        # interpolated high resolution PSFlet images to avoid having
        # to do this later, saving a factor of a few in time.
        ################################################################

        hires = np.zeros((hires_arrs[0].shape))
        if lam <= np.amin(lam_arr):
            hires[:] = hires_arrs[0]
        elif lam >= np.amax(lam_arr):
            hires[:] = hires_arrs[-1]
        else:
            i1 = np.amax(np.arange(len(lam_arr))[np.where(lam > lam_arr)])
            i2 = i1 + 1
            hires = hires_arrs[i1] * \
                (lam - lam_arr[i1]) / (lam_arr[i2] - lam_arr[i1])
            hires += hires_arrs[i2] * \
                (lam_arr[i2] - lam) / (lam_arr[i2] - lam_arr[i1])

        if not prefiltered:
            for i in range(hires.shape[0]):
                for j in range(hires.shape[1]):
                    hires[i, j] = ndimage.spline_filter(hires[i, j])

        ################################################################
        # Run through lenslet centroids at this wavelength using the
        # fitted coefficients in psftool to get the centroids.  For
        # each centroid, compute the weights for the four nearest
        # regions on which the high-resolution PSFlets have been made.
        # Interpolate the high-resolution PSFlets and take their
        # weighted average, adding this to the image in the
        # appropriate place.
        ################################################################

#         if finexy is None:
#             xcen, ycen = psftool.return_locations(
#                 lam, allcoef, xindx, yindx)
#         else:
#             xcen, ycen = fine_transform(
#                 lam, xindx, yindx, reflam, finexy[0], finexy[1])
        xcen, ycen = psftool.return_locations(
            lam, allcoef, xindx, yindx)
        if finexy is not None:
            xcen += finexy[0]
            ycen += finexy[1]
        xcen += padding
        ycen += padding
        xcen = np.reshape(xcen, -1)
        ycen = np.reshape(ycen, -1)
        for i in range(xcen.shape[0]):
            if not (xcen[i] > npix // (2 * upsample) and
                    xcen[i] < image.shape[1] - npix // (2 * upsample) and
                    ycen[i] > npix // (2 * upsample) and
                    ycen[i] < image.shape[0] - npix // (2 * upsample)):
                continue
            # central pixel -> npix*upsample//2
            iy1 = int(ycen[i] * upsample) - npix // 2
            iy2 = iy1 + npix
            ix1 = int(xcen[i] * upsample) - npix // 2
            ix2 = ix1 + npix
            yinterp = (y[iy1:iy2, ix1:ix2] - ycen[i] * upsample) + npix // 2
            xinterp = (x[iy1:iy2, ix1:ix2] - xcen[i] * upsample) + npix // 2

            # Now find the closest high-resolution PSFs

            x_hires = xcen[i] * 1. / image.shape[1]
            y_hires = ycen[i] * 1. / image.shape[0]

            x_hires = x_hires * hires_arrs[0].shape[1] - 0.5
            y_hires = y_hires * hires_arrs[0].shape[0] - 0.5

            totweight = 0

            if x_hires <= 0:
                i1 = i2 = 0
            elif x_hires >= hires_arrs[0].shape[1] - 1:
                i1 = i2 = hires_arrs[0].shape[1] - 1
            else:
                i1 = int(x_hires)
                i2 = i1 + 1

            if y_hires < 0:
                j1 = j2 = 0
            elif y_hires >= hires_arrs[0].shape[0] - 1:
                j1 = j2 = hires_arrs[0].shape[0] - 1
            else:
                j1 = int(y_hires)
                j2 = j1 + 1

            ##############################################################
            # Bilinear interpolation by hand.  Do not extrapolate, but
            # instead use the nearest PSFlet near the edge of the
            # image.  The outer regions will therefore have slightly
            # less reliable PSFlet reconstructions.  Then take the
            # weighted average of the interpolated PSFlets.
            ##############################################################

            weight22 = max(0, (x_hires - i1) * (y_hires - j1))
            weight12 = max(0, (x_hires - i1) * (j2 - y_hires))
            weight21 = max(0, (i2 - x_hires) * (y_hires - j1))
            weight11 = max(0, (i2 - x_hires) * (j2 - y_hires))
            totweight = weight11 + weight21 + weight12 + weight22
            weight11 /= totweight * nlam
            weight12 /= totweight * nlam
            weight21 /= totweight * nlam
            weight22 /= totweight * nlam

            ##############################################################
            # map_coordinates(prefilter=False) is linear in its input array,
            # and all four calls below share identical [yinterp, xinterp]
            # coordinates, so the weighted sum of four interpolations equals
            # a single interpolation of the weighted sum of the four hires
            # regions. Combining first cuts the map_coordinates call count
            # by 4x (exactly, for nsubarr=1, since the four regions are then
            # identical).
            ##############################################################
            combined = (weight11 * hires[j1, i1] + weight12 * hires[j1, i2] +
                       weight21 * hires[j2, i1] + weight22 * hires[j2, i2])
            hiresimg[iy1:iy2, ix1:ix2] += ndimage.map_coordinates(
                combined, [yinterp, xinterp], prefilter=False)

    hiresimg = hiresimg[padding * upsample:-padding *
                        upsample, padding * upsample:-padding * upsample]
    return hiresimg


def get_sim_hires(par, lam, upsample=10, nsubarr=1, npix=13, normalize=True):
    """
    Build high resolution images of the undersampled PSF using the
    monochromatic frames. This version of the function uses the perfect
    knowledge of the Gaussian PSFLet. Only valid if par.gaussian=True.
    All PSFLets are the same across the entire FOV

    Parameters
    ----------
    par : object
        Parameter object containing FWHM, FWHMlam, and gaussian attributes
    lam : float
        Wavelength for which to generate the high resolution PSF
    upsample : int, optional
        Upsampling factor for the high resolution array. Default is 10
    nsubarr : int, optional
        Number of subarrays in each dimension. Default is 1
    npix : int, optional
        Number of pixels in the base PSF. Default is 13
    normalize : bool, optional
        Whether to normalizealize the PSFlet. Default is True

    Returns
    -------
    hires_arr : ndarray
        4D array of shape (nsubarr, nsubarr, array_size, array_size) containing
        the high resolution PSFlets
    """
    # Determine side length of the upsampled array
    array_size = upsample * (npix + 1)  

    # Allocate memory for the array that we will fill out one slice at a time
    hires_arr = np.zeros((nsubarr, nsubarr, array_size, array_size))

    # Generate a grid of (X,Y) grid coordinates
    _x = np.arange(array_size) - array_size // 2
    _y = np.arange(array_size) - array_size // 2
    _x, _y = np.meshgrid(_x, _y)

    sigma_baseline = par.FWHM / 2.355 * upsample  # Calculate Gaussian sigma in units of pixels in the upsampled array
    sigma_scaled = sigma_baseline * lam / par.FWHMlam  # Scale this sigma by the current wavelength
    psflet = (erf((_x + 0.5) / (np.sqrt(2) * sigma_scaled)) -
            erf((_x - 0.5) / (np.sqrt(2) * sigma_scaled))) * \
        (erf((_y + 0.5) / (np.sqrt(2) * sigma_scaled)) -
     erf((_y - 0.5) / (np.sqrt(2) * sigma_scaled)))

    # Normalize the PSFLet, if desired
    if normalize:
        psflet *= upsample**2 / np.sum(psflet)

    # Because the output is expected to ahve nsubarr * nsubarr entries, fill the array with the same PSFLet
    for i in range(nsubarr):
        for j in range(nsubarr):
            hires_arr[i, j] = psflet

    return hires_arr


# def epsflets(subim,
#             upsample=5,
#             npix=13):
#     """
#     Estimates the underlying high-resolution PSFlets using Photutils tools
#     
#     Parameters
#     ----------
#     subim: 2D array
#         Array representing the subsection of the focal over which to average PSFlets,
#         assuming that they are all the same
#     upsample: int
#         Fits PSFlets and interpolates on a grid which has higher sampling than the original
#         image by a factor "upsample"
#     npix: int 
#         Number of pixels for each PSFlet model
#         
#     Returns
#     -------
#     data: 2D array
#         npix x npix array with the PSFlet model
#     """
#     data = subim.copy()
#     peaks_tbl = find_peaks(data, threshold=100.)
#     peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output
#     stars_tbl = Table()
#     stars_tbl['x'] = peaks_tbl['x_peak']
#     stars_tbl['y'] = peaks_tbl['y_peak']
#     mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.,
#                                                         maxiters=None)
#     data -= median_val
#     nddata = NDData(data=data)
#     stars = extract_stars(nddata, stars_tbl, size=npix)
#     epsf_builder = EPSFBuilder(oversampling=upsample, maxiters=3,
#                                progress_bar=False)
#     epsf, fitted_stars = epsf_builder(stars)
#     return epsf.data
#             

def gethires(x, y, good, image, upsample=5, nsubarr=5, npix=13, normalize=True):
    """
    Build high resolution images of the undersampled PSF using the
    monochromatic frames.

    Inputs:
    1.
    """

    ###################################################################
    # hires_arr has nsubarr x nsubarr high-resolution PSFlets.  Smooth
    # out the result very slightly to reduce the impact of poorly
    # sampled points.  The resolution on these images, which will be
    # passed to a multidimensional spline interpolator, is a factor of
    # upsample higher than the pixellation of the original image.
    ###################################################################

    hires_arr = np.zeros((nsubarr, nsubarr, upsample *
                          (npix + 1), upsample * (npix + 1)))
    _x = np.arange(3 * upsample) - (3 * upsample - 1) / 2.
    _x, _y = np.meshgrid(_x, _x)
    r2 = _x**2 + _y**2
    window = np.exp(-r2 / (2 * 0.3**2 * (upsample / 5.)**2))

    ###################################################################
    # yreg and xreg denote the regions of the image.  Each region will
    # have roughly 20,000/nsubarr**2 PSFlets from which to construct
    # the resampled version.  For 5x5 (default), this is roughly 800.
    ###################################################################

    for yreg in range(nsubarr):
        i1 = yreg * image.data.shape[0] // nsubarr
        i2 = i1 + image.data.shape[0] // nsubarr
        i1 = max(i1, npix)
        i2 = min(i2, image.data.shape[0] - npix)

        for xreg in range(nsubarr):
            j1 = xreg * image.data.shape[1] // nsubarr
            j2 = j1 + image.data.shape[1] // nsubarr
            j1 = max(j1, npix)
            j2 = min(j2, image.data.shape[1] - npix)

            ############################################################
            # subim holds the high-resolution images.  The first
            # dimension counts over PSFlet, and must hold roughly the
            # total number of PSFlets divided by upsample**2.  The
            # worst possible case is about 20,000/nsubarr**2.
            ############################################################

            k = 0
            subim = np.zeros((20000 / nsubarr**2, upsample *
                              (npix + 1), upsample * (npix + 1)))

            ############################################################
            # Now put the PSFlets in.  The pixel of index
            # [npix*upsample//2, npix*upsample//2] is the centroid.
            # The counter k keeps track of how many PSFlets contribute
            # to each resolution element.
            ############################################################

            for i in range(x.shape[0]):
                if x[i] > j1 and x[i] < j2 and y[i] > i1 and y[i] < i2 and good[i]:
                    xval = x[i] - 0.5 / upsample
                    yval = y[i] - 0.5 / upsample

                    ix = int((1 + int(xval) - xval) * upsample)
                    iy = int((1 + int(yval) - yval) * upsample)

                    if ix == upsample:
                        ix -= upsample
                    if iy == upsample:
                        iy -= upsample

                    iy1, ix1 = [int(yval) - npix // 2, int(xval) - npix // 2]
                    cutout = image.data[iy1:iy1 + npix + 1, ix1:ix1 + npix + 1]
#                     log.info('{:},{:},{:}'.format(k,iy,ix))
                    subim[k, iy::upsample, ix::upsample] = cutout
                    k += 1

            meanpsf = np.zeros((upsample * (npix + 1), upsample * (npix + 1)))
            weight = np.zeros((upsample * (npix + 1), upsample * (npix + 1)))

            ############################################################
            # Take the trimmed mean (middle 60% of the data) for each
            # PSFlet to avoid contamination by bad pixels.  Then
            # convolve with a narrow Gaussian to mitigate the effects
            # of poor sampling.
            ############################################################

            for ii in range(3):

                window1 = np.exp(-r2 / (2 * 1**2 * (upsample / 5.)**2))
                window2 = np.exp(-r2 / (2 * 1**2 * (upsample / 5.)**2))
                if ii < 2:
                    window = window2
                else:
                    window = window1

                if ii > 0:
                    for kk in range(k):
                        mask = 1. * (subim[kk] != 0)
                        if np.sum(mask) > 0:
                            A = np.sum(subim[kk] * meanpsf * mask)
                            A /= np.sum(meanpsf**2 * mask)

                            if A > 0.5 and A < 2:
                                subim[kk] /= A
                            else:
                                subim[kk] = 0

                            chisq = np.sum(mask * (meanpsf - subim[kk])**2)
                            chisq /= np.amax(meanpsf)**2

                            subim[kk] *= (chisq < 1e-2 * upsample**2)
                            # mask2 = np.abs(meanpsf - subim[kk])/(np.abs(meanpsf) + 0.01*np.amax(meanpsf)) < 1
                            # subim[kk] *= mask2
                            subim[kk] *= subim[kk] > -1e-3 * np.amax(meanpsf)

                subim2 = subim.copy()
                for i in range(subim.shape[1]):
                    for j in range(subim.shape[2]):

                        _i1 = max(i - upsample // 4, 0)
                        _i2 = min(i + upsample // 4 + 1, subim.shape[1] - 1)
                        _j1 = max(j - upsample // 4, 0)
                        _j2 = min(j + upsample // 4 + 1, subim.shape[2] - 1)

                        data = subim2[:k, _i1:_i2, _j1:_j2][np.where(
                            subim2[:k, _i1:_i2, _j1:_j2] != 0)]
                        if data.shape[0] > 10:
                            data = np.sort(data)[3:-3]
                            std = np.std(data) + 1e-10
                            mean = np.mean(data)

                            subim[:k,
                                  i,
                                  j] *= np.abs(subim[:k,
                                                     i,
                                                     j] - mean) / std < 3.5
                        elif data.shape[0] > 5:
                            data = np.sort(data)[1:-1]
                            std = np.std(data) + 1e-10
                            mean = np.mean(data)

                            subim[:k,
                                  i,
                                  j] *= np.abs(subim[:k,
                                                     i,
                                                     j] - mean) / std < 3.5

                        data = subim[:k, i, j][np.where(subim[:k, i, j] != 0)]
                        # data = np.sort(data)
                        npts = data.shape[0]
                        if npts > 0:
                            meanpsf[i, j] = np.mean(data)
                            weight[i, j] = npts

                meanpsf = signal.convolve2d(
                    meanpsf * weight, window, mode='same')
                meanpsf /= signal.convolve2d(weight, window, mode='same')

                val = meanpsf.copy()
                for jj in range(10):
                    tmp = val / signal.convolve2d(meanpsf, window, mode='same')
                    meanpsf *= signal.convolve2d(tmp,
                                                 window[::-1, ::-1], mode='same')

            ############################################################
            # Normalize all PSFs to unit flux when resampled with an
            # interpolator.
            ############################################################

            if normalize:
                meanpsf *= upsample**2 / np.sum(meanpsf)
            hires_arr[yreg, xreg] = meanpsf

    return hires_arr

# def gethires(x, y, good, image, upsample=5, nsubarr=5, npix=13, normalize=True):
#     """
#     Build high resolution images of the undersampled PSF using the
#     monochromatic frames.
# 
#     Inputs:
#     1.
#     """
# 
#     data = image.data
#     subim = data[:image.data.shape[0] // nsubarr,:data.shape[1] // nsubarr]
#     test = epsflets(subim,upsample,npix)
#     hires_arr = np.zeros((nsubarr, nsubarr, test.shape[0], test.shape[1]))
# 
#     for yreg in range(nsubarr):
#         i1 = yreg * data.shape[0] // nsubarr
#         i2 = i1 + data.shape[0] // nsubarr
#         i1 = max(i1, npix)
#         i2 = min(i2, data.shape[0] - npix)
#         for xreg in range(nsubarr):
#             j1 = xreg * data.shape[1] // nsubarr
#             j2 = j1 + data.shape[1] // nsubarr
#             j1 = max(j1, npix)
#             j2 = min(j2, data.shape[1] - npix)
#             subim = data[i1:i2,j1:j2]
#             hires_arr[yreg,xreg] = epsflets(subim,upsample,npix)
#             if normalize:
#                 hires_arr[yreg,xreg] *= upsample**2 / np.sum(hires_arr[yreg,xreg])
# 
#     return hires_arr


def makeHires(
        par,
        xindx,
        yindx,
        lam,
        allcoef,
        psftool,
        imlist=None,
        parallel=True,
        savehiresimages=True,
        upsample=5,
        nsubarr=5,
        npix=13,
        finexy=None,
        reflam=None):
    '''
    This function creates high-resolution models of the PSFLets
    for each wavelength in the input list. The process involves:

    1. For each wavelength:
       a. Calculate the positions of PSFLets on the detector
       b. If using real data (not Gaussian simulation):
          - Extract small image regions around each PSFLet
          - Combine these subimages to create a super-sampled PSFLet model
       c. If using Gaussian simulation:
          - Generate idealized Gaussian PSFLet models

    2. The detector is divided into nsubarr x nsubarr regions, and a separate
       high-resolution PSFLet model is created for each region to account for
       spatial variations across the detector.

    3. The resulting high-resolution PSFLet models have a spatial sampling 'upsample'
       times higher than the original detector pixels.

    4. If enabled, the function can use parallel processing to speed up the computation
       for multiple wavelengths.

    5. The high-resolution PSFLet models can optionally be saved as FITS files.

    Returns:
    hires_arrs : list of numpy.ndarray
        List of high-resolution PSFLet models for each input wavelength
    '''
    hires_arrs = []
    allxpos = []
    allypos = []
    allgood = []

    log.info('Making high-resolution PSFLet models')

    if parallel:
        log.info('Starting parallel computation')
        if not par.gaussian_hires:
            for i in range(len(lam)):
                # if finexy is None:
                #     xpos, ypos = psftool.return_locations(
                #         lam[i], allcoef, xindx, yindx)
                # else:
                #     xpos, ypos = fine_transform(
                #         lam[i], xindx, yindx, reflam, finexy[0], finexy[1])
                # if finexy is None:
                xpos, ypos = psftool.return_locations(
                    lam[i], allcoef, xindx, yindx)
                if finexy is not None:
                    xpos += finexy[0]
                    ypos += finexy[1]
                good = np.reshape(psftool.good, -1)
                xpos = np.reshape(xpos, -1)
                ypos = np.reshape(ypos, -1)
                allxpos += [xpos]
                allypos += [ypos]
                allgood += [good]
        # Each wavelength's high-res PSFLet is computed on a worker thread. The heavy lifting
        # (scipy.ndimage) releases the GIL, so threads give real speedup while sharing memory,
        # avoiding the pickling/spawn hazards of the old multiprocessing.Process pool.
        def _hires_task(i):
            if par.gaussian_hires:
                return get_sim_hires(par, lam[i], upsample, nsubarr)
            return gethires(allxpos[i], allypos[i], allgood[i], imlist[i], upsample, nsubarr, npix)

        # executor.map yields results in submission (wavelength) order, so hires_arrs stays ordered
        # and lam[i] matches each array.
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for i, high_res_array in enumerate(executor.map(_hires_task, range(len(lam)))):
                print(f'  [makeHires] Completed {i + 1} of {len(lam)} wavelengths', flush=True)
                hires_arrs += [high_res_array]

                if savehiresimages:
                    out = fits.HDUList(fits.PrimaryHDU(high_res_array.astype(np.float32)))
                    out.writeto(
                        os.path.join(par.wavecalDir, 'hires_psflets_lam%d.fits' % (lam[i])),
                        overwrite=True)
    else:
        log.info('No parallel computation')
        for i in range(len(lam)):
            if par.gaussian_hires:
                high_res_array = get_sim_hires(par, lam[i], upsample, nsubarr)
            else:
                # if finexy is None:
                #     xpos, ypos = psftool.return_locations(
                #         lam[i], allcoef, xindx, yindx)
                # else:
                #     xpos, ypos = fine_transform(
                #         lam[i], xindx, yindx, reflam, finexy[0], finexy[1])
                xpos, ypos = psftool.return_locations(
                    lam[i], allcoef, xindx, yindx)
                if finexy is not None:
                    xpos += finexy[0]
                    ypos += finexy[1]
                good = np.reshape(psftool.good, -1)
                xpos = np.reshape(xpos, -1)
                ypos = np.reshape(ypos, -1)
                high_res_array = gethires(xpos, ypos, good, imlist[i], upsample, nsubarr)
            hires_arrs += [high_res_array]

            # Validate savehiresimages parameter - should be boolean
            if not isinstance(savehiresimages, bool):
                raise ValueError(f"savehiresimages must be boolean (True/False), got {type(savehiresimages).__name__}: {savehiresimages}")

            if savehiresimages:
                # Apparently deprecated code that didn't get used? Commenting it out for now. 
                # di, dj = high_res_array.shape[0], high_res_array.shape[2]
                # outim = np.zeros((di * dj, di * dj))
                # for ii in range(di):
                #     for jj in range(di):
                #         outim[ii * dj:(ii + 1) * dj, jj * dj:(jj + 1) * dj] = high_res_array[ii, jj]
                out = fits.HDUList(fits.PrimaryHDU(high_res_array.astype(np.float32)))
                out.writeto(os.path.join(par.wavecalDir, 'hires_psflets_lam%d.fits' % (lam[i])), overwrite=True)

    return hires_arrs


def gauss(x, a, x0, sig, b):
    '''
    Simple gaussian function with usual inputs
    '''
    return b + a * np.exp(-(x - x0)**2 / (2. * sig**2))


def fit_monochromatic_cube(cube,
                           lamlist,
                           returnAll=False,
                           sigma_guess=5):
    '''
    Fits an extracted data cube with a gaussian to find the wavelength peak

    Parameters
    ----------
    cube: 3D ndarray
        The extracted datacube where all bad pixels are NaNs
    lamlist: 1D array
        List of wavelengths corresponding to the slices of the cube
        Suggested units: nanometers (in which sigma_guess is about 5)
    returnAll: boolean
        If True, return the full results of the curve fit function (popt,pcov)
        If False, return only the central wavelength (Default)
    sigma_guess: float
        Guess at the width of the gaussian fit in same units as lamlist (Default 5)
    '''
    vals = np.nansum(np.nansum(cube, axis=2), axis=1)
    popt, pcov = curve_fit(gauss,
                           lamlist,
                           vals,
                           p0=[np.amax(vals), lamlist[np.argmax(vals)], sigma_guess, 0]
                           )
    if returnAll: 
        return popt, pcov
    else: 
        return popt[1], np.sqrt(pcov)


def monochromatic_update(par, inImage, inLam, order=3, apodize=False):
    # TODO, add docstring. inImage is an Image object that contains the monochromatic image
    log.info(f"Making copies of wavelength solution from {os.path.join(par.wavecalDir, 'lamsol.dat')}")
    copy2(os.path.join(par.wavecalDir, "lamsol.dat"), os.path.join(par.wavecalDir, "lamsol_old.dat"))
    lamsol = np.loadtxt(os.path.join(par.wavecalDir, "lamsol.dat"))
    lam = lamsol[:, 0]
    allcoef = lamsol[:, 1:]
    psftool = PSFLets()
    oldcoef = psftool.monochrome_coef(inLam, lam, allcoef, order=order)

    log.info('Generating new wavelength solution')
    ysize, xsize = inImage.data.shape
    mask = np.ones((ysize, xsize))
    if apodize:
        y = np.arange(ysize)
        x = np.arange(xsize)
        x -= xsize // 2
        y -= ysize // 2
        x, y = np.meshgrid(x, y)

        r = np.sqrt(x**2 + y**2)
        mask = (r < min(ysize, xsize) // 2)

    x, y, good, newcoef = locatePSFlets(inImage, polyorder=order, mask=mask, sig=1., coef=oldcoef, phi=par.philens, scale=par.pitch / par.pixsize, nlens=par.nlens)
    psftool.geninterparray(lam, allcoef, order=order)
    dcoef = newcoef - oldcoef

    indx = np.asarray([0, 1, 4, 10, 11, 14])
    psftool.interp_arr[0][indx] += dcoef[indx]
    psftool.genpixsol(par, lam, allcoef, order=order, lam1=min(lam) / 1.01, lam2=max(lam) * 1.01)
    psftool.savepixsol(outdir=par.wavecalDir)

    #################################################################
    # Update coefficients at all wavelengths
    #################################################################
    for i in range(lamsol.shape[0]):
        lamsol[i, indx + 1] += dcoef[indx]

    #################################################################
    # Record the shift in the spot locations.
    #################################################################

    phi1 = np.mean([np.arctan2(oldcoef[4], oldcoef[1]),
                    np.arctan2(-oldcoef[11], oldcoef[14])])
    phi2 = np.mean([np.arctan2(newcoef[4], newcoef[1]),
                    np.arctan2(-newcoef[11], newcoef[14])])
    dx, dy, dphi = [dcoef[0], dcoef[10], phi2 - phi1]

    log.info('%.2f: x-shift from archival spot positions (pixels)' % dx)
    log.info('%.2f: y-shift from archival spot positions (pixels)' % dy)
    log.info(
        '%.2f: rotation from archival spot positions (degrees)' %
        (dphi * 180. / np.pi))

    log.info("Overwriting old wavecal")
    np.savetxt(os.path.join(par.wavecalDir, "lamsol.dat"), lamsol)
    log.info("Don't forget to run buildcalibrations again with makePolychrome=True!")
    return dx, dy, dphi


def evaluate_dispersion_solution_fit_quality(image_data, x_calc, y_calc,
                                             output_directory, lam=None, window_size=9,
                                             x_offset=0, y_offset=0, pixel_pitch=None):
    """
    Produce a diagnostic scatterplot of the dispersion-solution fit quality.

    Independently detects PSFlet peaks in a calibration image and measures, for each peak, the
    distance to the nearest PSFlet position predicted by the optimized polynomial solution. The
    resulting "calculated PSFlet location error" is displayed as a scatterplot colored by that
    distance, so the user can visualize how well the dispersion fit matches the real spots as a
    function of position on the sensor.

    Parameters
    ----------
    image_data: 2D ndarray
            The calibration image (possibly cropped to a fitting window) in which to detect peaks.
    x_calc, y_calc: ndarray
            Calculated PSFlet x/y positions from the optimized polynomial coefficients. These may be
            in full-frame coordinates while image_data is cropped; x_offset/y_offset reconcile the
            two frames.
    output_directory: string
            Directory in which to save the diagnostic PNG.
    lam: float (optional)
            Wavelength (nm) of the image, used only for the plot title and output filename.
    window_size: int
            Box size (pixels) passed to find_peaks for local-maximum detection. Default 9.
    x_offset, y_offset: int
            Pixel offset (the fitting-window origin) added to the measured peak coordinates so they
            share the full-frame coordinate system of x_calc/y_calc. Default 0.
    pixel_pitch: float (optional)
            Detector pixel pitch in microns. If provided, the plot and colorbar are scaled to display
            positions and distances in microns instead of pixels. Default None (use pixels).

    Notes
    -----
    Peak detection typically returns many thousands of spots by design. Nearest-neighbor matching
    is performed with a single vectorized scipy.spatial.cKDTree query rather than a per-peak loop.
    Background statistics (median and standard deviation) are computed internally using
    sigma-clipped statistics.
    """
    # Compute background statistics using sigma-clipped estimation.
    median, std = sigma_clipped_stats(image_data)[:2]

    # Detect local maxima above background, refining each to a center-of-mass centroid.
    threshold = np.max(image_data)/5 #alternatively, median + 10 * std
    log.info('evaluate_dispersion_solution_fit_quality: detecting peaks above background')
    peaks = find_peaks(image_data, threshold=threshold, box_size=window_size,
                       centroid_func=centroid_2dg)

    if peaks is None or len(peaks) == 0:
        log.info("evaluate_dispersion_solution_fit_quality: no peaks detected; skipping fit-quality plot")
        return

    # find_peaks returns refined centroid columns when a centroid_func is supplied.
    x_peak = np.asarray(peaks['x_centroid']) + x_offset
    y_peak = np.asarray(peaks['y_centroid']) + y_offset
    log.info(f'evaluate_dispersion_solution_fit_quality: matching {len(x_peak)} detected peaks '
             'against calculated PSFlet positions')

    # Build a KD-tree on the calculated PSFlet positions and query every measured peak at once.
    log.info(f'evaluate_dispersion_solution_fit_quality: building KD-tree for calculated PSFlet positions')
    calc_points = np.column_stack([np.asarray(x_calc).ravel(), np.asarray(y_calc).ravel()])
    tree = cKDTree(calc_points)
    distances, _ = tree.query(np.column_stack([x_peak, y_peak]))

    # Optionally convert from pixel units to physical units (microns).
    if pixel_pitch is not None:
        x_plot = x_peak * pixel_pitch
        y_plot = y_peak * pixel_pitch
        distances_plot = distances * pixel_pitch
        unit_str = 'microns'
        unit_abbr = 'μm'
        vmax = 2 * np.percentile(distances_plot,75)
    else:
        x_plot = x_peak
        y_plot = y_peak
        distances_plot = distances
        unit_str = 'pixels'
        unit_abbr = 'px'
        vmax = 2 * np.percentile(distances_plot,75)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_plot, y_plot, c=distances_plot, s=8, vmax=vmax)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f'Distance between PSFlet centers\n(measured vs. calculated via fit) ({unit_abbr})')
    ax.set_aspect('equal')
    ax.set_xlim([np.min(x_plot), np.max(x_plot)])
    ax.set_ylim([np.min(y_plot), np.max(y_plot)])
    ax.set_xlabel(f'Detector X ({unit_str})')
    ax.set_ylabel(f'Detector Y ({unit_str})')
    title = 'Calculated PSFlet location error'
    if lam is not None:
        title += f' at {lam:g} nm'
    ax.set_title(title)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    if output_directory is not None:
        suffix = f'_{lam:g}nm' if lam is not None else ''
        filename = os.path.join(output_directory, f'fit_quality{suffix}.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        log.info(f'Saved fit-quality diagnostic to {filename}')


def buildcalibrations(
        par,
        filelist=None,
        lamlist=None,
        order=3,
        inspect=False,
        genwavelengthsol=False,
        makehiresPSFlets=False,
        makePolychrome=False,
        makehiresPolychrome=False,
        makePSFWidths=False,
        savehiresimages=True,
        borderpix=4,
        upsample=5,
        nsubarr=3,
        npix=13,
        parallel=False,
        inspect_first=True,
        apodize=False,
        lamsol=None,
        threshold=0.0,
        finecal=False,
        pxthreshold=2,
        findthreshold=5.,
        trimfrac=0.0,
        apdiam=3,
        halfsize=5,
        snrthreshold=10,
        initcoef=None,
        readImgs=True,
        evaluate_fit_quality=False):
    """
    Master wavelength calibration function that generates all files required to process IFS cubes.

    This function performs the following key steps:
    1. Locates PSFlets in monochromatic calibration images
    2. Fits polynomial coefficients to describe PSFlet positions as a function of wavelength
    3. Generates high-resolution models of the PSFlets using sampling diversity (if enabled)
    4. Constructs polychromatic cubes for least-squares extraction
    5. Measures PSFlet widths (if enabled)

    The calibration process uses the sampling diversity in the monochromatic images to build
    high-resolution models of the PSFlets. These models are then used to create polychromatic
    cubes that represent how the PSFlets appear at different wavelengths across the detector.

    Parameters
    ----------
    par :   Parameter instance
            Contains all IFS parameters
    filelist: list of strings (optional)
            List of the fits files that contain the monochromatic calibration files. If None (default),
            use the files in par.filelist
    lamlist: list of floats (optional)
            Wavelengths in nm at which the files are taken. If None (default),
            use the files in par.lamlist
    order: int
            Order of the polynomial used to fit the PSFLet positions across the detector
    genwavelengthsol: Boolean
            If True, generate the wavelength calibration. Creates a text file with all
            polynomial coefficients that best fit the PSFLet positions at each wavelength.
            If False, then load an already-generated file.
    inspect: Boolean
            Whether or not to create PNG files that overlay PSFLet fitted position on the
            monochromatic pictures, to visually inspect the fitting results
    inspect_first: Boolean
            Whether or not to create a PNG file that overlays PSFLet fitted position on the
            monochromatic picture of the first file, to visually inspect the fitting results
    makehiresPSFlets: Boolean
            Whether or not to do a high-resolution fitting of the PSFs, using the sampling
            diversity. This requires high-SNR monochromatic images. The high-resolution fitting
            combines multiple slightly shifted PSFlet images to reconstruct a higher resolution
            model of the PSF shape.
    makePolychrome: Boolean
            Whether or not to build the polychrome cube used in the least squares extraction
    makePSFWidths: Boolean
            Whether or not to fit the PSFLet widths using the high-res PSFLets
    makehiresPolychrome: Boolean
            Whether or not to build a polychrome cube at a high spatial resolution for future
            subpixel interpolations
    outdir: string
            Directory in which to save the generated files
    savehiresimages: Boolean
            Whether to save fits files with the high-res PSFLets
    borderpix:  int
            Number of pixels that are not taken into account towards the edges of the detector
    upsample: int
            Upsampling factor for each high-resolution PSFLet
    nsubarr: int
            Detector will be divided into nsubarr x nsubarr regions. A high-resolution PSFLet
            will be determined in each region from the average of all PSFLets within that
            region
    parallel: Boolean
            NOTE: No longer beneficial after branch made on 7/24/2026, where the make_polychrome() math 
                was improved such that single-threaded computation is now faster by ~2x. 
            Whether or not to parallelize the computation for the high-resolution PSFLet and
            polychrome computation. The wavelength calibration step cannot be parallelized since
            each wavelength uses the previous wavelength solution as a guess input.
    apodize: Boolean
            Whether to fit the spots only using lenslets within a circle, ignoring the corners of
            the detector
    lamsol: 2D array
            Optional argument that, if not None and if genwavelengthsol==False, will take the argument
            and use it as the current wavelength calibration to build the polychrome.
    threshold: float
            Threshold under which to zero out the polychrome. This is only useful for reducing
            the file size of the polychrome, and has only very little impact on the extraction.
            To be safe, for science extractions threshold should be kept at its default value of 0.0
    finecal: boolean
            Whether or not to perform fine calibration of all psflets individually through a
            centroiding routine. Default: False
    pxthreshold: float
            Threshold under which the enhanced centroiding function will accept centroid corrections.
            If a new centroid is more than pxthreshold away from a solution from the normal polynomial
            calibration, it is rejected.
    findthreshold: float
            Number of standard deviations above which we look for point sources
    trimfrac: float
            Fraction of the psflets to discard during wavelength calibration (default 0.0)
    apdiam: float
            Aperture size in pixels used for snr calculation of the wavecal
    halfsize: float
            Half-size in pixels of the search region around each PSFlet for fine wavelength calibration
    snrthreshold: float
            Threshold below which we do not accept the fine wavelength calibration result, and stick
            to the original polynomial fit
    initcoef: numpy array
            Coefficient array corresponding to an initial guess of the polynomial map. Leave to None
            in order to start from scratch.
    par.fitting_window: list of int (optional par attribute)
            [xmin, xmax, ymin, ymax] region (in full-frame detector pixels) to crop all ingested
            images to before they are passed to locatePSFlets(). Restricts the PSFlet fit to a
            sub-region of the sensor. The returned positions and polynomial coefficients are
            offset back into full-frame detector coordinates, so lamsol.dat remains compatible with
            uncropped science data. Read from the optional par.fitting_window attribute; if it is
            not set on par, the full image is used.
    evaluate_fit_quality: Boolean
            If True, produce a diagnostic scatterplot of the dispersion-solution fit quality for the
            first image in the filelist. Detected PSFlet peaks are matched to the nearest
            calculated PSFlet position and colored by that distance, helping to visualize fit
            accuracy as a function of position on the sensor. Default: False


    Notes
    -----
    This function generates all the files required to process IFS cubes:
    lamsol.dat: contains a list of the wavelengths and the polynomial coefficients that
                describe the X,Y positions of all lenslets on the detector as a function
                                of lenslet position on the lenslet array.

                                File format:
                                - column 0 is wavelength in nm
                                - columns 1..N are the polynomial coefficients for detector X followed by
                                    the polynomial coefficients for detector Y
                                - for polynomial order p, the number of coefficients after the wavelength
                                    column is (p + 1) * (p + 2)
                                - within the X block and within the Y block, terms are ordered by increasing
                                    powers of lenslet-array x and y subject to ix + iy <= p

                                For order=3, the coefficient order within each coordinate block is:
                                1, y, y^2, y^3, x, x*y, x*y^2, x^2, x^2*y, x^3

                                If i and j denote lenslet indices on the centered lenslet grid, then the row
                                at wavelength lam defines:

                                        X(i, j; lam) = sum a[ix, iy](lam) * i^ix * j^iy
                                        Y(i, j; lam) = sum b[ix, iy](lam) * i^ix * j^iy

                                The low-level implementation of this ordering lives in
                                crispy.tools.locate_psflets.transform, and the preferred high-level API for
                                arbitrary wavelengths is crispy.tools.locate_psflets.PSFLets.return_locations.

                                For a fuller prose description, see docs/source/wavelength_solution.rst.
    polychromekeyRXX.fits:  where XX is replaced by the spectral resolution defined in the
                            parameters file. This is a multi-extension fits file with:
                            - a list of the central wavelengths at which the final cube will be reduced to
                            - an array of the X positions of all lenslets
                            - an array of the Y positions of all lenslets
                            - an array of booleans indicating whether that lenslet is good or not (e.g. when it is outside of the detector area)
                            NOTE: Seems to have a lot of redundancies with PSFLoc.fits. 
    polychromeRXX.fits: 3D arrays of size num_wavelengths x Npix x Npix with maps of the PSFLets put in their correct
                        positions for each wavelength bins that we want in the output cube. Each PSFLet
                        in each wavelength slice is used for least-squares fitting.
    hiresPolychromeRXX.fits: same as polychromeRXX.fits but this time using the high-resolution PSFLets
    PSFLoc.fits:    A multi-extension fits file that gives:
                - A list of the central wavelengths for each bin of each microspectra
                - the X and Y positions of all lenslets on the detector at each central bin wavelength
                - a boolean array indicating whether each lenslet is good or not (e.g. when it is outside of the detector area)

    """
    # Optional detector crop region [xmin, xmax, ymin, ymax] for the PSFlet fit.
    # Not set by default; travels on par when the caller wants a sub-region fit.
    fitting_window = par.fitting_window if hasattr(par, 'fitting_window') else None

    if par.outdir is not None:
        outdir = par.outdir
    else:
        outdir = par.wavecalDir  # Directory to save wavelength calibration files
    if filelist is None:
        if par.filelist is None:
            raise ValueError("No filelist provided and par.filelist is None")
        else:
            filelist = par.filelist  # List of calibration image filenames
    if lamlist is None:
        if par.lamlist is None:
            raise ValueError("No lamlist provided and par.lamlist is None")
        else:
            lamlist = par.lamlist  # List of wavelengths corresponding to calibration images

    lam1 = lamlist[0]  # Shortest wavelength
    lam2 = lamlist[-1]  # Longest wavelength

    # If the output directory doesn't exist, create it
    try:
        os.makedirs(outdir)
    except OSError:
        if not os.path.isdir(outdir):
            raise OSError(f"Failed to create directory {outdir} and it is not an existing directory")

    log.info("Building calibration files; placing results in " + outdir)

    tstart = time.time()  # Start time for performance tracking
    coef = initcoef  # Initial guess for polynomial coefficients
    allcoef = []  # List to store polynomial coefficients for each wavelength
    imlist = []  # List to store calibration images
    # xlist = []  # List to store x-coordinates of PSFlet centers
    # ylist = []  # List to store y-coordinates of PSFlet centers
    dylist = []  # List to store y-offsets from polynomial fit for fine calibration
    dxlist = []  # List to store x-offsets from polynomial fit for fine calibration
    snrlist = []  # List to store SNR values for fine calibration

    halfsize = 5  # Half-size of search region around each PSFlet for fine calibration

    # Get dimensions of the first calibration image and initialize a mask variable
    ysize, xsize = Image(filename=filelist[0]).data.shape
    mask = np.ones((ysize, xsize))

    # Define a a circular region inscribed in the mask
    if apodize:
        # Create coordinate grids centered at image center
        y = np.arange(ysize) - ysize // 2
        x = np.arange(xsize) - xsize // 2
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x**2 + y**2)        # Calculate radial distance from center
        mask = (r < min(ysize, xsize) // 2)        # Create circular mask with radius equal to half the smaller dimension

    if finecal:
        log.info('Implementing experimental fine calibration method - watch out for bugs!')

    # Open up and process the images one at a time
    if readImgs:
        for i, filepath in enumerate(filelist):
            im = Image(filename=filepath)
            plt.close('all')

            # Optionally crop every ingested image to the requested fitting window before any
            # statistics, masking, or PSFlet location is performed. The mask is full-frame, so it
            # is sliced to the same region to keep im.data * mask shape-consistent in locatePSFlets.
            if fitting_window is not None:
                xmin, xmax, ymin, ymax = fitting_window
                im.data = im.data[ymin:ymax, xmin:xmax]
                mask_use = mask[ymin:ymax, xmin:xmax]
            else:
                mask_use = mask

            mean, median, std = sigma_clipped_stats(im.data, sigma=3.0, maxiters=5)
            log.info('Mean, median, std: {:}'.format((mean, median, std)))

            # Set the inverse variance to be the mask
            # hpmask = gen_bad_pix_mask(im.data)
            # mask *= hpmask
            # mask *= (im.data-median>3*std)
            imlist += [im]
            if genwavelengthsol:
                # wavelength calibration step from CHARIS. Note that when a fitting_window is used,
                # the returned positions and coefficients are in the cropped coordinate frame; the
                # working copies (coef, x, y) are kept in that frame so the chained guess for the
                # next wavelength and the finecal cutouts below stay consistent with the cropped
                # im.data.
                x, y, good, coef = locatePSFlets(im, polyorder=order, mask=mask_use, sig=1.,
                                    coef=coef, phi=par.philens,
                                    scale=par.pitch / par.pixsize, nlens=par.nlens,
                                    trimfrac=trimfrac)

                # Offset the coefficients back into full-frame detector coordinates for storage so
                # that lamsol.dat remains valid for uncropped science data. A pure origin shift only
                # affects the two constant (translation) terms of the polynomial.
                if fitting_window is not None:
                    half_coef = (order + 1) * (order + 2) // 2
                    coef_fullframe = list(coef)
                    coef_fullframe[0] += xmin
                    coef_fullframe[half_coef] += ymin
                else:
                    coef_fullframe = list(coef)
                allcoef += [[lamlist[i]] + list(coef_fullframe)]

                # Evaluate the dispersion-solution fit quality for the first image only. The
                # detected peaks (from the cropped im.data) and the calculated positions are both
                # offset into full-frame detector coordinates so the diagnostic plot matches the
                # convention of the other calibration outputs.
                if evaluate_fit_quality:
                    x_offset = fitting_window[0] if fitting_window is not None else 0
                    y_offset = fitting_window[2] if fitting_window is not None else 0
                    evaluate_dispersion_solution_fit_quality(
                        im.data, x + x_offset, y + y_offset, outdir,
                        lam=lamlist[i], x_offset=x_offset, y_offset=y_offset, pixel_pitch=par.pixsize * 1E6)

                if finecal:
                    log.info('Finding individual centroids (experimental)')
                    # CRISPY-specific enhanced wavelength calibration step
                    dy = np.zeros_like(y)
                    dx = np.zeros_like(x)
                    snr = np.zeros_like(x)
                    mgrid = np.arange(2 * halfsize)
                    xgrid, ygrid = np.meshgrid(mgrid, mgrid)

                    for j in range(x.shape[0]):
                        for k in range(x.shape[1]):
                            xl = x[j, k]
                            yl = y[j, k]
                            xmin = int(xl - halfsize) + 1
                            ymin = int(yl - halfsize) + 1
                            if ymin > 0 and xmin > 0 and xmin + 2 * halfsize < xsize and ymin + 2 * halfsize < ysize:
                                # define cutout
                                cutout = im.data[ymin:ymin + 2 * halfsize, xmin:xmin + 2 * halfsize] - median

                                # here is the new centroiding function: we could change this to something more robust
                                dx[j, k], dy[j, k] = centroid_com(cutout)

                                # mask used for elementary aperture photometry
                                apmask = (xgrid - dx[j, k])**2 + (ygrid - dy[j, k])**2 < apdiam**2
                                apval = np.nansum(apmask * cutout)
                                # snr[j,k] = apval/(np.sqrt(np.nansum(apmask))*std)

                                # estimate of SNR, only valid for very high fluxes, could do better
                                snr[j, k] = np.sqrt(apval)
                                dy[j, k] -= y[j, k] - ymin
                                dx[j, k] -= x[j, k] - xmin

                    # Thresholding
                    dy[snr < snrthreshold] = 0.0
                    dx[snr < snrthreshold] = 0.0

                    # ignore if new centroid is too out of whack
                    dy[np.abs(dy) > pxthreshold] = 0.0
                    dx[np.abs(dx) > pxthreshold] = 0.0

                    dylist += [dy]
                    dxlist += [dx]
                    snrlist += [snr]

                    if inspect:
                        do_inspection(par, im.data, x + dx, y + dy, lamlist[i])
                    elif inspect_first and i == 0:
                        do_inspection(par, im.data, x + dx, y + dy, lamlist[i])

                else:
                    if inspect:
                        do_inspection(par, im.data, x, y, lamlist[i])
                    elif inspect_first and i == 0:
                        do_inspection(par, im.data, x, y, lamlist[i])

    if genwavelengthsol:
        log.info(f"Saving wavelength solution to {os.path.join(outdir, 'lamsol.dat')}")
        allcoef = np.asarray(allcoef)
        np.savetxt(f'{os.path.join(outdir, "lamsol.dat")}', allcoef)
        lam = allcoef[:, 0]  # Unnecessary duplicate of 'lamlist' from earlier?
        allcoef = allcoef[:, 1:]  # Strip away the first element (the wavelength) from each row of the master coefficient table 

        if finecal:
            log.info('Exporting fine calibration products...')
            xlistarr = np.array(dxlist)
            ylistarr = np.array(dylist)
            snrlistarr = np.array(snrlist)
            out = fits.HDUList(fits.PrimaryHDU(xlistarr.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'dxlistarr.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(ylistarr.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'dylistarr.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(snrlistarr.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'snrlistarr.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(np.mean(ylistarr, axis=0).T.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'dylistarr_mean.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(np.std(ylistarr, axis=0).T.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'dylistarr_std.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(np.mean(xlistarr, axis=0).T.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'dxlistarr_mean.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(np.std(xlistarr, axis=0).T.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'dxlistarr_std.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(np.mean(snrlistarr, axis=0).T.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'snrlistarr_mean.fits'), overwrite=True)
            out = fits.HDUList(fits.PrimaryHDU(np.std(snrlistarr, axis=0).T.astype(np.float32)))
            out.writeto(os.path.join(outdir, 'snrlistarr_std.fits'), overwrite=True)

    else:
        log.info("Loading wavelength solution from " + os.path.join(outdir, "lamsol.dat"))
        lam = np.loadtxt(os.path.join(outdir, "lamsol.dat"))[:, 0]
        allcoef = np.loadtxt(os.path.join(outdir, "lamsol.dat"))[:, 1:]

        if finecal:
            ylistarr = fits.getdata(os.path.join(outdir, 'dylistarr.fits'))
            xlistarr = fits.getdata(os.path.join(outdir, 'dxlistarr.fits'))
            snrlistarr = fits.getdata(os.path.join(outdir, 'snrlistarr.fits'))

    if finecal:
        finexy = [np.nanmean(xlistarr, axis=0), np.nanmean(ylistarr, axis=0), np.amin(snrlistarr, axis=0)]
    else:
        finexy = None

    log.info("Computing wavelength values at pixel centers")
    psftool = PSFLets()
    psftool.genpixsol(
        par,
        lam,
        allcoef,
        order=order,
        lam1=lam1 / 1.01,
        lam2=lam2 * 1.01,
        borderpix=borderpix,
        finexy=finexy)
    psftool.savepixsol(outdir=outdir)

    xindx = np.arange(-par.nlens // 2, par.nlens // 2) + 1
    xindx, yindx = np.meshgrid(xindx, xindx)

    if makehiresPSFlets:

        hires_arrs = makeHires(
            par,
            xindx,
            yindx,
            lam,
            allcoef,
            psftool,
            imlist=imlist,
            savehiresimages=savehiresimages,
            upsample=upsample,
            nsubarr=nsubarr,
            npix=npix,
            parallel=parallel,
            finexy=finexy,
            reflam=lam)

    hires_list = np.sort(glob.glob(os.path.join(par.wavecalDir, 'hires_psflets_lam???.fits')))
    # Now generate some arrays that describe the PSFLet width as a function of various things
    if makePSFWidths:
        log.info("Computing PSFLet widths...")
        if not makehiresPSFlets:
            hires_arrs = [fits.open(filename)[0].data for filename in hires_list]
            lam_hires = [int(re.sub('.*lam', '', re.sub('.fits', '', filename)))
                         for filename in hires_list]
        else:
            lam_hires = lam.copy()

        hires_shape = hires_arrs[0].shape
        # Initialize an array to hold the PSF widths for each subarray and wavelength
        sigma_vs_subarray = np.zeros((len(hires_list), hires_shape[0], hires_shape[1]))  

        # Create an x-axis array centered at zero for the high-resolution PSFLets
        _x = np.arange(hires_shape[2]) / float(upsample)
        _x -= _x[_x.shape[0] // 2]

        # Measure the gaussian sigma across a slice that is the average of ~3 columns
        # NOTE that this method of calculating sigma breaks down if:
        # _x is not centered at 0, PSF is not centered at _x=0, PSF is not well-approximated by a Gaussian
        # or the PSF sigma is comparable to the window size, the PSF is on a nonzero background.
        # This method might be computationally faster than fitting a Gaussian, but it is not as robust. Perhaps it is sufficient.
        for i in range(sigma_vs_subarray.shape[0]):
            for j in range(sigma_vs_subarray.shape[1]):
                for k in range(sigma_vs_subarray.shape[2]):
                    row = np.sum(hires_arrs[i][j, k, :, hires_shape[3] // 2 - 1:hires_shape[3] // 2 + 2], axis=1)
                    sigma_vs_subarray[i, j, k] = np.sum(row * _x**2)
                    sigma_vs_subarray[i, j, k] /= np.sum(row)

            sigma_vs_subarray[i] = np.sqrt(sigma_vs_subarray[i])

        mean_x = psftool.xindx[:, :, psftool.xindx.shape[-1] // 2]
        mean_y = psftool.yindx[:, :, psftool.yindx.shape[-1] // 2]

        # Initialize an array to store PSF widths for each lenslet, at each calibration wavelength
        sigma_vs_calwavelength = np.zeros((len(lam_hires), mean_x.shape[0], mean_x.shape[1]))

        ix = mean_x * hires_arrs[0].shape[1] / par.npix - 0.5  # x-coordinates of lenslets in the high-resolution PSFlet array
        iy = mean_y * hires_arrs[0].shape[0] / par.npix - 0.5  # y-coordinates of lenslets in the high-resolution PSFlet array

        for i in range(sigma_vs_subarray.shape[0]):
            sigma_vs_calwavelength[i] = ndimage.map_coordinates(sigma_vs_subarray[i], [iy, ix], order=3, mode='nearest')

        # Initialize an array for storing PSF widths for each lenslet at each "pixel wavelength" (i.e. the wavelength at each pixel in the detector)
        sigma_vs_pixelwavelength = np.ones((psftool.xindx.shape))  
        for i in range(mean_x.shape[0]):
            for j in range(mean_x.shape[1]):
                if psftool.good[i, j]:
                    fit = interpolate.interp1d(np.asarray(lam_hires), sigma_vs_calwavelength[:, i, j],
                                               bounds_error=False, fill_value='extrapolate')
                    sigma_vs_pixelwavelength[i, j] = fit(psftool.lam_indx[i, j])

        # Save this cube of PSFwidths vs. pixel wavelengths to a .fits file
        log.info("Saving PSFLet widths to " + os.path.join(outdir, "PSFwidths.fits"))
        out = fits.HDUList(fits.PrimaryHDU(sigma_vs_pixelwavelength.astype(np.float32)))
        out.writeto(os.path.join(outdir, 'PSFwidths.fits'), overwrite=True)

        # Also save this to a .fits file with the contents of PSFloc.fits for convenience
        log.info("Also saving PSFLet widths to " + os.path.join(outdir, "calib.fits") + " along with PSFloc.fits contents")
        calib_hdus = fits.open(os.path.join(outdir, 'PSFloc.fits'))
        outkey = fits.HDUList(calib_hdus[0])
        outkey.append(calib_hdus[1])
        outkey.append(calib_hdus[2])
        outkey.append(calib_hdus[3])
        outkey.append(calib_hdus[4])
        outkey.append(fits.PrimaryHDU(sigma_vs_pixelwavelength.astype(np.float32)))
        outkey.writeto(os.path.join(outdir, 'calib.fits'), overwrite=True)

    if makePolychrome:
        if not makehiresPSFlets:
            hires_arrs = [fits.open(filename)[0].data for filename in hires_list]

        # Pre-filter each calibration PSFlet array once, ahead of the wavelength-bin loop
        # below, instead of re-filtering the (linearly interpolated) hires array on every
        # sub-wavelength inside make_polychrome. spline_filter is linear and commutes with
        # that interpolation, so this produces numerically equivalent results while avoiding
        # ~nlam x num_wavelengths redundant filter passes over the same calibration data.
        # A separate list is built (rather than filtering hires_arrs in place) since
        # hires_arrs may be reused below for makehiresPolychrome.
        log.info('Prefiltering hires_arrs...')
        hires_arrs_prefiltered = [h.astype(np.float64).copy() for h in hires_arrs]
        for h in hires_arrs_prefiltered:
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    h[i, j] = ndimage.spline_filter(h[i, j])

        # Create an array of wavelengths that represent the midpoints/endpoints of the wavelength bins
        lam_midpts, lam_endpts = calculateWaveList(par, lam, method='lstsq')
        # TODO, rename all instances of 'num_wavelengths' to 'num_wavelengths' for clarity. 
        num_wavelengths = len(lam_endpts)  # The number of unique wavelength bins
        polyimage = np.zeros((num_wavelengths - 1, ysize, xsize))

        # Initialize some arrays where we will store information about the x/y position of each PSF, 
        # as well as whether or not that PSF is "good" (i.e. falls on the detector)
        xpos = []
        ypos = []
        good = []

        log.info('Making polychrome cube')
        if not parallel:
            for i in range(num_wavelengths - 1):
                log.info(f'  Wavelength bin {i + 1} of {num_wavelengths - 1}')
                polyimage[i] = (lam_endpts[i + 1] - lam_endpts[i]) * make_polychrome(lam_endpts[i],
                                                                                     lam_endpts[i + 1],
                                                                                     hires_arrs_prefiltered,
                                                                                     lam,
                                                                                     psftool,
                                                                                     allcoef,
                                                                                     xindx,
                                                                                     yindx,
                                                                                     ysize,
                                                                                     xsize,
                                                                                     finexy=finexy,
                                                                                     reflam=lam,
                                                                                     upsample=upsample,
                                                                                     prefiltered=True,)
                _x, _y = psftool.return_locations(lam_midpts[i], allcoef, xindx, yindx)
                if finecal:
                    _x += finexy[0]
                    _y += finexy[1]

                # Append the x/y positions and "good" boolean array to the lists.
                # If the wavelength calibration was restricted to a sub-region of the
                # detector (par.fitting_window), use those bounds instead of the full
                # detector so that lenslets outside the calibrated area are excluded,
                # matching the validity check done in PSFLets.genpixsol().
                if fitting_window is None:
                    _good = (_x > borderpix) * (_x < xsize - borderpix) * \
                        (_y > borderpix) * (_y < ysize - borderpix)
                else:
                    xmin, xmax, ymin, ymax = fitting_window
                    _good = (_x > xmin + borderpix) * (_x < xmax - borderpix) * \
                        (_y > ymin + borderpix) * (_y < ymax - borderpix)
                xpos += [_x]
                ypos += [_y]
                good += [_good]
        else:
            # Each wavelength bin's polychrome slice is computed on a worker thread. The heavy
            # numpy/scipy.ndimage work releases the GIL, so threads speed this up while sharing
            # memory, avoiding the pickling/spawn hazards of the old multiprocessing.Process pool.
            def _poly_task(i):
                return make_polychrome(lam_endpts[i], lam_endpts[i + 1], hires_arrs_prefiltered, lam,
                                       psftool, allcoef, xindx, yindx, ysize,
                                       xsize, finexy, lam, upsample, prefiltered=True)

            # executor.map yields results in submission order, so index == i throughout.
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for i, poly in enumerate(executor.map(_poly_task, range(num_wavelengths - 1))):
                    print(f'  [makePolychrome] Completed {i + 1} of {num_wavelengths - 1} wavelength bins', flush=True)
                    polyimage[i] = poly * \
                        (lam_endpts[i + 1] - lam_endpts[i])
                    _x, _y = psftool.return_locations(lam_midpts[i], allcoef, xindx, yindx)
                    if finecal:
                        _x += finexy[0]
                        _y += finexy[1]

                    # Append the x/y positions and "good" boolean array to the lists.
                    # See the non-parallel branch above for why fitting_window is used
                    # here when set.
                    if fitting_window is None:
                        _good = (_x > borderpix) * (_x < xsize - borderpix) * \
                            (_y > borderpix) * (_y < ysize - borderpix)
                    else:
                        xmin, xmax, ymin, ymax = fitting_window
                        _good = (_x > xmin + borderpix) * (_x < xmax - borderpix) * \
                            (_y > ymin + borderpix) * (_y < ymax - borderpix)
                    xpos += [_x]
                    ypos += [_y]
                    good += [_good]

        log.info('Saving polychrome cube')
        polyimage[polyimage < threshold] = 0.0
        out = fits.HDUList(fits.PrimaryHDU(polyimage.astype(np.float32)))
        out.writeto(f"{os.path.join(outdir, f'polychromeR{par.R}.fits.gz')}", overwrite=True)
        out = fits.HDUList(fits.PrimaryHDU(np.sum(polyimage, axis=0).astype(np.float32)))
        out.writeto(f"{os.path.join(outdir, f'polychromeR{par.R}stack.fits.gz')}", overwrite=True)

    else:
        lam_midpts, lam_endpts = calculateWaveList(par, lam, method='lstsq')
        xpos = []
        ypos = []
        good = []

        for i in range(len(lam_midpts)):
            _x, _y = psftool.return_locations(
                lam_midpts[i], allcoef, xindx, yindx)
            if finecal:
                _x += finexy[0]
                _y += finexy[1]
            _good = (_x > borderpix) * (_x < xsize - borderpix) * \
                (_y > borderpix) * (_y < ysize - borderpix)
            xpos += [_x]
            ypos += [_y]
            good += [_good]

    # Save an array with information about the x/y position of each PSF, as well as whether or not that PSF is "good"
    log.info('Saving wavelength calibration cube')
    outkey = fits.HDUList(fits.PrimaryHDU(lam_midpts))
    outkey.append(fits.PrimaryHDU(np.asarray(xpos)))
    outkey.append(fits.PrimaryHDU(np.asarray(ypos)))
    outkey.append(fits.PrimaryHDU(np.asarray(good).astype(np.uint8)))
    if par.R is not None:
        filename = os.path.join(outdir, f'polychromekeyR{par.R}.fits')
    else:
        filename = os.path.join(outdir, 'polychromekey.fits')
    outkey.writeto(filename, overwrite=True)

    if makehiresPolychrome:
        log.info('Making high-resolution polychrome cube (can use lots of memory)')
        if not makehiresPSFlets:
            hires_list = np.sort(glob.glob(f"{par.wavecalDir}hires_psflets_lam???.fits"))
            hires_arrs = [fits.open(filename)[0].data for filename in hires_list]

        # As above (see makePolychrome block), pre-filter each calibration PSFlet array once
        # rather than re-filtering it on every sub-wavelength inside make_hires_polychrome.
        # Rebuilt here (rather than reusing the makePolychrome block's version) since
        # hires_arrs may have just been reloaded above.
        log.info('Prefiltering hires_arrs...')
        hires_arrs_prefiltered = [h.astype(np.float64).copy() for h in hires_arrs]
        for h in hires_arrs_prefiltered:
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    h[i, j] = ndimage.spline_filter(h[i, j])

        lam_midpts, lam_endpts = calculateWaveList(par, lam, method='lstsq')
        num_wavelengths = len(lam_endpts)
        hirespoly = np.zeros((num_wavelengths - 1, ysize * upsample, xsize * upsample))

        if not parallel:
            for i in range(num_wavelengths - 1):
                hirespoly[i] = (lam_endpts[i + 1] - lam_endpts[i]) * make_hires_polychrome(lam_endpts[i],
                                                                                           lam_endpts[i + 1],
                                                                                           hires_arrs_prefiltered,
                                                                                           lam,
                                                                                           psftool,
                                                                                           allcoef,
                                                                                           xindx,
                                                                                           yindx,
                                                                                           ysize,
                                                                                           xsize,
                                                                                           upsample=upsample,
                                                                                           prefiltered=True) / upsample**2
        else:
            # Each wavelength bin's high-res polychrome slice is computed on a worker thread. The
            # heavy numpy/scipy.ndimage work releases the GIL, so threads speed this up while
            # sharing memory, avoiding the pickling/spawn hazards of the old
            # multiprocessing.Process pool.
            def _hirespoly_task(i):
                return make_hires_polychrome(lam_endpts[i], lam_endpts[i + 1], hires_arrs_prefiltered, lam,
                                             psftool, allcoef, xindx, yindx,
                                             ysize, xsize, upsample, prefiltered=True)

            # executor.map yields results in submission order, so index == i throughout.
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                for i, poly in enumerate(executor.map(_hirespoly_task, range(num_wavelengths - 1))):
                    print(f'  [makeHiresPolychrome] Completed {i + 1} of {num_wavelengths - 1} wavelength bins', flush=True)
                    hirespoly[i] = poly * \
                        (lam_endpts[i + 1] - lam_endpts[i]) / upsample**2

        log.info('Saving hi-res polychrome cube')
        out = fits.HDUList(fits.PrimaryHDU(hirespoly.astype(np.float32)))
        out.writeto(f"{os.path.join(outdir, f'hirespolychromeR{par.R}.fits.gz')}", overwrite=True)
        out = fits.HDUList(fits.PrimaryHDU(np.sum(hirespoly, axis=0).astype(np.float32)))
        out.writeto(f"{os.path.join(outdir, f'hiresPolychromeR{par.R}stack.fits')}", overwrite=True)

    log.info(f"Total time elapsed: {time.time() - tstart:.0f} s")


def derivative_of_lamsol_at_wavelength(lenslet_ind_x, lenslet_ind_y, lamsol_df, wavelength, plot_mosaic=False, lamsol_fit_order=4):
    """
    Compute the derivative of coefficients with respect to wavelength at a specific wavelength.

    Parameters:
    -----------
    lenslet_ind_x : array
        x indices of lenslets
    lenslet_ind_y : array
        y indices of lenslets
    lamsol_df : pandas.DataFrame
        Contents of the lamsol data file. Stored as a DataFrame where the index is wavelength and columns contain coefficients
    wavelength : float
        The specific wavelength at which to evaluate the derivative
    plot_mosaic : bool, optional
        Whether to plot a mosaic of wavelength vs. coefficient plots with the best-fit polynomial overlaid
    order : int, optional
        Order of the polynomial fit to use for the coefficients

    Returns:
    --------
    dx_dlambda : array
        The derivative of the x-coefficients with respect to wavelength evaluated at the specified wavelength
    dy_dlambda : array
        The derivative of the y-coefficients with respect to wavelength evaluated at the specified wavelength

    """
    # Intelligently guess the order of the polynomial fit from the number of coefficients 
    # Remember that the 0th column is not a coefficient, but a wavelength.
    num_coeffs = len(lamsol_df.columns) - 1
    for order in range(10):
        if (order + 1) * (order + 2) == num_coeffs:
            break 

    # fit a low-order polynomial to each coefficient as a function of wavelength
    # and take the derivative of that polynomial at the specified wavelength
    # Optionally, plot a mosaic of wavelength vs. coefficient plots with the best-fit polynomial overlaid
    if plot_mosaic:
        nrows, ncols = 4, (num_coeffs + 3) // 4
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 8))
    coefficient_derivatives = []
    for i, col in enumerate(lamsol_df.columns[1:]):
        poly = np.poly1d(np.polyfit(lamsol_df[0], lamsol_df[col], deg=lamsol_fit_order))  # Fit a polynomial
        dpoly = poly.deriv()  # Derivative of the polynomial
        coefficient_derivatives.append(dpoly(wavelength))  # Evaluate the derivative at the specified wavelength

        # Optionally, fill out the mosaic with the best-fit polynomial and the data points
        if plot_mosaic:
            ax[i // (nrows + 1), i % ncols].scatter(lamsol_df[0], lamsol_df[col], label='Data')
            ax[i // (nrows + 1), i % ncols].plot(lamsol_df[0], poly(lamsol_df[0]), 'r--', label='Best-fit')
            ax[i // (nrows + 1), i % ncols].set_title(f'Coefficient {i}')
            # WRONG, DO NOT USE: ax[i // (nrows + 1), i % ncols].set_title(f'Coefficient {i}, X^{i // (order +1)} Y^{i % (order + 1)}')
    if plot_mosaic:
        fig.tight_layout()
        plt.show(block=False)

    # Create some blank arrays that we will fill in with dx/dlambda and dy/dlambda values
    dx_dlambda = np.zeros(np.asarray(lenslet_ind_x).shape)
    dy_dlambda = np.zeros(np.asarray(lenslet_ind_y).shape)

    # Calculate dx/dlambda and dy/dlambda using the standard method
    i = 0
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            # term_to_add = coefficient_derivatives[i] * lenslet_ind_x**ix * lenslet_ind_y**iy
            # print(f'Coefficient {i} (X^{ix} Y^{iy}) -> {term_to_add}')
            dx_dlambda += coefficient_derivatives[i] * lenslet_ind_x**ix * lenslet_ind_y**iy
            # print(f'X^{ix} Y^{iy} -> {coefficient_derivatives[i] * lenslet_ind_x**ix * lenslet_ind_y**iy}')
            i += 1
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            dy_dlambda += coefficient_derivatives[i] * lenslet_ind_x**ix * lenslet_ind_y**iy
            # print(f'X^{ix} Y^{iy} -> {coefficient_derivatives[i] * lenslet_ind_x**ix * lenslet_ind_y**iy}')
            i += 1
    
    # TEMPORARY, TO BE DELETED. Just making sure we can calculate the powers on the x/y terms 
    # for i in range(20):
    #     print(f'Coefficient {i}: X^{i // (order +1)} Y^{i % ((order + 1) - i // (order +1))}')

    return dx_dlambda, dy_dlambda

def interpolate_lamsol_wavelengths(lamsol_df, target_wavelengths=None, lamsol_fit_order=4, plot_mosaic=False):
    """
    Interpolate the lamsol coefficients at specific target wavelengths using polynomial fits.

    Parameters:
    -----------
    lamsol_df : pandas.DataFrame
        Contents of the lamsol data file. Stored as a DataFrame where the index is wavelength and columns contain coefficients
    target_wavelengths : list
        List of wavelengths at which to interpolate the coefficients
    lamsol_fit_order : int, optional
        Order of the polynomial fit to use for the coefficients
    plot_mosaic : bool, optional
        Whether to plot a mosaic of wavelength vs. coefficient points with best-fit polynomials overlaid

    Returns:
    --------
    interpolated_df : pandas.DataFrame
        DataFrame with the same format as lamsol_df, where each row corresponds to a target wavelength
        and columns contain the interpolated coefficients at that wavelength
    """
    if target_wavelengths is None:
        raise ValueError("No target wavelengths provided for interpolation")

    # Fit one polynomial per coefficient column.
    fitted_polynomials = []
    for col in lamsol_df.columns[1:]:
        fitted_polynomials.append(np.poly1d(np.polyfit(lamsol_df[0], lamsol_df[col], deg=lamsol_fit_order)))

    if plot_mosaic:
        num_coeffs = len(fitted_polynomials)
        nrows, ncols = 4, (num_coeffs + 3) // 4
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11, 8))
        ax = np.atleast_1d(ax).ravel()
        for i, (col, poly) in enumerate(zip(lamsol_df.columns[1:], fitted_polynomials)):
            ax[i].scatter(lamsol_df[0], lamsol_df[col], label='Data')
            ax[i].plot(lamsol_df[0], poly(lamsol_df[0]), 'r--', label='Best-fit')
            ax[i].set_title(f'Coefficient {i}')
        for i in range(num_coeffs, len(ax)):
            ax[i].axis('off')
        fig.tight_layout()

    rows = []
    for wavelength in target_wavelengths:
        coeffs_at_wavelength = [wavelength]
        for poly in fitted_polynomials:
            coeffs_at_wavelength.append(poly(wavelength))  # Evaluate the polynomial at the target wavelength
        rows.append(coeffs_at_wavelength)

    return pd.DataFrame(rows, columns=lamsol_df.columns)

# Display a plot of dispersion values at a few different wavelengths
def illustrate_dispersion(wavelengths_to_plot, lamsol_filepath, nlens, output_directory=None, lamsol_fit_order=4,
                          sensor_dimensions=[6248, 4176], lenslets_to_plot=None,
                          sensor_regions_to_inspect=[[3124, 2088]], window_size=100,
                          plotting_bounds_pixels=None):
    """
    Generate plots illustrating the dispersion map at specified wavelengths,
    and a plot of dispersion vs. wavelength for one or more user-specified lenslets.

    Parameters:
    -----------
    wavelengths_to_plot : float or list
        Wavelengths at which to display a dispersion map
    nlens : int
        Number of lenslets along one dimension of the array
    lamsol_filepath : str
        Filepath to the lamsol.dat file
    output_directory : str, optional
        Directory path where plots should be saved. If None, plots are not saved.
    order : int, optional
        Order of the polynomial fit to use for the lamsol coefficients
    sensor_dimensions : tuple, optional
        Dimensions of the sensor in pixels, used for plotting boundaries of the detector
    lenslets_to_plot : list[list[int, int]], optional
        Lenslet coordinates to use for the dispersion-vs-wavelength plot, in the form
        [[x1, y1], [x2, y2], ...] where [0, 0] corresponds to the center lenslet.
        If None, defaults to [[0, 0]].
    sensor_regions_to_inspect : list[list[float, float]], optional
        List of [x, y] sensor pixel coordinates at which to produce a scatter plot of
        PSFlet positions colored by wavelength. Each plot is a window_size x window_size
        region centered on the given coordinate. If None, defaults to the center of the
        sensor: [[sensor_dimensions[0]/2, sensor_dimensions[1]/2]].
    window_size : int, optional
        Side length in pixels of the square window around each region of interest. Default 100.
    plotting_bounds_pixels : list[float, float, float, float], optional
        Axis limits [xmin, xmax, ymin, ymax] for the dispersion scale maps (dispersion-at-wavelength
        and trace length plots). The colormap is also scaled to values that fall within this region.
        If None, uses the full sensor extent defined by sensor_dimensions.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray | dict]
        Wavelength array and dispersion values. If one lenslet is provided, returns
        a 1D dispersion array for backward compatibility. If multiple lenslets are
        provided, returns a dictionary keyed by (x_lenslet, y_lenslet).
    """
    # Read in the lamsol data file
    lamsol_df = pd.read_csv(lamsol_filepath, delimiter=' ', engine='python', header=None)

    # Intelligently guess the order of the polynomial fit from the number of coefficients 
    # Remember that the 0th column is not a coefficient, but a wavelength.
    num_coeffs = lamsol_df.shape[1] - 1
    for order in range(10):
        if (order + 1) * (order + 2) == num_coeffs:
            break
    print('Guessed polynomial order for lamsol coefficients: ', order)

    # Generate some arrays that represent the x/y indices of the lenslets
    # Also, create a mask of lenslets that fall on the detector
    lenslet_ind_x = np.arange(-nlens // 2, nlens // 2) + 1
    lenslet_ind_x, lenslet_ind_y = np.meshgrid(lenslet_ind_x, lenslet_ind_x)

    #########################################################################
    # Pre-compute master table of (x, y, wavelength) for all lenslets x all wavelengths
    #########################################################################
    rows = []
    for _, row in lamsol_df.iterrows():
        wl = row[0]
        coef = row.values[1:]
        xt, yt = transform(lenslet_ind_x, lenslet_ind_y, order=order, coef=coef)
        rows.append(pd.DataFrame({'x': xt.ravel(), 'y': yt.ravel(), 'wavelength': wl}))
    psf_table = pd.concat(rows, ignore_index=True)

    #########################################################################
    # Display a dispersion map at each wavelength of interest
    #########################################################################

    if plotting_bounds_pixels is not None:
        xmin_plot, xmax_plot, ymin_plot, ymax_plot = plotting_bounds_pixels
    else:
        xmin_plot, xmax_plot, ymin_plot, ymax_plot = 0, sensor_dimensions[0], 0, sensor_dimensions[1]

    for wavelength in wavelengths_to_plot:
        # Determine the x/y coordinates of the lenslets on the detector, at the wavelength closest to the desired wavelength
        coefficients = lamsol_df.loc[(lamsol_df[0] - wavelength).abs().idxmin()].values[1:]
        x_transformed, y_transformed = transform(lenslet_ind_x, lenslet_ind_y, order=order, coef=coefficients)
        mask = (x_transformed >= 0) & (x_transformed < sensor_dimensions[0]) & \
            (y_transformed >= 0) & (y_transformed < sensor_dimensions[1])

        dx_dlambda, dy_dlambda = derivative_of_lamsol_at_wavelength(lenslet_ind_x, lenslet_ind_y, lamsol_df, wavelength, plot_mosaic=False, lamsol_fit_order=lamsol_fit_order)
        dispersion_nm_per_pix = 1 / np.sqrt(dx_dlambda**2 + dy_dlambda**2)

        bounds_mask = mask & \
            (x_transformed >= xmin_plot) & (x_transformed <= xmax_plot) & \
            (y_transformed >= ymin_plot) & (y_transformed <= ymax_plot)
        vmin = dispersion_nm_per_pix[bounds_mask].min() if bounds_mask.any() else None
        vmax = dispersion_nm_per_pix[bounds_mask].max() if bounds_mask.any() else None

        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(x_transformed[mask], y_transformed[mask], c=dispersion_nm_per_pix[mask], s=20, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Dispersion (nm/pixel)')
        ax.set_xlim(xmin_plot, xmax_plot)
        ax.set_ylim(ymin_plot, ymax_plot)
        ax.set_aspect('equal')
        ax.set_xlabel('Detector X (pixels)')
        ax.set_ylabel('Detector Y (pixels)')
        ax.set_title(f'Dispersion at {wavelength} nm')
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        

        if output_directory is not None:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            filename = os.path.join(output_directory, f'dispersion_map_{wavelength}nm.png')
            fig.savefig(filename, dpi=300, bbox_inches='tight')

    #########################################################################
    # Plot dispersion vs. wavelength for one or more user-specified lenslets
    #########################################################################
    if lenslets_to_plot is None:
        lenslets_to_plot = [[0, 0]]

    lenslet_keys = [tuple(lenslet_pair) for lenslet_pair in lenslets_to_plot]
    dispersion_by_lenslet = {lenslet_key: [] for lenslet_key in lenslet_keys}

    for wavelength in lamsol_df[0]:
        for lenslet_key in lenslet_keys:
            lenslet_x, lenslet_y = lenslet_key
            if not (-nlens // 2 <= lenslet_x <= nlens // 2 - 1 and -nlens // 2 <= lenslet_y <= nlens // 2 - 1):
                raise ValueError(
                    f"Lenslet {list(lenslet_key)} is out of bounds for nlens={nlens}. "
                    f"Allowed lenslet coordinates are [{-nlens // 2}, {nlens // 2 - 1}] on each axis."
                )

            # if (wavelength == lamsol_df[0].iloc[0]) and (lenslet_key == lenslet_keys[0]):
            #     plot_mosaic = True
            # else:
            #     plot_mosaic = False
            dx_dlambda, dy_dlambda = derivative_of_lamsol_at_wavelength(
                lenslet_x,
                lenslet_y,
                lamsol_df,
                wavelength,
                lamsol_fit_order=lamsol_fit_order,
                plot_mosaic=False
            )
            dispersion_nm_per_pix = 1 / np.sqrt(dx_dlambda**2 + dy_dlambda**2)
            dispersion_by_lenslet[lenslet_key].append(dispersion_nm_per_pix)

    # Compute a single representative pixel position per lenslet at the median wavelength
    mid_wavelength = lamsol_df[0].iloc[len(lamsol_df) // 2]
    mid_coef = lamsol_df.loc[(lamsol_df[0] - mid_wavelength).abs().idxmin()].values[1:]
    lenslet_pixel_positions = {}
    for lenslet_key in lenslet_keys:
        lenslet_x, lenslet_y = lenslet_key
        px, py = transform(lenslet_x, lenslet_y, order=order, coef=mid_coef)
        lenslet_pixel_positions[lenslet_key] = (float(px), float(py))

    fig, (ax_disp, ax_pos) = plt.subplots(1, 2, figsize=(10, 4))

    colors = plt.cm.tab10.colors
    for i, lenslet_key in enumerate(lenslet_keys):
        color = colors[i % len(colors)]
        ax_disp.plot(
            lamsol_df[0],
            dispersion_by_lenslet[lenslet_key],
            label=f'Lenslet {list(lenslet_key)}',
            color=color
        )
        px, py = lenslet_pixel_positions[lenslet_key]
        ax_pos.scatter(px, py, color=color, s=60, label=f'Lenslet {list(lenslet_key)}', zorder=3)

    # ax_disp.set_xlim(550,800)
    # ax_disp.set_ylim(2,3.4)
    ax_disp.set_xlabel('Wavelength (nm)')
    ax_disp.set_ylabel('Dispersion (nm/pixel)')
    ax_disp.set_title('Dispersion vs. Wavelength')
    ax_disp.grid(True, alpha=0.3)
    legend = ax_disp.legend()
    legend.set_zorder(1000)

    ax_pos.set_xlabel('Detector X (pixels)')
    ax_pos.set_ylabel('Detector Y (pixels)')
    ax_pos.set_title('Lenslet Positions')
    ax_pos.set_aspect('equal')
    if lenslet_pixel_positions:
        all_px = [v[0] for v in lenslet_pixel_positions.values()]
        all_py = [v[1] for v in lenslet_pixel_positions.values()]
        x_margin = max((max(all_px) - min(all_px)) * 0.05, 1)
        y_margin = max((max(all_py) - min(all_py)) * 0.05, 1)
        ax_pos.set_xlim(min(all_px) - x_margin, max(all_px) + x_margin)
        ax_pos.set_ylim(min(all_py) - y_margin, max(all_py) + y_margin)
    legend_pos = ax_pos.legend()
    legend_pos.set_zorder(1000)
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    if output_directory is not None:
        filename = os.path.join(output_directory, f'dispersion_vs_wavelength.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')

    #########################################################################
    # Plot per-lenslet spectral-trace clocking angle
    #########################################################################
    # For each lenslet, the set of (x, y) positions across all wavelengths traces a near-linear
    # spectral trace on the detector. We fit the principal axis of that point set (total least
    # squares / PCA) and report its clocking angle relative to the detector x-axis.
    #
    # x_tracks / y_tracks: 2-D arrays, shape (n_wavelengths, n_lenslets_flat).
    #   Row i   = detector pixel positions of every lenslet at wavelength i.
    #   Column j = the pixel track of lenslet j across all wavelengths (its spectral trace).
    # The lenslet ordering is identical at every wavelength (ravel of the same lenslet grid),
    # so column j always refers to the same physical lenslet.
    x_tracks = []
    y_tracks = []
    for _, row in lamsol_df.iterrows():
        coef_row = row.values[1:]  # polynomial coefficients for this wavelength (skip col 0 = wavelength)
        xt, yt = transform(lenslet_ind_x, lenslet_ind_y, order=order, coef=coef_row)
        x_tracks.append(xt.ravel())  # flatten the 2-D lenslet grid to a 1-D vector
        y_tracks.append(yt.ravel())
    x_tracks = np.array(x_tracks)  # shape: (n_wavelengths, n_lenslets_flat)
    y_tracks = np.array(y_tracks)  # shape: (n_wavelengths, n_lenslets_flat)

    # Mean-subtract each lenslet's track along the wavelength axis so the covariance terms are
    # centred on the trace midpoint rather than on the detector origin.
    # x_centered[i, j] = how far lenslet j's x position at wavelength i deviates from its mean x.
    x_centered = x_tracks - x_tracks.mean(axis=0)  # shape: (n_wavelengths, n_lenslets_flat)
    y_centered = y_tracks - y_tracks.mean(axis=0)

    # Per-lenslet 2×2 scatter-matrix elements, summed over the wavelength axis.
    # Sxx[j] = sum_i (x_i - x_mean)^2  for lenslet j  (variance in x across wavelengths)
    # Syy[j] = sum_i (y_i - y_mean)^2  for lenslet j  (variance in y)
    # Sxy[j] = sum_i (x_i - x_mean)(y_i - y_mean)  (cross-term; non-zero for tilted traces)
    Sxx = np.sum(x_centered**2, axis=0)   # shape: (n_lenslets_flat,)
    Syy = np.sum(y_centered**2, axis=0)
    Sxy = np.sum(x_centered * y_centered, axis=0)

    # Principal-axis angle via the closed-form 2×2 eigenvector formula.
    # For a 2×2 symmetric matrix [[Sxx, Sxy], [Sxy, Syy]], the dominant eigenvector direction is
    #   theta = 0.5 * arctan2(2*Sxy, Sxx - Syy)
    # which gives the orientation of the elongated axis of the point cloud, i.e. the trace tilt.
    # Result is in [-90°, 90°], so near-horizontal traces land close to 0°.
    clocking_angle_rad = 0.5 * np.arctan2(2.0 * Sxy, Sxx - Syy)
    clocking_angle_deg = np.degrees(clocking_angle_rad)  # shape: (n_lenslets_flat,)

    # Representative pixel position of each lenslet at the median wavelength (mid_coef from above).
    # Each scatter point is plotted at the lenslet's mid-wavelength detector position so the map
    # reflects where each trace physically sits on the sensor.
    x_mid, y_mid = transform(lenslet_ind_x, lenslet_ind_y, order=order, coef=mid_coef)
    x_mid = x_mid.ravel()  # shape: (n_lenslets_flat,) — matches clocking_angle_deg
    y_mid = y_mid.ravel()

    # Boolean mask selecting only the lenslets whose mid-wavelength position falls inside the
    # plotting bounds; used to scale the colormap symmetrically to the visible region rather than
    # to outliers at the detector edge or outside the crop window.
    clocking_bounds_mask = \
        (x_mid >= xmin_plot) & (x_mid <= xmax_plot) & \
        (y_mid >= ymin_plot) & (y_mid <= ymax_plot)
    if clocking_bounds_mask.any():
        angle_extent = np.max(np.abs(clocking_angle_deg[clocking_bounds_mask]))
    else:
        angle_extent = np.max(np.abs(clocking_angle_deg)) if clocking_angle_deg.size else 1.0
    # vmin/vmax are equal and opposite so the zero-clocking colour (white in RdBu_r) maps to 0°.
    vmin_angle, vmax_angle = -angle_extent, angle_extent

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_mid, y_mid, c=clocking_angle_deg, s=20, cmap='RdBu_r',
                         vmin=vmin_angle, vmax=vmax_angle)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Clocking angle (degrees)')
    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_aspect('equal')
    ax.set_xlabel('Detector X (pixels)')
    ax.set_ylabel('Detector Y (pixels)')
    ax.set_title('Spectral Trace Clocking Angle')
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    if output_directory is not None:
        filename = os.path.join(output_directory, 'trace_clocking_angle.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')

    #########################################################################
    # Plot spectral trace length vs lenslet by evaluating the distance between the shortest and longest wavelengths
    #########################################################################
    wavelength_min = lamsol_df[0].min()
    wavelength_max = lamsol_df[0].max()
    x_positions_low, y_positions_low = transform(lenslet_ind_x, lenslet_ind_y, order=order, coef=lamsol_df.loc[lamsol_df[0] == wavelength_min].values[0, 1:])
    x_positions_high, y_positions_high = transform(lenslet_ind_x, lenslet_ind_y, order=order, coef=lamsol_df.loc[lamsol_df[0] == wavelength_max].values[0, 1:])
    trace_lengths = np.sqrt((x_positions_high - x_positions_low)**2 + (y_positions_high - y_positions_low)**2)
    
    trace_bounds_mask = \
        (x_positions_low >= xmin_plot) & (x_positions_low <= xmax_plot) & \
        (y_positions_low >= ymin_plot) & (y_positions_low <= ymax_plot)
    vmin_trace = trace_lengths[trace_bounds_mask].min() if trace_bounds_mask.any() else None
    vmax_trace = trace_lengths[trace_bounds_mask].max() if trace_bounds_mask.any() else None


    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(x_positions_low, y_positions_low, c=trace_lengths, 
                         s=20, vmin=vmin_trace, vmax=vmax_trace)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Trace Length (pixels)')
    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_aspect('equal')
    ax.set_xlabel('Detector X (pixels)')
    ax.set_ylabel('Detector Y (pixels)')
    ax.set_title(f'Spectral Trace Length\n[{wavelength_min} - {wavelength_max} nm]')
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    if output_directory is not None:
        filename = os.path.join(output_directory, f'spectral_trace_length_map.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    #########################################################################
    # Scatter plots of PSFlet positions colored by wavelength, per region of interest
    #########################################################################
    if sensor_regions_to_inspect is None:
        sensor_regions_to_inspect = [[sensor_dimensions[0] / 2, sensor_dimensions[1] / 2]]

    half = window_size / 2
    for region in sensor_regions_to_inspect:
        cx, cy = region
        region_df = psf_table[
            (psf_table['x'] >= cx - half) & (psf_table['x'] <= cx + half) &
            (psf_table['y'] >= cy - half) & (psf_table['y'] <= cy + half)
        ]
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(region_df['x'], region_df['y'],
                             c=region_df['wavelength'], s=20, cmap='viridis')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Wavelength (nm)')
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect('equal')
        ax.set_xlabel('Detector X (pixels)')
        ax.set_ylabel('Detector Y (pixels)')
        ax.set_title(f'PSFlet Map\n(region center: {cx:.0f}, {cy:.0f})')
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        if output_directory is not None:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            filename = os.path.join(output_directory, f'wavelength_map_{cx:.0f}_{cy:.0f}.png')
            fig.savefig(filename, dpi=300, bbox_inches='tight')

    #########################################################################
    # Plot PSF centroid position vs. wavelength for user-specified lenslets
    #########################################################################
    sort_idx = np.argsort(lamsol_df[0].values)
    sorted_wavelengths = lamsol_df[0].values[sort_idx]

    x_positions_by_lenslet = {}
    y_positions_by_lenslet = {}
    for lenslet_key in lenslet_keys:
        lenslet_x, lenslet_y = lenslet_key
        flat_idx = np.where(
            (lenslet_ind_x.ravel() == lenslet_x) & (lenslet_ind_y.ravel() == lenslet_y)
        )[0]
        if len(flat_idx) == 0:
            raise ValueError(
                f"Lenslet {list(lenslet_key)} not found in lenslet grid for nlens={nlens}. "
                f"Allowed lenslet coordinates are [{-(nlens // 2) + 1}, {nlens // 2}] on each axis."
            )
        flat_idx = flat_idx[0]
        x_pos = x_tracks[:, flat_idx][sort_idx]
        y_pos = y_tracks[:, flat_idx][sort_idx]
        x_positions_by_lenslet[lenslet_key] = x_pos - x_pos[0]
        y_positions_by_lenslet[lenslet_key] = y_pos - y_pos[0]

    fig, ax_left = plt.subplots(figsize=(7, 5))
    ax_right = ax_left.twinx()

    for i, lenslet_key in enumerate(lenslet_keys):
        color = colors[i % len(colors)]
        ax_left.plot(sorted_wavelengths, x_positions_by_lenslet[lenslet_key],
                     color=color, label=f'ΔX lenslet {list(lenslet_key)}', zorder=10)
        ax_right.plot(sorted_wavelengths, y_positions_by_lenslet[lenslet_key],
                      color=color, linestyle='--', label=f'ΔY lenslet {list(lenslet_key)}', zorder=10)

    ax_left.set_xlabel('Wavelength (nm)')
    ax_left.set_ylabel('ΔX position (pixels)')
    ax_right.set_ylabel('ΔY position (pixels)')
    ax_left.tick_params(axis='y')
    ax_right.tick_params(axis='y')
    ax_left.set_title('PSFlet position vs. wavelength')
    ax_left.grid(True, alpha=0.3)

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    legend = fig.legend(left_handles + right_handles, left_labels + right_labels,
                        framealpha=1.0, loc='center left', bbox_to_anchor=(1.15, 0.5),
                        bbox_transform=ax_right.transAxes)
    legend.set_zorder(1000)
    fig.tight_layout(rect=[0, 0, 0.72, 1])
    plt.show(block=False)
    plt.pause(0.1)

    if output_directory is not None:
        filename = os.path.join(output_directory, 'psflet_position_vs_wavelength.png')
        fig.savefig(filename, dpi=300, bbox_inches='tight')

    #########################################################################
    # PSFLoc.fits visualization
    #########################################################################
    # PSFloc.fits is a sibling calibration product of lamsol.dat, written by PSFLets.savepixsol().
    # It stores, per lenslet, the detector pixel positions/wavelengths of each microspectrum along with
    # a valid-wavelength count and an on-detector validity flag. It may or may not be present in the
    # calibration directory, so skip this section gracefully if the file is absent.
    psfloc_filepath = os.path.join(os.path.dirname(lamsol_filepath), 'PSFloc.fits')
    if not os.path.isfile(psfloc_filepath):
        print(f'PSFloc.fits not found at {psfloc_filepath}; skipping PSFLoc.fits visualization.')
    else:
        # Read the 5 extensions by index (extension 0 is the PrimaryHDU with no EXTNAME), matching the
        # index-based access convention in PSFLets.loadpixsol().
        with fits.open(psfloc_filepath) as hdulist:
            psfloc_extensions = [
                ('lam_indx', hdulist[0].data, 'Wavelength (nm)'),  # 3D: wavelength at each microspectrum pixel
                ('xindx', hdulist[1].data, 'X-coordinate (pixels)'),     # 3D: dispersion-axis pixel index
                ('yindx', hdulist[2].data, 'Y-coordinate (pixels)'),     # 3D: cross-dispersion pixel position
                ('nlam', hdulist[3].data, 'Number of wavelength bins'),      # 2D: number of valid wavelengths per lenslet
                ('good', hdulist[4].data, 'On-detector validity flag'),      # 2D: on-detector validity flag
            ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle('PSFloc.fits snapshot (one slice through each extension)')
        axes_flat = axes.ravel()
        for idx, (ax, (name, data, cbar_label)) in enumerate(zip(axes_flat, psfloc_extensions)):
            if data.ndim == 3:
                # Display the middle nlens x nlens slice of the stack.
                mid_slice = data.shape[2] // 2
                slice_2d = data[:, :, mid_slice]
                title = f'Extension {idx}\n{name} (slice {mid_slice} / {data.shape[2] - 1})'
            else:
                slice_2d = data
                title = f'Extension {idx}\n{name}'
            if not any(keyword in title for keyword in ['xindx','yindx','good']):
                vmin = 0.9 * np.max(slice_2d)
            else:
                vmin = 0
            im = ax.imshow(slice_2d, origin='lower', vmin=vmin)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
            ax.set_title(title)
            ax.set_xlabel('Lenslet Index')
            ax.set_ylabel('Lenslet Index')

        # Hide the unused 6th panel (2x3 grid, only 5 extensions).
        axes_flat[5].axis('off')
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        if output_directory is not None:
            filename = os.path.join(output_directory, 'PSFloc_snapshot.png')
            fig.savefig(filename, dpi=300, bbox_inches='tight')

    #########################################################################
    # polychromekeyR<NN>.fits visualization
    #########################################################################
    # The polychrome key is a companion calibration product that stores, for each lenslet and wavelength
    # bin, the expected detector pixel position and an on-detector validity flag. It may or may not be
    # present in the calibration directory, so skip this section gracefully if absent.
    import glob as _glob
    polychrome_candidates = sorted(_glob.glob(os.path.join(os.path.dirname(lamsol_filepath), 'polychromekeyR*.fits')))
    if not polychrome_candidates:
        print(f'No polychromekeyR*.fits file found in {os.path.dirname(lamsol_filepath)}; skipping polychrome key visualization.')
    else:
        polychrome_filepath = polychrome_candidates[0]
        with fits.open(polychrome_filepath) as hdulist:
            lam_midpts   = hdulist[0].data   # 1D: wavelength bin midpoints (nm)
            xpos_stack   = hdulist[1].data   # 3D (N_lam, nlens, nlens): X pixel position per lenslet/wavelength
            ypos_stack   = hdulist[2].data   # 3D (N_lam, nlens, nlens): Y pixel position per lenslet/wavelength
            good_stack   = hdulist[3].data   # 3D (N_lam, nlens, nlens): on-detector validity flag (uint8)

        fname = os.path.basename(polychrome_filepath)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f'{fname} snapshot')

        # Extension 0 — wavelength bin midpoints as a connected scatter plot.
        ax = axes[0, 0]
        ax.plot(np.arange(len(lam_midpts)), lam_midpts, marker='o', markersize=6)
        ax.set_title('Extension 0\nlam_midpts')
        ax.set_xlabel('Bin index')
        ax.set_ylabel('Wavelength (nm)')

        # Extensions 1–3 — middle wavelength-slice of each 3D stack, displayed as 2D images.
        mid_slice_index = xpos_stack.shape[0] // 2
        image_panels = [
            (axes[0, 1], xpos_stack[mid_slice_index], 'X-coordinate (pixels)', 1),
            (axes[1, 0], ypos_stack[mid_slice_index], 'Y-coordinate (pixels)', 2),
            (axes[1, 1], good_stack[mid_slice_index], 'On-detector validity flag', 3),
        ]
        for ax, data, cbar_label, ext_idx in image_panels:
            im = ax.imshow(data, origin='lower')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
            ax.set_title(f'Extension {ext_idx}\n{cbar_label} (slice {mid_slice_index} / {xpos_stack.shape[0] - 1})')
            ax.set_xlabel('Lenslet Index')
            ax.set_ylabel('Lenslet Index')

        fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        if output_directory is not None:
            filename = os.path.join(output_directory, f'{os.path.splitext(fname)[0]}_snapshot.png')
            fig.savefig(filename, dpi=300, bbox_inches='tight')

    #########################################################################
    # Return final dispersion values for each lenslet, along with the wavelength array
    #########################################################################

    if len(lenslet_keys) == 1:
        return lamsol_df[0].values, np.array(dispersion_by_lenslet[lenslet_keys[0]])

    dispersion_by_lenslet = {
        lenslet_key: np.array(dispersion_values)
        for lenslet_key, dispersion_values in dispersion_by_lenslet.items()
    }
    return lamsol_df[0].values, dispersion_by_lenslet
