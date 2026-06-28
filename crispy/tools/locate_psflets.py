#!/usr/bin/env python

import numpy as np
from astropy.io import fits
import copy
from scipy import signal, ndimage, optimize, interpolate
import matplotlib.pyplot as plt
from crispy.tools.image import Image
import glob
import re
import os
from photutils.centroids import centroid_2dg
from photutils.detection import find_peaks

from crispy.tools.initLogger import getLogger
log = getLogger('crispy')


class PSFLets:
    """
    Helper class to deal with the PSFLets on the detector. Does most of the heavy lifting
    during the wavelength calibration step.
    """

    def __init__(self, load=False, infile=None, infiledir='.'):
        '''
        Initialize the class

        Parameters
        ----------
        load: Boolean
            Whether to load an already-existing wavelength calibration file
        infile: String
            If load is True, this is the name of the file
        infiledir: String
            If load is True, this is the directory in which the file resides
        '''
        self.xindx = None
        self.yindx = None
        self.lam_indx = None
        self.nlam = None
        self.nlam_max = None
        self.interp_arr = None
        self.order = None

        if load:
            self.loadpixsol(infile, infiledir)

    def loadpixsol(self, infile=None, infiledir='./calibrations'):
        '''
        Loads existing wavelength calibration file

        Parameters
        ----------
        infile: String
            Name of the file
        infiledir: String
            Directory in which the file resides
        '''
        if infile is None:
            infile = re.sub('//', '/', infiledir + '/PSFloc.fits')
        hdulist = fits.open(infile)

        try:
            self.lam_indx = hdulist[0].data
            self.xindx = hdulist[1].data
            self.yindx = hdulist[2].data
            self.nlam = hdulist[3].data.astype(int)
            self.good = hdulist[4].data.astype(int)
        except BaseException:
            raise RuntimeError(f"File {infile} does not appear to contain a CHARIS wavelength solution in the appropriate format.")
        self.nlam_max = np.amax(self.nlam)

    def savepixsol(self, outdir="calibrations/"):
        '''
        Saves wavelength calibration file

        Parameters
        ----------
        outdir: String
            Directory in which to put the file. The file is named PSFloc.fits and is a
            multi-extension FITS file, each extension corresponding to:
            0. the list of wavelengths at which the calibration is done
            1. a 2D ndarray with the X position of all lenslets
            2. a 2D ndarray with the Y position of all lenslets
            3. a 2D ndarray with the number of valid wavelengths for a given lenslet (some wavelengths fall outside of the detector area)

        '''
        if not os.path.isdir(outdir):
            raise IOError(f"Attempting to save pixel solution to directory {outdir}.  Directory does not exist.")
        outfile = re.sub('//', '/', outdir + '/PSFloc.fits')
        out = fits.HDUList([
            fits.PrimaryHDU(self.lam_indx),  # Primary HDU, no EXTNAME
            fits.ImageHDU(self.xindx, name='XINDX'),
            fits.ImageHDU(self.yindx, name='YINDX'),
            fits.ImageHDU(self.nlam.astype(int), name='nlam'),
            fits.ImageHDU(self.good.astype(int), name='good')
        ])
        try:
            out.writeto(outfile, overwrite=True)
        except BaseException:
            raise

    def geninterparray(self, lam, allcoef, order=3):
        '''
        Set up array to solve for best-fit polynomial fits to the
        coefficients of the wavelength solution.  These will be used
        to smooth/interpolate the wavelength solution, and
        ultimately to compute its inverse.

        Parameters
        ----------
        lam: array
            Wavelengths in nm
        allcoef: list of lists floats
            Polynomial coefficients of wavelength solution
        order: int
            Order of polynomial wavelength solution

        Notes
        -----
        Populates the attribute interp_arr in PSFLet class
        '''

        self.interp_arr = np.zeros((order + 1, allcoef.shape[1]))
        self.order = order
        log_wavelength_powers = np.ones((lam.shape[0], order + 1))  # Initialize an array of wavelength terms for fitting
        for i in range(1, order + 1):
            log_wavelength_powers[:, i] = np.log(lam)**i  # Why use the log of wavelength for fitting?
        for i in range(self.interp_arr.shape[1]):
            coef = np.linalg.lstsq(log_wavelength_powers, allcoef[:, i])[0]
            self.interp_arr[:, i] = coef

    # COMMENTED OUT: Function commented out to avoid confusion until we understand its purpose. 
    # Note from Evan: Looks like this is an unfinished version of another return_locations function that is still in use elsewhere in this module. 
    # def return_locations_short(self, coef, xindx, yindx):
    #     '''
    #     Returns the x,y detector location of a given lenslet for a given polynomial fit

    #     Parameters
    #     ----------
    #     coef: lists floats
    #         Polynomial coefficients of fit for a single wavelength
    #     xindx: int
    #         X index of lenslet in lenslet array
    #     yindx: int
    #         Y index of lenslet in lenslet array

    #     Returns
    #     -------
    #     interp_x: float
    #         X coordinate on the detector
    #     interp_y: float
    #         Y coordinate on the detector
    #     '''
    #     # TODO, where does this 'coeforder' come from? Does this parent function actually get called from anywhere? Because clearly it's going to throw an error if it does.
    #     interp_x, interp_y = transform(xindx, yindx, coeforder, coef) 
    #     return interp_x, interp_y

    def return_res(self, lam, allcoef, xindx, yindx,
                   order=3, lam1=None, lam2=None):
        '''
        Returns the spectral resolution and interpolated wavelength array

        Parameters
        ----------
        lam: float
            Wavelength in nm
        allcoef: list of lists floats
            Polynomial coefficients of wavelength solution
        xindx: int
            X index of lenslet in lenslet array
        yindx: int
            Y index of lenslet in lenslet array
        order: int
            Order of polynomial wavelength solution
        lam1: float
            Shortest wavelength in nm
        lam2: float
            Longest wavelength in nm

        Returns
        -------
        interp_lam: array
            Array of wavelengths
        R: float
            Effective spectral resolution
        '''

        if lam1 is None:
            lam1 = np.amin(lam) / 1.04
        if lam2 is None:
            lam2 = np.amax(lam) * 1.03

        interporder = order

        if self.interp_arr is None:
            self.geninterparray(lam, allcoef, order=order)

        coeforder = int(np.sqrt(allcoef.shape[1])) - 1
        nlam_for_spline = 100

        interp_lam = np.linspace(lam1, lam2, nlam_for_spline)
        dy = []
        dx = []

        for i in range(nlam_for_spline):
            coef = np.zeros((coeforder + 1) * (coeforder + 2))
            for k in range(1, interporder + 1):
                coef += k * self.interp_arr[k] * np.log(interp_lam[i])**(k - 1)
            _dx, _dy = transform(xindx, yindx, coeforder, coef)

            dx += [_dx]
            dy += [_dy]

        R = np.sqrt(np.asarray(dy)**2 + np.asarray(dx)**2)

        return interp_lam, R

    def monochrome_coef(self, lam, alllam=None, allcoef=None, order=3):
        if self.interp_arr is None:
            if alllam is None or allcoef is None:
                raise ValueError(
                    "Interpolation array has not been computed.  Must call monochrome_coef with arrays.")
            self.geninterparray(alllam, allcoef, order=order)

        coef = np.zeros(self.interp_arr[0].shape)
        for k in range(self.order + 1):
            coef += self.interp_arr[k] * np.log(lam)**k
        return coef

    def return_locations(self, lam, allcoef, xindx, yindx, order=3):
        '''
        Calculates the detector coordinates of lenslet located at `xindx`, `yindx`
        for desired wavelength `lam`

        Parameters
        ----------
        lam: float
            Wavelength in nm
        allcoef: list of lists floats
            Polynomial coefficients of wavelength solution
        xindx: int, or array of int
            X index of lenslet in lenslet array
        yindx: int, or array of int
            Y index of lenslet in lenslet array
        order: int
            Order of polynomial wavelength solution

        Returns
        -------
        interp_x: float
            X coordinate on the detector
        interp_y: float
            Y coordinate on the detector
        '''
        if len(allcoef.shape) == 1:
            coeforder = int(np.sqrt(allcoef.shape[0])) - 1
            interp_x, interp_y = transform(xindx, yindx, coeforder, allcoef)
            return interp_x, interp_y

        if self.interp_arr is None:
            self.geninterparray(lam, allcoef, order=order)

        coeforder = int(np.sqrt(allcoef.shape[1])) - 1
        if not (coeforder + 1) * (coeforder + 2) == allcoef.shape[1]:
            raise ValueError("Number of coefficients incorrect for polynomial order.")

        coef = np.zeros((coeforder + 1) * (coeforder + 2))
        for k in range(self.order + 1):
            coef += self.interp_arr[k] * np.log(lam)**k
        interp_x, interp_y = transform(xindx, yindx, coeforder, coef)

        return interp_x, interp_y

#     def return_fine_locations(self, lam, xindx, yindx, xlistarr, ylistarr):
#         '''
#         Calculates the detector coordinates of lenslet located at `xindx`, `yindx`
#         for desired wavelength `lam`, using the fine calibration method
# 
#         Parameters
#         ----------
#         lam: float
#             Wavelength in nm
#         xindx: int
#             X index of lenslet in lenslet array
#         yindx: int
#             Y index of lenslet in lenslet array
#         order: int
#             Order of polynomial wavelength solution
# 
#         Returns
#         -------
#         interp_x: float
#             X coordinate on the detector
#         interp_y: float
#             Y coordinate on the detector
#         '''
#         interp_x, interp_y = fine_transform(xindx, yindx, lam, xlistarr, ylistarr)

    def genpixsol(
            self,
            par,
            lam,
            allcoef,
            order=3,
            lam1=None,
            lam2=None,
            borderpix=4,
            finexy=None,
            plot_wavelength_map=False):
        '''
        Calculates the wavelength at the center of each pixel within a microspectrum for all lenslets.

        Parameters
        ----------
        par : object
            Object containing parameters like nlens (number of lenslets) and npix (number of pixels).
        lam : array
            Wavelengths in nm for which we have calibration data.
        allcoef : list of floats
            List describing the polynomial coefficients that best fit the lenslets to pixel position,
            for all wavelengths.
        order : int, optional
            Order of the polynomial fit. Default is 3.
        lam1 : float, optional
            Lowest wavelength in nm to consider. If None, uses min(lam).
        lam2 : float, optional
            Highest wavelength in nm to consider. If None, uses max(lam).
        borderpix : int, optional
            Number of pixels to exclude at the edges of the detector. Default is 4.
        finexy : tuple, optional
            Fine adjustments to x and y positions and SNR threshold.
        plot_wavelength_map : boolean
            Display a plot that helps relate the detector image to particular lenslets and wavelenghts. Helpful for troubleshooting. 
           
        Returns
        -------
        None
            But populates the following attributes of the PSFlet class:
            - xindx: array of integer pixel indices along dispersion axis for each lenslet
            - yindx: array of floats indicating the cross-dispersion axis for each lenslet
            - nlam: number of valid wavelengths for each lenslet
            - lam_indx: wavelength index that corresponds to integer pixel indices
            - nlam_max: maximum number of wavelengths for any lenslet
            - good: boolean array indicating valid lenslets

        Notes
        -----
        This function is a crucial part of the wavelength calibration process,
        converting from lenslet and wavelength space to detector pixel space.
        '''

        # Set wavelength limits if not provided
        if lam1 is None:
            lam1 = np.amin(lam)
        if lam2 is None:
            lam2 = np.amax(lam)
        interporder = order

        # Generate interpolation array of size [order+1, number of coefficients] if not already done
        if self.interp_arr is None:
            # Create interpolation array to smooth/interpolate wavelength solution
            self.geninterparray(lam, allcoef, order=order)

        # Verify the number of coefficients matches the polynomial order
        coeforder = int(np.sqrt(allcoef.shape[1])) - 1
        if not (coeforder + 1) * (coeforder + 2) == allcoef.shape[1]:
            raise ValueError("Number of coefficients incorrect for polynomial order.")

        # Create grid of lenslet indices
        xindx = np.arange(-par.nlens // 2, par.nlens // 2)
        xindx, yindx = np.meshgrid(xindx, xindx)

        # Set up interpolation grid
        nlam_for_spline = 100
        interp_x = np.zeros(tuple([nlam_for_spline] + list(xindx.shape)))
        interp_y = np.zeros(interp_x.shape)
        interp_lam = np.linspace(lam1, lam2, nlam_for_spline)

        # Calculate x and y positions in pixel-space for each lenslet at each interpolated wavelength
        for i in range(nlam_for_spline):
            coef = np.zeros((coeforder + 1) * (coeforder + 2))
            for k in range(interporder + 1):
                coef += self.interp_arr[k] * np.log(interp_lam[i])**k
            interp_x[i], interp_y[i] = transform(xindx, yindx, coeforder, coef)

        # Apply fine adjustments if provided
        if finexy is not None:
            interp_x += finexy[0]
            interp_y += finexy[1]

        # Initialize output arrays
        x = np.zeros(tuple(list(xindx.shape) + [1000]))  # NOTE, Why is x initialized to this size? Where does the 1000 come from? Something to do with determining the maximum number of wavelengths per lenslet later?
        y = np.zeros(x.shape)
        nlam = np.zeros(xindx.shape, np.int32)
        lam_out = np.zeros(y.shape)
        good = np.ones(xindx.shape)  # An array for tracking whether or not any interpolated wavelengths from this lenslet fall outside the detector

        # Apply SNR threshold if fine adjustments are provided
        if finexy is not None:
            good *= finexy[2] > 10

        # Process each lenslet
        for ix in range(xindx.shape[0]):
            for iy in range(xindx.shape[1]):
                # NOTE from Evan Bray (2025-09-02): I think the following line was a bug and that pix_y and pix_x are reversed, 
                # but it ends up working out in the end because when we assign values to self.xindx we use the y-spline.
                pix_y = interp_x[:, ix, iy]
                pix_x = interp_y[:, ix, iy]

                # Check if lenslet falls within valid detector area
                if (np.any(pix_x < borderpix) or np.any(pix_x > par.npix - borderpix) or
                    np.any(pix_y < borderpix) or np.any(pix_y > par.npix - borderpix)):
                    good[ix, iy] = 0
                    continue

                # Handle reversed wavelength order
                if pix_y[-1] < pix_y[0]:
                    good[ix, iy] = 0
                else:
                    try:
                        # Create spline representation of wavelength vs. y-pixel position
                        tck_y = interpolate.splrep(pix_y, interp_lam, k=3, s=0)
                    except Exception:
                        good[ix, iy] = 0
                        log.error('Error on wavelength calibration for lenslet ({:})'.format((ix, iy)))

                if good[ix, iy]:
                    # Determine y-pixel range for this lenslet
                    y1, y2 = [int(np.amin(pix_y)) + 1, int(np.amax(pix_y))]
                    # Create spline representation of x-position vs. wavelength
                    tck_x = interpolate.splrep(interp_lam, pix_x, k=1, s=0)

                    nlam[ix, iy] = y2 - y1 + 1
                    y[ix, iy, :nlam[ix, iy]] = np.arange(y1, y2 + 1)
                    # Evaluate wavelengths at integer pixel values
                    lam_out[ix, iy, :nlam[ix, iy]] = interpolate.splev(y[ix, iy, :nlam[ix, iy]], tck_y) 
                    x[ix, iy, :nlam[ix, iy]] = interpolate.splev(lam_out[ix, iy, :nlam[ix, iy]], tck_x)

        # Determine maximum number of wavelengths for any lenslet
        # nlam_max = np.count_nonzero(y,axis=2).max() # Proposed, more elegant improvement to the method below
        for nlam_max in range(x.shape[-1]):
            if np.all(y[:, :, nlam_max] == 0):
                break

        # Populate class attributes with computed values
        self.xindx = y[:, :, :nlam_max]  # array of integer pixel indices along dispersion
        self.yindx = x[:, :, :nlam_max]  # array of floats indicating the cross. disp. axis
        self.nlam = nlam
        self.lam_indx = lam_out[:, :, :nlam_max]  # wavelengths at int. pixel indices
        self.nlam_max = np.amax(nlam)
        self.good = good
       
        # Make a plot of pixel wavelengths vs pixel position
        if plot_wavelength_map:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_aspect('equal')

            # Flatten the arrays and create a mask for non-zero wavelengths
            xind_temp = np.arange(self.xindx.shape[1])
            yind_temp = np.arange(self.xindx.shape[0])
            xind_temp, yind_temp = np.meshgrid(xind_temp, yind_temp)
            x = self.xindx.flatten()
            y = self.yindx.flatten()
            wavelengths = self.lam_indx.flatten()
            mask = wavelengths != 0

            # Create scatter plot with masked data
            scatter = ax.scatter(x[mask], y[mask], c=wavelengths[mask], s=5, cmap='viridis')

            # Add annotations with the index of each lenslet overlaid with each microspectrum
            # Warning, uncommenting the following line will make the plot slower to render
            # for i in range(par.nlens):
            #     for j in range(par.nlens):
            #         # print(i,j)
            #         xpos, ypos = [self.xindx[i, j, 0], self.yindx[i, j, 0]]
            #         if xpos == 0 or ypos == 0:
            #             continue
            #         ax.annotate(f'({i},{j})', xy=(xpos-2, ypos+1), color='black', fontsize=12)
            ax.set_xlim(100, 200)
            ax.set_ylim(100, 200)
            ax.set_xlabel('pixels')
            ax.set_ylabel('pixels')
            ax.set_title('Lenslet + Wavelength Map')
            fig.colorbar(scatter, ax=ax, label='Wavelength (nm)')
            fig.tight_layout()
            plt.show()


def initcoef(order, scale, phi, x0=0, y0=0):
    """
    Generate a set of null-transformation coefficients, but fill out the elements 
    corresponding to 0th order translation + rotation. The higher order correction terms
    will get filled out at a later step. 

    Parameters
    ----------
    order: int
        The polynomial order of the grid distortion
    scale: float
        The linear separation in pixels of the PSFlets.
    phi:   float
        The pitch angle of the lenslets.
    x0:    float
        x offset to apply to the central pixel. Default 0
    y0:    float
        y offset to apply to the central pixel. Default 0

    Returns
    -------
    coef: list of floats
        A list of length (order+1)*(order+2) to be optimized.

    Notes
    -----
    The list of coefficients has space for a polynomial fit of the
    input order (i.e., for order 3, up to terms like x**3 and x**2*y,
    but not x**3*y).  It is all zeros in the output apart from the
    rotation matrix given by scale and phi.
    """

    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except BaseException:
        raise ValueError("Polynomial order must be integer")

    n = (order + 1) * (order + 2)
    coef = np.zeros((n))

    coef[0] = x0
    coef[1] = scale * np.cos(phi)
    coef[order + 1] = -scale * np.sin(phi)
    coef[n // 2] = y0
    coef[n // 2 + 1] = scale * np.sin(phi)
    coef[n // 2 + order + 1] = scale * np.cos(phi)

    return list(coef)


def transform(x, y, order, coef):
    """
    Apply the coefficients given to transform the coordinates using
    a polynomial.

    Parameters
    ----------
    x:     ndarray
        Rectilinear grid
    y:     ndarray of floats
        Rectilinear grid
    order: int
        Order of the polynomial fit
    coef:  list of floats
        List of the coefficients.  Must match the length required by order = (order+1)*(order+2).  
        For example, coefficient terms are arranged as follows for the case when order=3:
        [element of coef array]: [term in polynomial]
        (Terms for _x)
        0: x^0 * y^0
        1: x^0 * y^1
        2: x^0 * y^2
        3: x^0 * y^3
        4: x^1 * y^0
        5: x^1 * y^1
        6: x^1 * y^2
        7: x^2 * y^0
        8: x^2 * y^1
        9: x^3 * y^0
        (Terms for _y)
        10: x^0 * y^0
        11: x^0 * y^1
        12: x^0 * y^2
        13: x^0 * y^3
        14: x^1 * y^0
        15: x^1 * y^1
        16: x^1 * y^2
        17: x^2 * y^0
        18: x^2 * y^1
        19: x^3 * y^0

    Returns
    -------
    _x:    ndarray
        Transformed coordinates
    _y:    ndarray
        Transformed coordinates

    """

    try:
        if not len(coef) == (order + 1) * (order + 2):
            raise ValueError(
                "Number of coefficients incorrect for polynomial order.")
    except BaseException:
        raise AttributeError("order must be integer, coef should be a list.")

    try:
        if not order == int(order):
            raise ValueError("Polynomial order must be integer")
        else:
            if order < 1 or order > 5:
                raise ValueError("Polynomial order must be >0, <=5")
    except BaseException:
        raise ValueError("Polynomial order must be integer")

    _x = np.zeros(np.asarray(x).shape)
    _y = np.zeros(np.asarray(y).shape)

    # Calculating the new _x,_y coordinates is done in this way (as opposed to more traditional matrix multiplication)
    # because of how the coefficient matrix is arranged as a 1D array. 
    i = 0
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _x += coef[i] * x**ix * y**iy
            i += 1
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            _y += coef[i] * x**ix * y**iy
            i += 1

    return [_x, _y]


def revealCoefs(coef, order):

    i = 0
    s = 'i and j are the integer coordinates of the lenslets in the array\n'
    s += 'X coordinates:\n'
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            s += '{:} * i^{:} j^{:} + \n'.format(coef[i], ix, iy)
            i += 1
    s += 'Y coordinates:\n'
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            s += '{:} * i^{:} j^{:} + \n'.format(coef[i], ix, iy)
            i += 1
    return s


def fine_transform(lam, x, y, reflam, xlistarr, ylistarr):
    """
    Apply the coefficients given to transform the coordinates using
    a polynomial.

    Parameters
    ----------
    lam: float or 1D ndarray
        Desired wavelength (or array of wavelength)
    x:     ndarray
        Rectilinear grid
    y:     ndarray
        Rectilinear grid
    reflam: float or 1D ndarray
        Reference wavelength array at which xlistarr and ylistarr were computed
    xlistarr:     ndarray
        Centroid coordinates
    ylistarr:     ndarray
        Centroid coordinates

    Returns
    -------
    _x:    ndarray
        Transformed coordinates
    _y:    ndarray
        Transformed coordinates

    """

    if hasattr(lam, "__len__"):
        _x = np.zeros((len(lam), x.shape[0], x.shape[1]))
        _y = np.zeros((len(lam), y.shape[0], y.shape[1]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tck = interpolate.splrep(reflam, xlistarr[:, i, j])
                _x[:, i, j] = interpolate.splev(lam, tck, der=0)
                tck = interpolate.splrep(reflam, ylistarr[:, i, j])
                _y[:, i, j] = interpolate.splev(lam, tck, der=0)

    else:
        _x = np.zeros((x.shape[0], x.shape[1]))
        _y = np.zeros((y.shape[0], y.shape[1]))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tck = interpolate.splrep(reflam, xlistarr[:, i, j])
                _x[i, j] = interpolate.splev(lam, tck, der=0)
                tck = interpolate.splrep(reflam, ylistarr[:, i, j])
                _y[i, j] = interpolate.splev(lam, tck, der=0)

    return [_x, _y]


def new_transform(x, y, order, coef):
    """
    Apply the coefficients given to transform the coordinates using
    a polynomial.

    Parameters
    ----------
    x:     ndarray
        Rectilinear grid
    y:     ndarray of floats
        Rectilinear grid
    order: int
        Order of the polynomial fit
    coef:  list of floats
        List of the coefficients.  Must match the length required by
        order = (order+1)*(order+2)

    Returns
    -------
    _x:    ndarray
        Transformed coordinates
    _y:    ndarray
        Transformed coordinates

    """
    Xlist = []
    Ylist = []
    # i = 0
    for ix in range(order + 1):
        for iy in range(order - ix + 1):
            Xlist.append(x**ix * y**iy)
            Ylist.append(x**ix * y**iy)

    Xlist = np.array(Xlist)
    Ylist = np.array(Ylist)
    ncoefs = (order + 1) * (order + 2) // 2
    Xcoefs = coef[:ncoefs]
    Ycoefs = coef[-ncoefs:]
    X = np.dot(Xlist.T, Xcoefs)
    Y = np.dot(Ylist.T, Ycoefs)
    return X, Y


def corrval(coef, x, y, input_image, order, trimfrac=0.1, show_plots=False, low_orders_only=False):
    """
    Given an array of (x,y) coordinates representing the PSFLet centers, 
    determine the flux at each corresponding location in the input image. 
    The negative sum of these fluxes shall represent the "correlation score" for this image, 
    which we seek to minimize (i.e. a large negative number). When this is done, we know that
    we have found an optimal set of transformation coefficients. 

    Parameters
    ----------
    coef:     list of floats
        coefficients for polynomial transformation
    x: ndarray
        coordinates of lenslets
    y: ndarray
        coordinates of lenslets
    input_image: ndarray
        image from which flux values will be extracted
    order: int
        order of the polynomial fit
    trimfrac: float
        fraction of outliers (high & low combined) to trim in the interest of removing outliers. 
        Default 0.1 (5% trimmed on the high end, 5% on the low end)
    show_plots: bool
        If True, displays a plot of the input_image with _x and _y coordinates overlaid
    low_orders_only: bool
        If True, all coefficients beyond 0th- and 1st-order terms (indices 3+) are zeroed
        out before evaluating the transform. Only the constant and linear terms (indices
        0, 1, 2) are used. Default is False.

    Returns
    -------
    score:    float
        Negative sum of PSFlet fluxes, to be minimized
    """

    if low_orders_only:
        coef = np.array(coef, dtype=float)
        half = (order + 1) * (order + 2) // 2
        keep = [0, 1, order + 1, half, half + 1, half + order + 1]
        mask = np.ones(len(coef), dtype=bool)
        mask[keep] = False
        coef[mask] = 0.0

    #################################################################
    # Use np.nan for lenslet coordinates outside the FOV,
    # discard these from the calculation before trimming.
    #################################################################

    _x, _y = transform(x, y, order, coef)
    vals = ndimage.map_coordinates(input_image, [_y, _x], mode='constant',
                                   cval=np.nan, prefilter=False)
    vals_ok = vals[np.where(np.isfinite(vals))]

    if trimfrac > 0.0:
        iclip = int(vals_ok.shape[0] * trimfrac // 2)
        vals_sorted = np.sort(vals_ok)
        score = -1 * np.sum(vals_sorted[iclip:-iclip])
    else:
        vals_sorted = np.sort(vals_ok)
        score = -1 * np.sum(vals_sorted)

    if show_plots:
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(input_image, cmap='viridis')
        ax.scatter(_x, _y, c='r', s=1)
        ax.set_title(f"Coefficients:\n{[f'{c:.1f}' for c in coef]}", fontsize=10)
        # ax.set_xlim(0, input_image.shape[1])
        # ax.set_ylim(0, input_image.shape[0])
        # Add a margin of 1%
        ax.margins(0.01)
        fig.colorbar(im)
        # Add score textbox
        props = dict(boxstyle='round', facecolor='white', alpha=1.0)
        ax.text(0.95, 0.95, f'Score: {score:.2f}', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        fig.tight_layout()
        plt.show(block=False)

    return score


# COMMENTED OUT: Function unused in repository, commented out as requested
# def corrvalsum(coef, x, y, filtered, order, trimfrac=0.1, gsize=2):
#     # TODO, is this function used anywhere in this repo? If not, comment it out. 
#     # TODO, add doscring
#     _x, _y = transform(x, y, order, coef)
#     ydim, xdim = filtered.shape
#     s = 0.0
#     ry = np.reshape(_y, -1)
#     rx = np.reshape(_x, -1)
#     for i in range(len(ry)):
#         yi = ry[i]
#         xi = rx[i]
#         xmin = int(xi) - gsize
#         xmax = xmin + 2 * gsize
#         ymin = int(yi) - gsize
#         ymax = ymin + 2 * gsize
#         if ymin > 2 * gsize and xmin > 2 * gsize and xmax < xdim - 2 * gsize and ymax < ydim - 2 * gsize:
#             # dx = xi - int(xi)
#             # dy = yi - int(yi)
#             # s+=np.sum(simplepsf(size=2*gsize,fwhm=fwhm,offx=dx,offy=dy)*filtered[ymin:ymax,xmin:xmax])
#             # s+=np.sum(gausspsf(size=2*gsize,fwhm=fwhm,offx=dx,offy=dy)*filtered[ymin:ymax,xmin:xmax])
#             s += np.sum(filtered[ymin:ymax, xmin:xmax])
#     return -s


def optimize_coef_from_image(unfiltered, coef, lenslet_ind_x, lenslet_ind_y,
                             polyorder, scale, trimfrac,
                             image_fractions=[0.1, 0.3, 0.7], show_plots=False):
    """
    Optimize the PSFlet location transformation coefficients by fitting to a
    sequence of increasingly large centered crops of the full image.

    This avoids a single large jump from a small initial fit (e.g. a central
    2x2 PSFlet fit) straight to the full-size image. Instead, each crop is fit
    in turn, with the result of one crop seeding the next, so every L-BFGS-B
    call starts from a nearly-correct guess. All fits use low_orders_only=False.

    Coordinates are handled relative to the center of each crop. For the first
    (smallest) crop, the translation terms (x0, y0) are established by rastering
    a grid of offsets and keeping the best correlation score. When stepping up
    to a larger crop, the translation terms are shifted by the difference in
    half-crop sizes. After the final crop, the translation terms are converted
    to full-image coordinates before returning.

    Parameters
    ----------
    unfiltered: ndarray
        Full-size image (Gaussian-convolved but not spline-filtered). Each crop
        is spline-filtered here, since corrval uses prefilter=False.
    coef: list of floats
        Initial guess of the coefficients (typically from a central 2x2 fit,
        providing good scale/rotation/shear terms).
    lenslet_ind_x: ndarray
        Full grid of lenslet x indices.
    lenslet_ind_y: ndarray
        Full grid of lenslet y indices.
    polyorder: int
        Order of the polynomial coordinate transformation.
    scale: float
        Scale factor for the PSFlet grid, used for raster range and bounds.
    trimfrac: float
        Fraction of lenslet outliers to trim in corrval.
    image_fractions: list of floats
        Ascending fractional sizes of the full image to fit to in sequence.
        Default [0.1, 0.3, 0.7].
    show_plots: bool
        If True, display the crop with the overlaid PSFlet scatterplot after
        each fit in its own figure. Default False.

    Returns
    -------
    coef: ndarray
        Best-fit polynomial coefficients, with translation terms expressed in
        full-image coordinates.
    """
    ydim, xdim = unfiltered.shape
    x0_term = 0
    y0_term = (polyorder + 1) * (polyorder + 2) // 2
    coef = copy.deepcopy(coef)

    nlens_full = lenslet_ind_x.shape[0]
    prev_w, prev_h = None, None
    for i,frac in enumerate(image_fractions):
        w = int(frac * xdim)
        h = int(frac * ydim)
        cropped_image = ndimage.interpolation.spline_filter(
            unfiltered[(ydim // 2 - h // 2):(ydim // 2 + h // 2),
                       (xdim // 2 - w // 2):(xdim // 2 + w // 2)])

        # Use the central fraction of the lenslet grid to match the image crop. This saves fitting time and minimizes odds of finding redundant solutions.
        nlens_sub = int(frac * nlens_full)
        lc = nlens_full // 2
        lenslet_ind_x_sub = lenslet_ind_x[lc - nlens_sub // 2:lc + nlens_sub // 2,
                                           lc - nlens_sub // 2:lc + nlens_sub // 2]
        lenslet_ind_y_sub = lenslet_ind_y[lc - nlens_sub // 2:lc + nlens_sub // 2,
                                           lc - nlens_sub // 2:lc + nlens_sub // 2]

        if prev_w is None:
            # First (smallest) cropped_image: raster a grid of translation offsets to
            # establish a robust x0/y0 around the crop center.
            log.info("Rastering through translation coefficients on initial image fraction")
            correlation_score_best = 0
            coef_current_best = copy.deepcopy(coef)
            for ix in np.arange(-(scale + 1) // 2, (scale + 1) // 2, 0.5):
                for iy in np.arange(-(scale + 2) // 2, (scale + 2) // 2, 0.5):
                    coef[x0_term] = ix + (w / 2)
                    coef[y0_term] = iy + (h / 2)
                    correlation_score_current = corrval(coef, lenslet_ind_x_sub, lenslet_ind_y_sub,
                                                         cropped_image, polyorder, trimfrac, show_plots=False)
                    if correlation_score_current < correlation_score_best:
                        correlation_score_best = correlation_score_current
                        coef_current_best = copy.deepcopy(coef)
            coef = coef_current_best
        else:
            # Stepping up to a larger crop: shift translation terms by the
            # difference in half-crop sizes.
            coef[x0_term] += (w - prev_w) / 2
            coef[y0_term] += (h - prev_h) / 2

        # Constrain the translation terms to +/-scale of the current value.
        bounds = [(None, None)] * len(coef)
        bounds[x0_term] = (coef[x0_term] - scale, coef[x0_term] + scale)
        bounds[y0_term] = (coef[y0_term] - scale, coef[y0_term] + scale)
        log.info(f'Fitting coefficients for the central {frac}/1.0 of the image')
        res = optimize.minimize(corrval, coef, args=(lenslet_ind_x_sub, lenslet_ind_y_sub,
                                cropped_image, polyorder, trimfrac, False, False),
                                method='L-BFGS-B', bounds=bounds)
        coef = res.x.copy()

        if show_plots:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cropped_image, origin='lower', cmap='viridis')
            _x, _y = transform(lenslet_ind_x_sub, lenslet_ind_y_sub, polyorder, coef)
            ax.scatter(_x, _y, s=5, c='r')
            ax.set_title(f'PSFlet Locations After Fit on Central {frac}/1.0 of Image')
            ax.set_xlim(0, cropped_image.shape[1])
            ax.set_ylim(0, cropped_image.shape[0])
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.show(block=False)
            plt.pause(0.1)

        prev_w, prev_h = w, h

    # Convert the translation terms from final-crop coordinates to full-image coordinates.
    coef[x0_term] += xdim / 2 - prev_w / 2
    coef[y0_term] += ydim / 2 - prev_h / 2

    return coef


def locatePSFlets(inImage, mask, polyorder=2, sig=0.7, coef=None, trimfrac=0.1,
                  phi=np.arctan2(1.926, -1), scale=15.02, nlens=108, finesearch=3,
                  show_plots=False):
    """
    function locatePSFlets takes an Image class, assumed to be a
    monochromatic grid of spots with read noise and shot noise, and
    returns the estimated positions of the spot centroids.  This is
    designed to constrain the domain of the PSFlet fitting later in
    the pipeline.

    Parameters
    ----------
    inImage: Image class
        Assumed to be a monochromatic grid of spots
    polyorder: float
        order of the polynomial coordinate transformation. Default 2.
    sig: float
        standard deviation of convolving Gaussian used
        for estimating the grid of centroids.  Should be close
        to the true value for the PSFlet spots.  Default 0.7.
    coef: list
        initial guess of the coefficients of polynomial coordinate transformation
    trimfrac: float
        fraction of lenslet outliers (high & low
        combined) to trim in the minimization.  Default 0.1
        (5% trimmed on the high end, 5% on the low end)
    mask: ndarray
        Mask array for the image
    phi: float
        Rotation angle for the PSFlet grid. Default np.arctan2(1.926, -1)
    scale: float
        Scale factor for the PSFlet grid. Default 15.02
    nlens: int
        Number of lenslets. Default 108
    finesearch: int
        Fine search parameter. Default 3 

    Returns
    -------
    x: 2D ndarray
        Estimated spot centroids in x.
    y: 2D ndarray
        Estimated spot centroids in y.
    good:2D boolean ndarray
        True for lenslets with spots inside the detector footprint
    coef: list of floats
        List of best-fit polynomial coefficients

    Notes
    -----
    The coefficients, if not supplied, are initially set to the
    known pitch angle and scale.  A loop then does a quick check to find
    reasonable offsets in x and y.  With all of the first-order polynomial
    coefficients set, the optimizer refines these and the higher-order
    coefficients.  This routine seems to be relatively robust down to
    per-lenslet signal-to-noise ratios of order unity (or even a little
    less).

    """

    #############################################################
    # Convolve the image with a Gaussian, apply a filter, then centroid on the PSFLets.
    #############################################################

    kernel_x = np.arange(-1 * int(3 * sig + 1), int(3 * sig + 1) + 1)
    kernel_x, kernel_y = np.meshgrid(kernel_x, kernel_x)
    gaussian = np.exp(-(kernel_x**2 + kernel_y**2) / (2 * sig**2))

#     if mask is None:
#         unfiltered = signal.convolve2d(inImage.data, gaussian, mode='same')
#     else:
#         unfiltered = signal.convolve2d(
#             inImage.data * inImage.ivar, gaussian, mode='same')
#         unfiltered /= signal.convolve2d(inImage.ivar,
#                                         gaussian, mode='same') + 1e-10
    unfiltered = signal.convolve2d(inImage.data * mask, gaussian, mode='same')

    filtered = ndimage.interpolation.spline_filter(unfiltered)

    #############################################################
    # lenslet_ind_x/y: Grid of lenslet IDs, Lenslet (0, 0) will be referred to as the center.
    #############################################################

    # gridfrac = 10
    ydim, xdim = inImage.data.shape
    # x = np.arange(-(ydim//gridfrac), ydim//gridfrac + 1)
    lenslet_ind_x = np.arange(-nlens // 2, nlens // 2) + 1
    lenslet_ind_x, lenslet_ind_y = np.meshgrid(lenslet_ind_x, lenslet_ind_x)

    #############################################################
    # Set up polynomial coefficients, convert from lenslet
    # coordinates to coordinates on the detector array.
    # Then optimize the coefficients.
    # We want to start with a decent guess, so we use a grid of
    # offsets.  Seems to be robust down to SNR/PSFlet ~ 1
    # Create slice indices for subimages to perform the intial
    # fits on. The new dimensionality in both x and y is 2*subsize
    #############################################################

    # Determine PSF location coefficients if they were not given
    if coef is None:

        log.info("Initializing PSFlet location transformation coefficients")
        correlation_score_best = 0  # Initialize best correlation value
        num_psflets_for_subimage = int(nlens // 2.5)   # Approximately how many PSFlets do we want on each side in the subimage?
        subshape = int(scale * num_psflets_for_subimage)  # Define size of subimage for initial optimization. 
        subfiltered = ndimage.interpolation.spline_filter(unfiltered[(ydim // 2 - subshape // 2 - 1):ydim // 2 + subshape // 2,
                                                        (xdim // 2 - subshape // 2 - 1):xdim // 2 + subshape // 2])


        #########################################################
        # Fit central 2x2 PSFlets
        #########################################################
        
        # Locate PSFs in the subfiltered image with the find_peaks function and identify the one closest to the center
        peaks = find_peaks(subfiltered, threshold=np.max(subfiltered)/3)
        subcenter_x, subcenter_y = [subfiltered.shape[1] / 2, subfiltered.shape[0] / 2]
        distances = np.sqrt((peaks['x_peak'].value - subcenter_x)**2 + (peaks['y_peak'].value - subcenter_y)**2)
        closest_peak_index = np.argmin(distances)
        peak_x = peaks['x_peak'].value[closest_peak_index]
        peak_y = peaks['y_peak'].value[closest_peak_index]
        
        # Define a new windowed image that contains this PSFlet, along with the one to its right, above, and above-right.
        # So this windowed image should contain precisely 4 PSFlets. We will use this image to estimate the coefficients that affect shear.
        border_offset = scale // 2
        xmin = np.min([peak_x - scale * np.sin(phi) , peak_x]) - border_offset
        xmax = peak_x + np.max([scale*(np.cos(phi) - np.sin(phi)),scale*np.cos(phi)]) + border_offset
        y_min = np.min([peak_y + scale * np.sin(phi) , peak_y]) - border_offset
        y_max = peak_y + np.max([scale*(np.cos(phi)+np.sin(phi)),scale*np.cos(phi)]) + border_offset
        subfiltered_sub = subfiltered[int(y_min):int(y_max), int(xmin):int(xmax)]
        peaks = find_peaks(subfiltered_sub, threshold = np.max(subfiltered_sub) / 2, centroid_func=centroid_2dg)
        if show_plots:
            plt.subplots();plt.imshow(subfiltered_sub,origin='lower');plt.show()
            
        # Make a grid of 2x2 lenslets and use the corrval() function to optimize the translation coefficients for the PSFlet locations in subfiltered_sub
        lenslet_ind_x_temp = np.arange(2)
        lenslet_ind_x_temp, lenslet_ind_y_temp = np.meshgrid(lenslet_ind_x_temp, lenslet_ind_x_temp)  # Re-using the x-array for this operation since the lenslet array is square
        coef = initcoef(order=polyorder, x0=peaks['x_peak'].min(), y0=peaks['y_peak'][peaks['x_peak'].argmin()], scale=scale, phi=phi)  # Define an initial set of coefficients with a reasonable guess for x0/y0
        res = optimize.minimize(corrval, coef, args=(lenslet_ind_x_temp, lenslet_ind_y_temp,
                subfiltered_sub, polyorder, trimfrac, False, True), method='Powell')
        coef_optimized = res.x.copy()
        
        # Display a plot of the PSFlet locations after initial optimization, if desired
        show_plots=True
        if show_plots:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.imshow(subfiltered_sub, origin='lower', cmap='viridis')
            _x, _y = transform(lenslet_ind_x_temp, lenslet_ind_y_temp, polyorder, coef_optimized)
            ax.scatter(_x, _y, s=10, c='r')
            ax.set_title(f'PSFlet Locations After Initial \nShear Optimization')
            plt.show(block=False)
            plt.pause(0.1)
        
        ###############################################
        # Fit subfiltered image
        ###############################################
        
        # Fit the coefficients on a sequence of increasingly large image fractions,
        # stepping up gradually rather than jumping straight to the full image.
        log.info("Optimizing PSFlet location transformation coefficients over image fractions for frame " + inImage.filename)
        coef_optimized = optimize_coef_from_image(
            unfiltered, coef_optimized, lenslet_ind_x, lenslet_ind_y,
            polyorder, scale, trimfrac,
            image_fractions=[0.1, 0.3, 0.5, 0.7, 0.9], show_plots=show_plots)

        log.info('Array origin: {:}'.format((coef_optimized[0], coef_optimized[(polyorder + 1) * (polyorder + 2) // 2])))

    #############################################################
    # If we have coefficients from last iteration, assume that we
    # are now at a slightly higher wavelength, so try out offsets
    # that are slightly to the right to get a good initial guess.
    #############################################################

    else:
        log.info("Initializing transformation coefficients with previous values")
        correlation_score_best = 0
        coef_baseline = list(coef[:])  # Save a copy of the starting input coefficients as our baseline

        for ix in np.arange(-finesearch, finesearch, 0.2):
            for iy in np.arange(-finesearch, finesearch, 0.2):
                coef = coef_baseline[:]  # Make a temporary copy of the coefficient array to work with
                coef[0] += ix
                coef[(polyorder + 1) * (polyorder + 2) // 2] += iy

                correlation_score_current = corrval(coef, x, y, filtered, polyorder, trimfrac)
                if correlation_score_current < correlation_score_best:
                    correlation_score_best = correlation_score_current
                    coef_current_best = copy.deepcopy(coef)
        coef_optimized = coef_current_best

    # Now perform the minimiation routine on the full image. 
    log.info("Performing final optimization of PSFlet location transformation coefficients for frame " + inImage.filename)
    # Set up bounds for translation coefficients (x0 and y0) to constrain them ±scale from current value
    half_coef = (polyorder + 1) * (polyorder + 2) // 2
    bounds = [(None, None)] * len(coef_optimized)
    bounds[0] = (coef_optimized[0] - scale, coef_optimized[0] + scale)  # x0: constrain to ±scale
    bounds[half_coef] = (coef_optimized[half_coef] - scale, coef_optimized[half_coef] + scale)  # y0: constrain to ±scale
    res = optimize.minimize(corrval, coef_optimized, 
                            args=(lenslet_ind_x, lenslet_ind_x, filtered, polyorder, trimfrac, False, False), method='L-BFGS-B', bounds=bounds)

    coef_optimized = res.x
    log.info(f'Lenslet array origin (pixels): {(coef_optimized[0], coef_optimized[(polyorder + 1) * (polyorder + 2) // 2])}')

    if not res.success:
        log.info("WARNING: Optimizing PSFlet location transformation coefficients may have failed for frame " + inImage.filename)
    _x, _y = transform(lenslet_ind_x, lenslet_ind_x, polyorder, coef_optimized)

    # Display a plot of the PSFlet locations after final optimization, if desired
    if show_plots:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.imshow(unfiltered, origin='lower', cmap='viridis')
        ax.scatter(_x, _y, s=10, c='r')
        ax.set_xlim(0, xdim)
        ax.set_ylim(0, ydim)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title('PSFlet Locations After Full Optimization')
        plt.show(block=False)
        plt.pause(0.1)

    # Create an array to describe whether or not each PSFlet lies within the detector
    good_psflets = (_x > 5) * (_x < xdim - 5) * (_y > 5) * (_y < ydim - 5)

    return [_x, _y, good_psflets, coef_optimized]
