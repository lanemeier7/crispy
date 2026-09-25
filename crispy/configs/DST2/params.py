#!/usr/bin/env python

import os
import numpy as np
from numpy import sqrt, arcsin
from astropy.io import fits


class Params(object):

    def __init__(self, codeRoot='../'):
        '''
        Main class containing all the sim parameters
        '''

        self.saveRotatedInput = False
        self.saveLensletPlane = False
        self.saveDetector = True
        self.codeRoot = codeRoot
        self.prefix = self.codeRoot + '/ReferenceFiles'
        self.exportDir = self.codeRoot + '/SimResults'
        self.unitTestsOutputs = self.codeRoot + '/unitTestsOutputs'
        self.wavecalDir = self.prefix + '/Calibra_20190128/'
        self.filelist = []
        self.lamlist = []

        ######################################################################
        # Basic resolution/configuration parameters
        ######################################################################

        self.nlens = 201            # Number of lenslets across array
        self.pitch = 110e-6         # Lenslet pitch (meters)
        self.interlace = 3.         # Interlacing
        self.philens = -arcsin(1. / sqrt(self.interlace**2 + 1))  # Rotation angle of the lenslets (radians).
        self.pinhole = True         # Use a pinhole grid?
        self.lenslet_sampling = 1. / 2.4  # lenslet size in lambda/D
        self.lenslet_wav = 686.     # Wavelength at which this is defined (nm)

        ######################################################################
        # Detector stuff
        ######################################################################

        # NOTE: historically CRISPY was built with square detectors in mind, so npix refers to both dimensions. Here we have a non-square detector, so npix refers to the x-dimension, the larger of the two dimensions.
        self.npix = 6248            # Number of pixels in final detector. 
        self.pixsize = 3.76e-6      # Pixel size (meters)
        self.fitting_window = [1212, 4936, 0, 3724]  # Pixel bounds [xmin, xmax, ymin, ymax] for fitting lamsol.dat
        self.pxperdetpix = 1        # Oversampling of the final detector pixels
        self.convolve = True        # Whether to convolve kernels with Gaussian (simulating defocus)
        self.FWHM = 1.85               # FWHM at detector plane (detector pixels)
        self.FWHMlam = 686.         # Wavelength at which FWHM is defined (nm)
        self.gaussian = False       # Use standard Gaussian kernels instead of library
        self.gaussian_hires = False  # Use Gaussians for hires PSFLet matching

        # Placeholder detector noise parameters (unused)
        # self.RN = 0.2               # Read noise (electrons)
        # self.CIC = 1e-3             # Clock-induced charge (electrons/pixel/frame)
        # self.dark = 1e-5            # Dark current (electrons/pixel/second)
        # self.Traps = False          # Use trap-assisted recombination model

        self.QE = 1.0               # detector QE; need to make this wavelength-dependent
        # self.losses = 0.34          # total losses for on-axis PSF (given by J. Krist)
        # self.Nreads = 10            # number of reads for a frame
        # self.timeframe = 1000       # time in second for a frame (from file)

        ######################################################################
        # Spectrograph stuff
        ######################################################################

        self.distortPISCES = False  # If True, use measured PISCES distortion/dispersion
        self.BW = 0.18              # Spectral bandwidth (if distortPISCES==False)
        self.npixperdlam = 2.0      # Number of pixels per spectral resolution element
        self.nchanperspec_lstsq = 2.0  # num_wavelengths per pixel for least squares
        self.R = 120                 # Spectral resolving power

        # carry-over old parameter names
        self.lenslet_wav = self.lenslet_wav
        self.lenslet_sampling = self.lenslet_sampling

        self.makeHeader()

    def makeHeader(self):
        self.hdr = fits.PrimaryHDU().header
        self.hdr.append(('comment', ''), end=True)
        self.hdr.append(('comment', '*' * 60), end=True)
        self.hdr.append(('comment', '*' * 22 + ' General parameters ' + '*' * 18), end=True)
        self.hdr.append(('comment', '*' * 60), end=True)
        self.hdr.append(('comment', ''), end=True)
        self.hdr.append(('NLENS', self.nlens, '# lenslets across array'), end=True)
        self.hdr.append(('PITCH', self.pitch, 'Lenslet pitch (meters)'), end=True)
        self.hdr.append(('INTERLAC', self.interlace, 'Interlacing'), end=True)
        self.hdr.append(('PHILENS', self.philens * 180. / np.pi, 'Rotation angle of the lenslets (deg)'), end=True)
        self.hdr.append(('PIXSIZE', self.pixsize, 'Pixel size (meters)'), end=True)
        self.hdr.append(('LENSAMP', self.lenslet_sampling, 'Lenslet sampling (lam/D)'), end=True)
        self.hdr.append(('LSAMPWAV', self.lenslet_wav, 'Lenslet sampling wavelength (nm)'), end=True)
        self.hdr.append(('FWHM', self.FWHM, 'FHWM of PSFLet at detector (pixels)'), end=True)
        self.hdr.append(('FWHMLAM', self.FWHMlam, 'Wavelength at which FWHM is defined (nm)'), end=True)
        self.hdr.append(('NPIX', self.npix, 'Number of detector pixels'), end=True)
        self.hdr.append(('DISPDIST', self.distortPISCES, 'Use PISCES distortion/dispersion?'), end=True)
        if self.distortPISCES:
            self.hdr.append(('BW', self.BW, 'Bandwidth'), end=True)
            self.hdr.append(('PIXPRLAM', self.npixperdlam, 'Pixels per resolution element'), end=True)
            self.hdr.append(('R', self.R, 'Spectral resolution'), end=True)
