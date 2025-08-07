#!/usr/bin/env python
import os
import sys

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup


setup(
    name='crispy',
    version="1.0.0", 
    author='Maxime Rizzo',
    author_email = 'maxime.j.rizzo@nasa.gov',
    url = 'https://github.com/mjrfringes/crispy',
    packages =['crispy','crispy.tools'],
    license = 'GNU GPLv3',
    description ='The Coronagraph and Rapid Imaging Spectrograph in Python',
    package_dir = {"crispy":'crispy', "crispy.tools":'crispy/tools'},
    include_package_data=True,
    python_requires='>=3.8',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
        ],
    include_dirs = ['crispy','crispy/tools'],
    install_requires = [
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'matplotlib>=3.3.0',
        'astropy>=4.0',
        'photutils>=1.0.0'
    ],
)
