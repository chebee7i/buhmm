#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Installation script.

"""

from __future__ import print_function

import os
import sys
import warnings

try:
    from setuptools import setup
    has_setuptools = True
except ImportError:
    from distutils.core import setup
    has_setuptools = False

import distutils
from distutils.core import Extension
from distutils.command import install_data
from distutils.command.build_ext import build_ext

class my_install_data(install_data.install_data):
    # A custom install_data command, which will install it's files
    # into the standard directories (normally lib/site-packages).
    def finalize_options(self):
        if self.install_dir is None:
            installobj = self.distribution.get_command_obj('install')
            self.install_dir = installobj.install_lib
        print('Installing data files to {0}'.format(self.install_dir))
        install_data.install_data.finalize_options(self)

def has_cython():
    """Returns True if Cython is found on the system."""
    try:
        import Cython
        return True
    except ImportError:
        return False

def check_opt(name):
    x = eval('has_{0}()'.format(name.lower()))
    msg = "%(name)s not found. %(name)s extensions will not be built."
    if not x:
        warnings.warn(msg % {'name':name})
    return x

def main():

    cmdclass = {'install_data': my_install_data}

    try:
        import Cython.Distutils
    except ImportError:
        msg = "Cython is required. Please install it before proceeding."
        print(msg)
        raise
    else:
        cmdclass['build_ext'] = Cython.Distutils.build_ext

    try:
        import numpy as np
    except ImportError:
        msg = "NumPy is required. Please install it before proceeding."
        print(msg)
        raise

    counts = Extension(
        "buhmm.counts",
        ["buhmm/counts.pyx"],
        include_dirs=[np.get_include()],
        libraries=['m'],
    )

    # Active Cython modules
    cython_modules = [
        counts,
    ]

    other_modules = []

    ext_modules = cython_modules + other_modules

    data_files = ()

    install_requires = [
        'Cython >= 0.20',
        'numpy >= 1.8',
        'iterutils >= 0.1.6',
        'six >= 1.4.0', # 1.4.0 includes six.moves.range.
    ]

    packages = [
        'buhmm',
    ]

    # Tests
    package_data = dict(zip(packages, [['tests/*.py']]*len(packages)))

    desc = "Bayesian inference for unifilar hidden Markov models in Python."
    kwds = {
        'name':                 "buhmm",
        'version':              "0.1.0dev",
        'url':                  "http://github.com/chebee7i/buhmm",

        'packages':             packages,
        'package_data':         package_data,
        'provides':             ['buhmm'],
        'install_requires':     install_requires,
        'ext_modules':          ext_modules,
        'cmdclass':             cmdclass,
        'data_files':           data_files,
        'include_package_data': True,

        'author':               "Humans",
        'author_email':         "chebee7i@gmail.com",
        'description':          desc,
        'long_description':     open('README.md').read(),
        'license':              "MIT",
    }

    # Automatic dependency resolution is supported only by setuptools.
    if not has_setuptools:
        del kwds['install_requires']

    setup(**kwds)

if __name__ == '__main__':
    if sys.argv[-1] == 'setup.py':
        print("To install, run 'python setup.py install'\n")

    v = sys.version_info[:2]
    if v < (2, 7):
        msg = "buhmm requires Python version >2.7"
        print(msg.format(v))
        sys.exit(-1)

    main()
