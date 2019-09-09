#!/usr/bin/env python3

# Copyright (C) 20019-2019  CAMd
# Please see the accompanying LICENSE file for further information.

import os
import re
import sys
from setuptools import setup, find_packages
from distutils.command.build_py import build_py as _build_py
from glob import glob
from os.path import join


if sys.version_info < (3, 5, 0, 'final', 0):
    raise SystemExit('Python 3.5 or later is required!')


with open('Readme.md') as fd:
    long_description = fd.read()


setup(name='spacesense',
      version="0.1",
      description='Remote sensing handeling and processing',
      url='https://wiki.fysik.dtu.dk/ase',
      maintainer='ASE-community',
      maintainer_email='ase-users@listserv.fysik.dtu.dk',
      license='LGPLv2.1+',
      platforms=['unix'],
      packages=["spacesense",
                "spacesense.config",
                "spacesense.visualisation",
                ],
      install_requires=['numpy', 'scipy', 'matplotlib'],
      extras_require={},
      long_description=long_description,
      classifiers=[
          'Development Status :: 1 - Early',
          'GNU Lesser General Public License v3 or later (LGPLv3+)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'])

