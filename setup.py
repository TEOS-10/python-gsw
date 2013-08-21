#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import codecs
from setuptools import setup

from gsw import __version__, __authors__, __email__, __maintainer__

install_requires = ['numpy', 'nose']

classifiers = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Intended Audience :: Science/Research
Intended Audience :: Developers
Intended Audience :: Education
License :: OSI Approved :: MIT License
Operating System :: OS Independent
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Education
Topic :: Software Development :: Libraries :: Python Modules
"""

CHANGES = open('CHANGES.txt').read()
LICENSE = open('LICENSE.txt').read()
README = codecs.open('README.md', encoding='utf-8').read()

config = dict(name='gsw',
              version=__version__,
              packages=['gsw', 'gsw/gibbs', 'gsw/utilities', 'gsw/test'],
              package_data={'gsw': ['utilities/data/*.npz']},
              test_suite='tests',
              use_2to3=True,
              license=LICENSE,
              long_description=u'%s\n\n%s' % (README, CHANGES),
              classifiers=filter(None, classifiers.split("\n")),
              description='Gibbs SeaWater Oceanographic Package of TEOS-10',
              author=__authors__,
              author_email=__email__,
              maintainer=__maintainer__,
              maintainer_email=__email__,
              url='http://pypi.python.org/pypi/seawater/',
              download_url='https://pypi.python.org/pypi/gsw/',
              platforms='any',
              keywords=['oceanography', 'seawater', 'teos-10', 'gibbs'],
              install_requires=install_requires)

setup(**config)
