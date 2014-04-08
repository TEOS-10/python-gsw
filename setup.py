#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import io
import re
from setuptools import setup


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


VERSIONFILE = "gsw/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

email = "ocefpaf@gmail.com"
maintainer = "Filipe Fernandes"
authors = ['Eric Firing', u'Bjørn Ådlandsvik', 'Filipe Fernandes']

install_requires = ['numpy', 'nose']

LICENSE = read('LICENSE.txt')
long_description = read('README.txt', 'CHANGES.txt')

config = dict(name='gsw',
              version=verstr,
              packages=['gsw', 'gsw/gibbs', 'gsw/utilities', 'gsw/test'],
              package_data={'gsw': ['utilities/data/*.npz']},
              test_suite='tests',
              use_2to3=True,
              license=LICENSE,
              long_description=long_description,
              classifiers=['Development Status :: 4 - Beta',
                           'Environment :: Console',
                           'Intended Audience :: Science/Research',
                           'Intended Audience :: Developers',
                           'Intended Audience :: Education',
                           'License :: OSI Approved :: MIT License',
                           'Operating System :: OS Independent',
                           'Programming Language :: Python',
                           'Topic :: Education',
                           'Topic :: Scientific/Engineering',
                           ],
              description='Gibbs SeaWater Oceanographic Package of TEOS-10',
              author=authors,
              author_email=email,
              maintainer=maintainer,
              maintainer_email=email,
              url='http://pypi.python.org/pypi/seawater/',
              download_url='https://pypi.python.org/pypi/gsw/',
              platforms='any',
              keywords=['oceanography', 'seawater', 'TEOS-10', 'gibbs'],
              install_requires=install_requires)

setup(**config)
