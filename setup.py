#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
from gsw import __version__
from distutils.core import setup
try:  # Python 3
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2
    from distutils.command.build_py import build_py

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

readme = codecs.open('README.rst', encoding='utf-8')
config = dict(name='gsw',
              version=__version__,
              packages=['gsw', 'gsw/gibbs', 'gsw/utilities'],
              package_data={'gsw': ['utilities/data/*.npz']},
              license=open('LICENSE.txt').read(),
              description='Gibbs SeaWater Oceanographic Package of TEOS-10',
              long_description=readme.read(),
              author=u'Filipe Fernandes, Eric Firing, Ådlandsvik Bjørn',
              author_email='ocefpaf@gmail.com',
              maintainer='Filipe Fernandes',
              maintainer_email='ocefpaf@gmail.com',
              url='http://pypi.python.org/pypi/seawater/',
              download_url='https://pypi.python.org/pypi/gsw/',
              classifiers=filter(None, classifiers.split("\n")),
              platforms='any',
              cmdclass={'build_py': build_py},
              keywords=['oceanography', 'seawater'],
              install_requires=['numpy', 'nose']
             )

setup(**config)
