#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs

try:
    from setuptools import setup
    from setuptools.command.sdist import sdist
except ImportError:
    from distutils.core import setup
    from distutils.command.sdist import sdist

# TODO: find setuptools equivalent.
try:  # Python 3
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:  # Python 2
    from distutils.command.build_py import build_py


class sdist_hg(sdist):
    """Automatically generate the latest development version when creating a
    source distribution."""
    user_options = sdist.user_options + [('dev', None, "Add a dev marker")]

    def initialize_options(self):
        sdist.initialize_options(self)
        self.dev = 0

    def run(self):
        if self.dev:
            suffix = '.dev%d' % self.get_tip_revision()
            self.distribution.metadata.version += suffix
        sdist.run(self)

    def get_tip_revision(self, path=os.getcwd()):
        from mercurial.hg import repository
        from mercurial.ui import ui
        from mercurial import node
        repo = repository(ui(), path)
        tip = repo.changelog.tip()
        return repo.changelog.rev(tip)

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
              version='3.0.1',
              packages=['gsw',
                        'gsw/gibbs',
                        'gsw/utilities'],
              package_data={'': ['gsw/utilities/data/*.npz']},
              license=open('LICENSE.txt').read(),
              description='Gibbs SeaWater Oceanographic Package of TEOS-10',
              long_description=readme.read(),
              author='Filipe Fernandes, Eric Firing, Ådlandsvik Bjørn',
              author_email='ocefpaf@gmail.com',
              maintainer='Filipe Fernandes',
              maintainer_email='ocefpaf@gmail.com',
              url='http://pypi.python.org/pypi/seawater/',
              # TODO: upload to PyPI
              download_url='',
              classifiers=filter(None, classifiers.split("\n")),
              platforms='any',
              cmdclass={'build_py': build_py},
              # NOTE: python setup.py sdist --dev
              #cmdclass={'sdist': sdist_hg},
              keywords=['oceanography', 'seawater'],
              install_requires=['numpy', 'nose']
             )

setup(**config)
