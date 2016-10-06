from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup
import setuptools.command.build_py

import versioneer

import sys, os, subprocess, re

class build_py(setuptools.command.build_py.build_py):
    # Simplest way to use a specific list of fixers. Note use_2to3_fixers will
    # be ignored.
    fixer_names = ['lib2to3.fixes.fix_ne']

def main():

    cmdclass = {'build_py': build_py}
    cmdclass.update(versioneer.get_cmdclass())

    setup(name='pysb',
          version=versioneer.get_version(),
          description='Python Systems Biology modeling framework',
          long_description='PySB (pronounced "Pie Ess Bee") is a framework ' + \
              'for building rule-based mathematical models of biochemical ' + \
              'systems. It works nicely with scientific Python libraries ' + \
              'such as NumPy, SciPy and SymPy for model simulation and ' + \
              'analysis.',
          author='Jeremy Muhlich',
          author_email='jmuhlich@bitflood.org',
          url='http://pysb.org/',
          packages=['pysb', 'pysb.generator', 'pysb.importers', 'pysb.tools',
                    'pysb.examples', 'pysb.export', 'pysb.simulator',
                    'pysb.testing', 'pysb.tests'],
          scripts=['scripts/pysb_export'],
          # We should really specify some minimum versions here.
          install_requires=['numpy', 'scipy', 'sympy'],
          setup_requires=['nose'],
          tests_require=['coverage', 'pygraphviz', 'matplotlib', 'pexpect',
                         'pandas'],
          cmdclass=cmdclass,
          use_2to3=True,
          keywords=['systems', 'biology', 'model', 'rules'],
          classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Mathematics',
            ],
          )

if __name__ == '__main__':
    main()
