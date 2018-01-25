from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup
import versioneer


def main():

    cmdclass = versioneer.get_cmdclass()

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
          install_requires=['numpy', 'scipy', 'sympy', 'networkx'],
          setup_requires=['nose'],
          tests_require=['coverage', 'pygraphviz', 'matplotlib', 'pexpect',
                         'pandas', 'theano', 'h5py', 'mock', 'cython'],
          cmdclass=cmdclass,
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
