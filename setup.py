from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup
import versioneer
import os


def main():
    this_directory = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_directory, 'README.rst'), 'r') as f:
        long_description = f.read()

    cmdclass = versioneer.get_cmdclass()

    setup(name='pysb',
          version=versioneer.get_version(),
          description='Python Systems Biology modeling framework',
          long_description=long_description,
          long_description_content_type='text/x-rst',
          author='Jeremy Muhlich',
          author_email='jmuhlich@bitflood.org',
          url='http://pysb.org/',
          packages=['pysb', 'pysb.generator', 'pysb.importers', 'pysb.tools',
                    'pysb.examples', 'pysb.export', 'pysb.simulator',
                    'pysb.testing', 'pysb.tests'],
          scripts=['scripts/pysb_export'],
          # We should really specify some minimum versions here.
          python_requires='>=3.6',
          install_requires=['numpy', 'scipy>=1.1', 'sympy>=1.6', 'networkx',
                            'futures; python_version == "2.7"'],
          setup_requires=['nose'],
          tests_require=['coverage', 'pygraphviz', 'matplotlib', 'pexpect',
                         'pandas', 'h5py', 'mock', 'cython',
                         'python-libsbml', 'libroadrunner'],
          cmdclass=cmdclass,
          keywords=['systems', 'biology', 'model', 'rules'],
          classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Mathematics',
            ],
          )


if __name__ == '__main__':
    main()
