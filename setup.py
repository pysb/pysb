#!/usr/bin/env python

from distutils.core import setup
import distutils.cmd
import sys, os, subprocess, traceback, re

def main():

    rv_filename = 'RELEASE-VERSION'
    try:
        version = get_version()
        version_file = open(rv_filename, 'w')
        version_file.write(version)
        version_file.close() # ensure sdist build process sees new contents
    except Exception:
        try:
            version_file = open(rv_filename, 'r')
            version = version_file.read()
        except IOError as e:
            if e.errno == 2 and e.filename == rv_filename:
                sys.stderr.write(
"""This does not appear to be a git repository, and the file %s is not
present. In order to build or install PySB, please either download a
distribution from http://pypi.python.org/pypi/pysb or clone the git repository
at https://github.com/pysb/pysb.git\n""" % rv_filename)
                sys.exit(1)

    setup(name='pysb',
          version=version,
          description='Python Systems Biology modeling framework',
          long_description='PySB (pronounced "Pie Ess Bee") is a framework ' + \
              'for building rule-based mathematical models of biochemical ' + \
              'systems. It works nicely with scientific Python libraries ' + \
              'such as NumPy, SciPy and SymPy for model simulation and ' + \
              'analysis.',
          author='Jeremy Muhlich',
          author_email='jmuhlich@bitflood.org',
          url='http://pysb.org/',
          packages=['pysb', 'pysb.generator', 'pysb.tools', 'pysb.examples'],
          requires=['numpy', 'scipy', 'sympy'],
          cmdclass = {'test': test},
          keywords=['systems', 'biology', 'model', 'rules'],
          classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Scientific/Engineering :: Mathematics',
            ],
          )

class test(distutils.cmd.Command):
    description = "run tests (requires nose)"
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import nose
        config = nose.config.Config(exclude=[re.compile('examples')],
                                    env=os.environ)
        nose.run(defaultTest='pysb', config=config, argv=[''])

class GitError(Exception):
    pass

def get_version():
    """Get a nice version number from git-describe"""

    # ensure that we are working in a pysb git repo
    setup_path = os.path.abspath(os.path.dirname(__file__))
    print setup_path
    if not os.path.exists(os.path.join(setup_path, '.git')):
        raise Exception("setup.py is not in the root of a git repository; "
                        "aborting")
    os.chdir(setup_path)

    # run git describe
    gitcmd = ['git', 'describe', '--always', '--abbrev=4']
    try:
        gitproc = subprocess.Popen(gitcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        retcode = gitproc.wait()
        if retcode:
            raise GitError(gitproc.stderr.read())
        version = gitproc.stdout.next().strip()
        version = re.sub(r'^v', '', version)
        return version
    except (OSError, GitError) as e:
        raise Exception("Error running 'git describe' to determine version:\n\n" +
                        "command\n=====\n" + " ".join(gitcmd) + "\n\n" +
                        "error\n====\n" + str(e) + "\n")

if __name__ == '__main__':
    main()
