#!/usr/bin/env python

from distutils.core import setup
import sys, subprocess, traceback, re

def main():

    try:
        version_file = file('RELEASE-VERSION', 'w+')
        version = get_version()
        version_file.seek(0, 0)
        version_file.write(version)
    except Exception as e:
        try:
            version = version_file.read()
        except Exception as e:
            sys.stderr.write(str(e))
            return

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

class GitError(Exception):
    pass

def get_version():
    """Get a nice version number from git-describe"""
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
