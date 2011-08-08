#!/usr/bin/env python

from distutils.core import setup
import sys, subprocess, traceback, re

def main():

    try:
        version = get_version()
    except Exception as e:
        sys.stderr.write(str(e))
        return

    setup(name='pysb',
          version=version,
          description='Python Systems Biology modeling system',
          author='Jeremy Muhlich',
          author_email='jmuhlich@bitflood.org',
          packages=['pysb', 'pysb.generator', 'pysb.tools'],
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
