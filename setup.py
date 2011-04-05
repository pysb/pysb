#!/usr/bin/env python

from distutils.core import setup
import sys, subprocess, traceback, re

def main():

    # get a nice version number from git-describe
    gitcmd = ['git', 'describe', '--always', '--abbrev=4']
    try:
        gitproc = subprocess.Popen(gitcmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        retcode = gitproc.wait()
        if retcode:
            raise GitError(gitproc.stderr.read())
        version = gitproc.stdout.next().strip()
        version = re.sub(r'^v', '', version)
    except (OSError, GitError) as e:
        sys.stderr.write("Error running 'git describe' to determine version:\n\n")
        sys.stderr.write("command\n=====\n" + " ".join(gitcmd) + "\n\n")
        sys.stderr.write("error\n====\n" + str(e) + "\n")
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

if __name__ == '__main__':
    main()
