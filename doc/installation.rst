Installation
============

There are two different ways to install and use PySB:

1. **Install PySB natively on your computer (recommended).**

   *OR*

2. **Download a Docker container with PySB and Jupyter Notebook.** If you
   are familiar with `Docker`_, PySB can be installed from the Docker
   Hub by typing :command:`docker pull pysb/pysb`. Further details are
   below.

.. note::
    **Need Help?**
    If you run into any problems with installation, please visit our chat room:
    https://gitter.im/pysb/pysb

Option 1: Install PySB natively on your computer
------------------------------------------------

1. **Install Anaconda**

   Our recommended approach is to use `Anaconda`_, which is a distribution of
   Python containing most of the numeric and scientific software needed to
   get started. If you are a Mac or Linux user, have used Python before and
   are comfortable using ``pip`` to install software, you may want to skip
   this step and use your existing Python installation.

   Anaconda has a simple graphical installer which can be downloaded from
   https://www.anaconda.com/products/individual - select your operating system
   and download the 64 bit version. From PySB 2.0, we only support
   Python 3.x (see the :ref:`faq` for specific version support). The
   default installer options are usually appropriate.

2. **Install PySB**

   The installation is very straightforward with ``conda`` - type the
   following in a terminal:

       :command:`conda install -c alubbock pysb`

   .. note::
        You may wish to use the Anaconda prompt, which sets up the Anaconda
        paths automatically, rather than the standard command prompt or
        terminal on your operating system. Otherwise, you may have to use the
        full path to the ``conda`` command each time, and may end up using
        the system version of Python, rather than the Anaconda one.

   .. note::
        You can also install PySB using ``pip``, but in that case you will
        need to manually install `BioNetGen`_ into the default path for your
        platform (/usr/local/share/BioNetGen on Mac and Linux, c:\\Program
        Files\\BioNetGen on Windows), or set the `BNGPATH` environment
        variable to the BioNetGen path on your machine.

3. **Start Python and PySB**

   If you installed Python using `Anaconda`_ on Windows, search for and select
   ``IPython`` from your Start Menu (Windows). Otherwise, open a terminal
   and type ``python`` to get started (or ``ipython``, if installed).

   You will then be at the Python prompt. Type ``import pysb`` to try
   loading PySB. If no error messages appear and the next Python prompt
   appears, you have succeeded in installing PySB! You can now proceed to
   the :doc:`tutorial`.

Recommended additional software
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following software is not required for the basic operation of PySB, but
provides extra capabilities and features when installed.

* `cython`_
  Cython is a package for compiling Python code into C code on the fly. It
  is used by :class:`pysb.simulator.ScipyOdeSimulator` to greatly improve the
  speed of ODE integration. PySB will detect and use Cython automatically,
  if available. To install with Anaconda, type
  :command:`conda install cython`.
  With pip, type :command:`pip install cython`.

* `matplotlib`_

  This Python package allows you to plot the results of your simulations. It
  is not a hard requirement of PySB but many of the example scripts use it.
  `matplotlib`_ is included with `Anaconda`_. Otherwise, it can be installed
  with :command:`pip install matplotlib`.

* `pandas`_

  This Python package provides extra capabilities for examining large
  numerical datasets, with statistical summaries and database-like
  manipulation capabilities. It is not a hard requirement of PySB, but it is a
  useful addition, particularly with large sets of simulation results.
  `pandas`_ is included with `Anaconda`_. Otherwise, it can be installed with
  :command:`pip install pandas`.

* `IPython`_

  An alternate interactive Python shell, much improved over the standard one.
  `IPython`_ is included with `Anaconda`_. Otherwise, it can be installed
  with :command:`pip install ipython`.

* `Kappa`_ 4.0

  Kappa is a rule-based modeling tool that can produce several useful model
  visualizations or perform an agent-based model simulation. PySB optionally
  interfaces with its *KaSim* simulator and *KaSa* static analyzer.

  To install Kappa for PySB use, put the ``KaSim`` executable (and optionally
  ``KaSa`` if you have it) in ``/usr/local/share/KaSim`` (Mac or Linux) or
  ``C:\\Program Files\\KaSim`` (Windows). If you would like to put it somewhere
  else, set the ``KAPPAPATH`` environment variable to the full path to the
  folder containing the ``KaSim`` and ``KaSa`` executables. Note that if you
  have downloaded the official binary build of KaSim, it will be named something
  like ``KaSim_4.0_winxp.exe`` or ``KaSim_4.0_mac_OSX_10.10``. Regardless of
  where you install it, you will need to rename the file to strip out the
  version and operating system information so that you have just ``KaSim.exe``
  (Windows) or ``KaSim`` (Mac or Linux).

  On Anaconda, Kappa can be installed with
  :command:`conda install -c alubbock kappa`.

Option 2: Docker container with PySB and Jupyter Notebook
----------------------------------------------------------

Background
^^^^^^^^^^

`Docker`_ is a virtualization platform which encapsulates software within a
container. It can be thought of like a virtual machine, only it contains
just the application software (and supporting dependencies) and not a full
operating system stack.

Install Docker and the PySB software stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Install Docker**

   To use PySB with Docker, first you'll need to install Docker, which can be
   obtained from https://www.docker.com/community-edition#/download (Windows
   and Mac). Linux users should use their package manager (e.g. ``apt-get``).

2. **Download the PySB software stack from the Docker Hub**

   On the command line, this requires a single command:

       :command:`docker pull pysb/pysb`

   This only needs to be done once, or when software updates are required.

3. **Start the container**

   Start the Docker container with the following command (on Linux, the command
   may need to be prefixed with ``sudo``):

       :command:`docker run -it --rm -p 8888:8888 pysb/pysb`

   This starts the PySB Docker container with Jupyter notebook and connects it
   to port 8888.

4. **Open Jupyter Notebook in a web browser**

   Open a web browser of your choice and enter the address
   http://localhost:8888 in the address bar. You should see a web page with the
   Jupyter notebook logo. Several example and tutorial notebooks are included
   to get you started.

Important notes for Docker installations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To see graphics from matplotlib within the Jupyter Notebook, you'll need to
set the following option in your notebooks before calling any plot commands:

.. code-block:: ipython

    %matplotlib inline

Any Jupyter notebooks created will be saved in the container itself, rather
than on the host computer. Notebooks can be downloaded using the Jupyter
interface, or a directory on the host computer can be shared with the
container.

The PySB container builds on the Jupyter SciPy notebook, which contains
further information on the options available for the container (such
as sharing a directory with the host computer to preserve notebooks,
setting a password and more). Documentation from the Jupyter project is
available at
https://hub.docker.com/r/jupyter/scipy-notebook/

.. _Anaconda: https://www.anaconda.com/products/individual
.. _Docker: http://www.docker.org/
.. _Kappa: http://www.kappalanguage.org/
.. _Git: http://git-scm.com/
.. _IPython: http://ipython.org/
.. _OCaml: http://caml.inria.fr/ocaml/
.. _GraphViz: http://www.graphviz.org/
.. _pandas: http://pandas.pydata.org/
.. _Python: http://www.python.org/
.. _SciPy: http://www.scipy.org/
.. _NumPy: http://www.numpy.org/
.. _SymPy: http://www.sympy.org/
.. _matplotlib: http://matplotlib.org/
.. _BioNetGen: http://www.bionetgen.org/
.. _Perl: http://www.perl.org/
.. _Cython: http://cython.org/
