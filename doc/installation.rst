Installation
============

There are two different ways to install and use PySB:

1. **Download and run the virtual machine containing the complete PySB
   installation.** Users wishing to try out PySB, who are unfamiliar with the
   procedure for installing Python packages or who just want a simpler
   installation procedure, should choose this option.

   *OR*

2. **Install the necessary software dependencies natively on your computer.**
   Users who are comfortable with installing Python packages and compiling
   source code should choose this option.

Option 1: The PySB virtual machine
----------------------------------

For easy installation, we provide a pre-configured virtual machine (VM) running
the `Ubuntu Linux`_ operating system that comes with all necessary software
installed.  It also includes other useful software (e.g., `Git`_, `IPython`_,
`GraphViz`_, `Kappa`_, `OCaml`_), and has been designed to make getting
up-to-date versions of PySB and other required packages easy. The VM will
require 2GB of free hard drive space to install, plus an extra 500MB during the
download and import process.

In addition to the PySB virtual machine file itself, you'll need virtualization
software to run it, such as Oracle's free and open-source `VirtualBox`_.  The
instructions given below are for VirtualBox, but other virtualization software
such as `VMWare Player`_ (free) or `Parallels`_ can also be used. Here's the
installation procedure:

1. `Download VirtualBox <https://www.virtualbox.org/wiki/Downloads>`_ and
   install it.

2. `Download the PySB OVA <http://www.pysb.org/#download>`_ (Open Virtualization
   Appliance) file. The file is approximately 500MB. Double-click the downloaded
   .ova file to open it in VirtualBox, if your web browser doesn't offer to do
   so.

3. VirtualBox will now display the Appliance Import Wizard. Click the "Import"
   button to continue. Note that the newly created VM will occupy about 2GB of
   hard drive space. Once the import is complete, you may delete the .ova file.

4. In the VirtualBox Manager window, double-click the "PySB demo" entry to
   launch the VM.

Now you may use the virtual machine to create and work with PySB models. All
files created in the VM will be saved on a virtual disk image, and you may shut
down the VM and re-launch it later from the VirtualBox Manager without losing
your work. If you would like to share files between the VM and your desktop
system, see the VirtualBox documentation for instructions.


Option 2: Installing the dependencies yourself
----------------------------------------------

Required software
^^^^^^^^^^^^^^^^^

These are the minimum requirements needed to simulate a model and plot the
results. Listed versions are the ones that are known to work well. Later
versions should work but earlier versions may not. The major exception to this
guideline is Python itself -- see below.

* `Python`_ 2.7

  **PySB requires Python 2.7**! Earlier versions of Python (2.6 and lower) are not
  compatible, nor are later versions (3.x).

* `SciPy`_ 0.9
* `NumPy`_ 1.6
* `SymPy`_ 0.7
* `matplotlib`_ 1.1
* `BioNetGen`_ 2.2 (requires Perl -- see below)
* `Perl`_ 5.8

  Any newer 5.x version is OK too. Mac and Linux users can use the version of
  Perl included with their operating system. Windows users should get Strawberry
  Perl from http://strawberryperl.com/.

Recommended software
^^^^^^^^^^^^^^^^^^^^

* `IPython`_: An alternate interactive Python shell, much improved over the
  standard one.
* `Kappa`_: A rule-based modeling tool that can produce several useful model
  visualizations or perform an agent-based model simulation. PySB provides
  direct integration with some of these capabilities. Both the older `Kappa`
  (simplx/complx) and the newer `KaSim` packages are supported.

.. _Ubuntu Linux: http://www.ubuntu.com
.. _Kappa: http://www.kappalanguage.org
.. _Git: http://git-scm.com
.. _IPython: http://ipython.org/
.. _OCaml: http://caml.inria.fr/ocaml/
.. _GraphViz: http://www.graphviz.org/
.. _VirtualBox: https://www.virtualbox.org/
.. _VMWare Player: http://www.vmware.com/products/player/
.. _Parallels: http://www.parallels.com/
.. _Python: http://www.python.org/
.. _SciPy: http://www.scipy.org/
.. _NumPy: http://numpy.scipy.org/
.. _SymPy: http://sympy.org/
.. _matplotlib: http://matplotlib.org/
.. _BioNetGen: http://www.bionetgen.org/
.. _Perl: http://www.perl.org/
