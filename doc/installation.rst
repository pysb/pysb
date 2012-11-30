Installation
============

There are two ways to install and use PySB:

1. **Download and run the virtual machine containing the complete PySB
   installation.** Users wishing to try out PySB, who are unfamiliar with the
   procedure for installing Python packages, or who just want a simpler
   installation procedure should choose this option.

2. **Install the necessary software dependencies natively on your computer.**
   Users who prefer to run PySB natively and who are comfortable with
   installing Python packages should choose this option.

Downloading the PySB Virtual Machine
------------------------------------

For easy installation, we provide a pre-configured virtual machine running the
`Ubuntu Linux`_ operating system that comes with all necessary software
installed.  It also includes other useful software (e.g., `Git`_,
`iPython`_, `GraphViz`_, `Kappa`_, `OCaml`_), and has been preconfigured to
make getting up-to-date versions of PySB and other required packages easy.

In addition to the PySB virtual machine file itself, you'll need virtualization
software to run it, such as Oracle's free and open-source `Virtual Box`_.  The
instructions given below are for Virtual Box, but other virtualization software
such as `VMWare Player`_ (free) or `Parallels`_ can also be used. Here's the
installation procedure:

1. `Download the PySB virtual machine OVA (Open Virtualization Archive) file
   by clicking this link <http://www.pysb.org>`_. The file is approximately
   800MB.
   
2. Download Virtual Box at https://www.virtualbox.org/wiki/Downloads.
   etc.
    
.. _Ubuntu Linux: http://www.ubuntu.com
.. _Kappa: http://www.kappalanguage.org
.. _Git: http://git-scm.com
.. _iPython: http://ipython.org/
.. _OCaml: http://caml.inria.fr/ocaml/
.. _GraphViz: http://www.graphviz.org/
.. _Virtual Box: https://www.virtualbox.org/
.. _VMWare Player: http://www.vmware.com/products/player/
.. _Parallels: http://www.parallels.com/

Installing the Dependencies Yourself
------------------------------------

Required Packages
^^^^^^^^^^^^^^^^^

The following are what we consider the *necessary* to use PySB as a
biological simulation tool. The versions listed are the ones that are
known to work well with the material in this documentation. Later
versions *should* work and earlier versions *might* work. Advanced
users may want to replace these requirements as they see fit. 

* Python 2.7: You will need a version of the Python interpreter in your
  machine. 
* NumPy 1.7: You may not need NumPy for simple model building but you will
  want to have it for any sort of numerical manipulation of your
  model. The work presented here has been carried out using NumPy 1.7
  or later. 
* SymPy 0.7: Like NumPy, you may not need SymPy to carry out simple
  model building and instantiation but if you want to run numerical
  simulation,s SymPy will be a required tool for symbolic math manipulation.
* BioNetGen 2.1.8: The Biological Network Generator is a very useful tool
  for rules-based modeling. It is a very powerful and useful package
  for modeling and simulation of biological systems and provides a set
  of useful tools that could be used with PySB. As of now, PySB uses
  BioNetGen as a tool to generate the reaction connectivity network
  using its robust engine. If you want to generate biochemical
  representations of a biological system, you will need
  BioNetGen. BioNetGen depends on Perl 2, so you will need that as
  well. 
* SciPy 0.10: Scientific Python provides a suite of extremely useful
  tools for scientific computing in the Python environment. For
  example, SciPy provides the LSODA integrator interface that we use
  in PySB. 
* MatPlotLib 1.2 (PyLab): This package provides a very useful
  interface for generation, manipulation, export, etc of plots in two
  and three dimensions. If you want to visualize any type of plots you
  will need MatPlotLib. 

Recommended
^^^^^^^^^^^

* iPython 0.13: Even though iPython is not a *requirement* it is
  **strongly** recommended. iPython provides a very nice and simple
  shell interface for the Python interpreter with such niceties as tab
  completion, object exploration, running and editing from the shell,
  debugging, and history to name a few. You want this. 
* KaSim/Kappa : This wonderful rules-based package can be run natively
  from PySB to take advantage of its stochastic simulation
  capabilities and great visualization tools. It is a great complement
  to the modeling tools in BioNetGen.

