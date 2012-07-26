Getting Started
===============

Python Knowledge
---------------
For those unfamiliar with Python we strongly rogramming needs. 

Quick Python Overview (10 minutes or so): 
   * `OpenOpt Python for the impatient <http://openopt.org/PythonIntroduction>`_
   * `Instant Python <http://hetland.org/writing/instant-python.html>`_
   
Python for beginners, experienced users, or if you want a refresher:
   * `Official Python tutorial <http://docs.python.org/tutorial/>`_
   * `Python for non-programmers <http://wiki.python.org/moin/BeginnersGuide/NonProgrammers>`_
   * `Dive into Python <http://www.diveintopython.net/>`_
   * `Thinking in Python <http://www.mindview.net/Books/TIPython>`_
   * 

For experienced users of other languages:
   * `NumPy for Matlab <http://www.scipy.org/NumPy_for_Matlab_Users/>`_
      * Also the `Mathesaurus <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_
      * Matlab commands in Numerical Python `cheatsheet <http://mathesaurus.sourceforge.net/matlab-python-xref.pdf>`_
   * `Scientific Python <http://www.scipy.org/>`_
   


Requirements
------------

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
--------------------
* iPython 0.13: Even though iPython is not a *requirement* it is
  **strongly** recommended. iPython provides a very nice and simple
  shell interface for the Python interpreter with such niceties as tab
  completion, object exploration, running and editing from the shell,
  debugging, and history to name a few. You want this. 
* KaSim/Kappa : This wonderful rules-based package can be run natively
  from PySB to take advantage of its stochastic simulation
  capabilities and great visualization tools. It is a great complement
  to the modeling tools in BioNetGen.
* cookbooks 
