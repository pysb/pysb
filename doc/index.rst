.. pysb documentation master file, created by
   sphinx-quickstart on Thu Aug  4 14:26:58 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PySB documentation
==================

PySB is a framework for building mathematical models of biochemical systems as
Python programs. PySB abstracts the complex process of creating equations
describing interactions among multiple proteins (or other biomolecules) into a
simple and intuitive domain specific language embedded within Python. PySB
accomplishes this by automatically generating sets of `BNGL`_ or `Kappa`_ rules
and using the rules for simulation or analysis. PySB makes it straightforward
to divide models into modules and to call libraries of reusable elements
(macros) that encode standard biochemical actions. These features promote model
transparency, reuse and accuracy. PySB can handle simple models, modular
models, and multiple instances of models. PySB also interoperates with standard
scientific Python libraries such as `NumPy`_, `SciPy`_  and `SymPy`_ enabling
model simulation and analysis. 

We invite users to contribute and share their innovations and ideas to
make PySB a better open-source tool for the programming community. 

.. _BNGL: http://www.bionetgen.org
.. _Kappa: http://www.kappalanguage.org
.. _NumPy: http://numpy.scipy.org
.. _SciPy: http://www.scipy.org
.. _SymPy: http://sympy.org

.. toctree::
   :maxdepth: 3

   installation
   tutorial
   modules/index
   useful_references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

