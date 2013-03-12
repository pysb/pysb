PySB documentation
==================

PySB is a framework for building mathematical rule-based models of biochemical
systems as Python programs. PySB abstracts the complex process of creating
equations describing interactions among multiple proteins (or other
biomolecules) into a simple and intuitive domain specific language embedded
within Python. PySB accomplishes this by automatically generating sets of
`BNGL`_ or `Kappa`_ rules and using the rules for simulation or analysis. PySB
makes it straightforward to divide models into modules and to call libraries of
reusable elements (macros) that encode standard biochemical actions. These
features promote model transparency, reuse and accuracy. PySB interoperates with
standard scientific Python libraries such as `NumPy`_, `SciPy`_ and `SymPy`_ to
enable model simulation and analysis.

.. _BNGL: http://www.bionetgen.org
.. _Kappa: http://www.kappalanguage.org
.. _NumPy: http://numpy.scipy.org
.. _SciPy: http://www.scipy.org
.. _SymPy: http://sympy.org

Contents:

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

