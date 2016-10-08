PySB
====

.. image:: https://badges.gitter.im/pysb/pysb.svg
   :alt: Join the chat at https://gitter.im/pysb/pysb
   :target: https://gitter.im/pysb/pysb?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://api.travis-ci.org/pysb/pysb.png

Python Systems Biology modeling framework

http://pysb.org/

PySB (pronounced "Pie Ess Bee") is a framework for building rule-based
mathematical models of biochemical systems. It works nicely with
scientific Python libraries such as NumPy, SciPy and SymPy for model
simulation and analysis.

Installation
------------

PySB depends on the following:

  * numpy
  * scipy
  * sympy
  * Perl - http://www.perl.org/get.html
  * BioNetGen - http://bionetgen.org/

For full instructions, see the Installation chapter of the manual at
http://docs.pysb.org/en/latest/installation.html

Documentation
-------------

The manual is available online at http://docs.pysb.org/. You can also
generate the documentation locally by installing Sphinx and running
the following commands::

    $ cd doc
    $ make html

Then open _build/html/index.html in your web browser.
