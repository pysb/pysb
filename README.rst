PySB
====

Python Systems Biology modeling framework

http://pysb.org/

PySB (pronounced "Pie Ess Bee") is a framework for building rule-based
mathematical models of biochemical systems. It works nicely with
scientific Python libraries such as NumPy, SciPy and SymPy for model
simulation and analysis.

Installation
------------

The core of PySB is pure python and thus installs readily with
``easy_install`` or ``pip``::

    $ easy_install pysb

or::

    $ pip install pysb

However the majority of PySB's capabilities require the installation
of some other software and a few more Python modules, as described
below.

BioNetGen
~~~~~~~~~

The main rules-processing logic in PySB is handled by BioNetGen which
itself is written in Perl. If you don't already have Perl installed,
visit http://www.perl.org/get.html and download the installer for your
platform. Then download BioNetGen itself from
http://bionetgen.org/index.php/BioNetGen_Distributions and extract the
.tgz file into ``/usr/local/share`` on a Unix or Mac OS system, or
``c:\Program Files`` on a Windows system.

Python modules
~~~~~~~~~~~~~~

These python modules will give you the ability to simulate and analyze
your model.

* Required (provides crucial functionality)

  * numpy
  * scipy
  * sympy

* Optional (nice to have, but you can still get things done without them)

  * pygraphviz

Kappa
~~~~~

PySB can use the tools from the Kappa Lanaugage suite to visualize
your model as a "contact map" and "influence map". This functionality
requires ``complex`` and ``KaSim`` which can be downloaded from
http://kappalanguage.org/downloads .

Documentation
-------------

Everything is available online at http://docs.pysb.org/ . You can also
generate the documentation locally by running the following commands::

    $ cd doc
    $ make html

And then open _build/html/index.html in your web browser.
