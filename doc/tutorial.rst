.. _BioNetGen: http://bionetgen.org/index.php/Documentation
.. _Kappa: http://www.kappalanguage.org/documentation

Tutorial
========

This tutorial will walk you through the creation of your first PySB
model. It will cover the basics, provide a guide through the different
programming constructs and finally deal with more complex
rule-building. Users should be able to write simple programs and
follow the programs in the example sections after finishing this
section. 

.. note:: Familiarity with rules-based biomodel encoding tools such as
   `BioNetGen`_ or `Kappa`_ would be useful to users unfamiliar with
   *Rules-based* approaches to modeling. Although we start from the
   basics in this tutorial, some familiarity with these tools will be
   useful.

.. warning:: A basic understanding of the Python programming language
   is essential for the use of PySB. Although the user can go through
   this tutorial and develop an understanding of the PySB tools,
   advanced programming with PySB will require understanding of
   Python. Some useful tutorials/guides include the `Official Python
   Tutorial <http://docs.python.org/tutorial/>`_, `Dive into Python
   <http://www.diveintopython.net/>`_, `Numerical Python (NumPy)
   <http://numpy.scipy.org/>`_, and `Scientific Python (SciPy)
   <http://scipy.org/Getting_Started>`_.


Basic rule-based modeling and PySB ==================================
In rules-based modeling, units that undergo transformations such as
proteins, small molecules, protein complexes, etc are termed
*species*. The interactions among these *species* are then represented
using structured objects that describe the interactions between the
*species* and constitute what we describe as *rules*. The specific
details of how species and rules are specified can vary across
different rules-based modeling approaches. In PySB we have chosen to
ascribe to the approaches found in `BioNetGen`_ and `Kappa`_, but
other approaches are certainly possible for advanced users interested
in modifying the source code. Each rule, describing the interaction
between *species* or sets of *species* must be assigned a set of
*parameters* associated with the nature of the *rule*. Given that
`BioNetGen`_ and `Kappa`_ both describe interactions using a
mass-action kinetics formalism, the *parameters* will necessarily
consist of reaction rates. In what follows we describe how a model can
be instantiated in PySB, how *species* and *rules* are specified, and
how to run a simple simulation.


The Empty Model
---------------

We begin by creating a model, which we will call ``mymodel``. Open your
favorite Python code editor and create a file called
:file:`mymodel.py`. The first lines of a PySB program must contain
these lines so you can type them or paste them in your editor::

    from pysb import *

    Model()

Now we have the simplest possible model -- the empty model!

To verify that your model is valid and your PySB installation is
working, run :file:`mymodel.py` through the Python interpreter by
typing the following command at your command prompt::

   python mymodel.py

If all went well, you should not see any output. This is to be
expected, because this PySB script *defines* a model but does not
execute any contents. We will revisit these concepts once we have
added some components to our model.

Monomers
--------

Chemical *species* in PySB, whether they are small molecules,
proteins, or representations of many molecules are all composed of
*Monomers*. *Monomers* are the subunit that defines how a *species*
can be defined and used. A *Monomer* is defined using the keyword
``Monomer`` followed by the desired *monomer* name and the *sites*
relevant to that monomer. In PySB, like in `BioNetGen`_ or `Kappa`_,
there are two types of *sites*, namely bond-making/breaking sites (aka
transformation sites) and state sites. The former allow for the
description of bonds between *species* while the latter allow for the
assignment of *states* to species. Following the first lines of code
entered into your model in the previous section we will add a
*monomer* named 'Bid' with a bond site 'b' and a state site 's'::

    Monomer('Bid', ['b', 's'])

Note that this looks like a Python function call.  This is because it
*is* in fact a Python function call! [#func]_ The first argument to
the function is a string specifying the monomer's name, and the second
argument is a list of strings specifying the names of its sites.
There is also a third, optional argument for specifying whether any of
the sites are "state sites" and the list of valid states for those
sites.  We'll get to state sites a bit later.

Let's define two monomers in our model, corresponding to EGF
(epidermal growth factor) and EGFR (the EGF receptor)::

    Monomer('EGF', ['r'])
    Monomer('EGFR', ['l', 'd'])

Note that although the EGF monomer only has one site 'r', you must
still use the square brackets.

Now our model file should look like this::

    from pysb import *

    Model()

    Monomer('EGF', ['r'])
    Monomer('EGFR', ['l', 'd'])

We can run ``python mymodel.py`` again and verify there are no errors,
but now that we have some components in this model let's *do*
something with it.

Run the Python interpreter with no arguments to enter interactive mode
(be sure to do this from the same directory where you've saved
:file:`mymodel.py`) and run the following code:

    >>> from mymodel import model
    >>> for m in model.monomers:
    ...     print m
    ... 
    Monomer(name='EGF', sites=['r'], site_states={})
    Monomer(name='EGFR', sites=['l', 'd'], site_states={})

In the first line, we treat :file:`mymodel.py` as a module and import
its symbol ``model``.  In the second and third lines, we loop over the
``monomers`` attribute of ``model``, printing each element of that
list.  The output for each monomer is a more verbose, explicit
representation of the same call we used to define it. [#mkw]_

Here we can start to see how PySB is a bit different from most other
modeling tools.  With other tools, we typically create a text file
with a certain syntax, then pass that text file through the tool in
order to perform some task and produce an output file.  In PySB on the
other hand we write Python code defining our model in a regular Python
module, and the elements we define in that module can be inspected and
manipulated as Python objects. We'll explore this concept more fully
in the next section, but for now let's cover the other types of
components we can add to our model.

Parameters
----------

A ``Parameter`` is a named constant floating point number used as a
reaction rate constant, compartment volume or initial (boundary)
condition for a species (*parameter* in BNG). A parameter is defined
using the keyword ``Parameter`` followed by its name and value. Here
is how you would define a parameter named 'kf1' with the value
:math:`4 \times 10^{-7}`::

    Parameter('kf1', 4e-7)

The second argument may be any numeric expression, but best practice
is to use a floating-point literal in scientific notation as shown in
the example above.

Rules
-----

Compartments
------------

Initial conditions
------------------

Observables
-----------


Simulation and analysis
-----------------------

Higher-order rules
------------------

.. rubric:: Footnotes

.. [#func] Technically speaking it's a constructor, not just any old
   function.

.. [#mkw] The astute Python programmer will recognize this as the
   ``repr`` of the monomer object, using keyword arguments in the
   constructor call.
