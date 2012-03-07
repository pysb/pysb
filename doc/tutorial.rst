Tutorial
========

This tutorial will walk you through the creation of your first PySB
model.  First we will learn how to use basic rule-based modeling
constructs, then take a deeper look at some simulation and analysis
tools available through PySB, and finally explore PySB's most unique
feature, so-called "higher-order" rules.

.. note:: We assume a basic understanding of rule-based modeling,
   specifically the formalism known as the Kappa calculus.  We will be
   referencing concepts from the BioNetGen (BNG) implementation of
   this formalism. `Section 1 of the BNG tutorial
   <http://bionetgen.org/index.php/BioNetGen_Tutorial#Structure_of_the_Input_File>`_
   serves as a good introduction to the language.

Basic rule-based modeling
-------------------------

First steps
~~~~~~~~~~~

Let's create our model, which we'll call ``mymodel``. Open your
favorite Python code editor and create a file called
:file:`mymodel.py`.  The first two lines of every model must be as
follows, so copy them into your file and save it::

    from pysb import *

    Model()

Now we have the simplest possible model -- the empty model!

To verify that your model is valid and your PySB installation is
working, run :file:`mymodel.py` through the Python interpreter by
typing the following command at your command prompt::

   python mymodel.py

If all went well, you will not see any output. This is to be expected,
because a PySB model script *defines* a model but doesn't *do*
anything with it. We'll revisit this issue once we've added some more
components to our model to make it more interesting.

Monomers
~~~~~~~~

The first type of component we'll typically add to a new model is a
``Monomer``, which represents a protein or other molecule in the
system being modeled (*molecule type* in BNG).  A monomer is defined
using the keyword ``Monomer`` followed by the name you'd like to
assign to the monomer and any sites it may have.  Here is how you
would declare a monomer named 'EGFR' with two sites 'l' and 'd'::

    Monomer('EGFR', ['l', 'd'])

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
~~~~~~~~~~

Rules
~~~~~

Compartments
~~~~~~~~~~~~

Initial conditions
~~~~~~~~~~~~~~~~~~

Observables
~~~~~~~~~~~


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
