
==========
 Tutorial
==========

This tutorial will walk you through the creation of your first PySB
model. It will cover the basics, provide a guide through the different
programming constructs and finally deal with more complex
rule-building. Users should be able to write simple programs and
follow the programs in the example sections after finishing this
section. In what follows we will assume you are using *iPython* as your
interactive python REPL but the standard *python* REPL could be used
as well. 

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


Basic rule-based modeling and PySB
==================================
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

The key components that every model in PySB needs are:

* Model definition: this instantiates the model object
* Monomer definition: this instantiates the monomers that are allowed
  in our model.
* Parameters: These are the numerical parameters needed to create a
  mass-action or stochastic simulation to solve the system's evolution
  over time.
* Rules: the set of statements that describe how *species*, that is
  separate instances of monomers, interact as prescribed by the
  parameters involved in a given rule.

For what follows we will use an example taken from our work in
`extrinsic apoptosis signaling`_. In this work the initiator caspases,
activated by an upstream signal, play an essential role activating the
effector Bcl-2 proteins downstream. Caspase-8, a representative
initiator caspase, and Bid, a representative effector BH3 protein,
bind to create a complex. Caspase-8 then cleaves the protein Bid to
create truncated Bid. This is usually considered a two-step process as
follows: ::

            kf
   C8 + Bid ↔ C8:Bid   <--- Complex formation step
            kr

          kc
   C8:Bid → C8 + tBid  <--- Complex dissociation step

Where tBid is the truncated Bid. The parameters *kf*, *kr*, and *kc*
represent the forward, reverse, and catalytic rates that dictate the
consumption of Bid via catalysis by C8 and the formation of tBid. For
completeness we write the ODEs that represent this system below: ::

   d[C8]/dt     = -kf*[C8]*[Bid] + kr*[C8:Bid] + kc*[C8:Bid]
   d[Bid]/dt    = -kf*[C8]*[Bid] + kr*[C8:Bid]
   d[C8:Bid]/dt =  kf*[C8]*[Bid] - kr*[C8:Bid] - kc*[C8:Bid]
   dt[tBid]/dt  =  kc*[C8:Bid] 
   
The species names in square braces represent concentrations, usually
give in molar (M) and time in seconds. These ordinary differential
equations (ODEs) are then integrated numerically to obtain the
evolution of the system over time. As shown, the parameters are needed
to instantiate the equations but the manner in which the parameters
influence the concentration changes in each chemical *species* (the
terms on the left of the equations) is determined by the manner in
which the chemical equations are written. We term this connectivity
between chemical species as the system *topology*. This *topology*
along with a number of parameters, dictates the output of a given
model. The connectivity between the chemical reactants specify the
manner in which the equations are written. We will explore how one
could instantiate a model, add different actions to the model, and
create multiple instances of a model *without* having to resort to the
tedious and repetitive writing of equations as those listed above.

The Empty Model
===============

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
========

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
*monomer* named 'Bid' with a bond site 'b::

    Monomer('Bid', ['b'])

Note that this looks like a Python function call.  This is because it
*is* in fact a Python function call! [#func]_ The first argument to
the function is a string (ecnlosed in quotation marks) specifying the
monomer's name and the second argument is a list of strings specifying
the names of its sites. Note that a monomer does not need to have
state sites. There is also a third, optional argument for
specifying whether any of the sites are "state sites" and the list of
valid states for those sites.  We will introduce state sites later.

Let's define two monomers in our model, corresponding to Caspase-8, an
initiator caspase involved in apoptosis (**C8**) and BH3-interacting domain death
agonist (**Bid**) (ref?)::

    Monomer('C8', ['b'])
    Monomer('Bid', ['b', 'S'])

Note that although the C8 monomer only has one site 'b', you must
still use the square brackets to indicate a *list* of binding
sites. Anticipating what comes below, the *'S'* site will become a
state site and hence, we choose to represent it in upper case but this
is not mandatory. 

Now our model file should look like this::

    from pysb import *

    Model()

    Monomer('C8', ['b'])
    Monomer('Bid', ['b', 'S'])


We can run ``python mymodel.py`` again and verify there are no errors,
but you should still have not output given that we have not *done*
anything with the monomers. Now we can do something with them.

Run the *ipython* (or *python*) interpreter with no arguments to enter
interactive mode (be sure to do this from the same directory where
you've saved :file:`mymodel.py`) and run the following code::

   >>> from mymodel import model
   >>> for m in model.monomers:
   ...     print m
   ... 

You should see the following output::

    Monomer(name='C8', sites=['b'], site_states={})
    Monomer(name='Bid', sites=['b', 'S'], site_states={})

In the first line, we treat :file:`mymodel.py` as a *module* [#mod]_
and import its symbol ``model``.  In the second and third lines, we
loop over the ``monomers`` attribute of ``model``, printing each
element of that list.  The output for each monomer is a more verbose,
explicit representation of the same call we used to define it. [#mkw]_

Here we can start to see how PySB is different from other modeling
tools.  With other tools, text files are typically created with a
certain syntax, then passed through an execution tool to perform a
task and produce an output, whether on the screen or to an output
file.  In PySB on the other hand we write Python code defining our
model in a regular Python module, and the elements we define in that
module can be inspected and manipulated as Python objects
interactively in one of the Python REPLs such as *iPython* or
*Python*. We will explore this concept in more detail in the next
section, but for now we will cover the other types components needed
to create a working model.

Parameters
==========

A ``Parameter`` is a named constant floating point number used as a
reaction rate constant, compartment volume or initial (boundary)
condition for a species (*parameter* in BNG). A parameter is defined
using the keyword ``Parameter`` followed by its name and value. Here
is how you would define a parameter named 'kf1' with the value
:math:`4 \times 10^{-7}`::

    Parameter('kf1', 4.0e-7)

The second argument may be any numeric expression, but best practice
is to use a floating-point literal in scientific notation as shown in
the example above. For our model we will need three parameters, one
each for the forward, reverse, and catalytic reactions in our
system. Go to your :file:`mymodel.py` file and add the lines
corresponding to the parameters so that your file looks like this::

   from pysb import *

   Model()

   Monomer('C8', ['b'])
   Monomer('Bid', ['b', 'S'])

   Parameter('kf', 1.04e-06)
   Parameter('kr', 1.04e-06)
   Parameter('kc', 1.04e-06)

Once this is done start the *ipython* (or *python*) intepreter and
enter the following commands:: 

   >>> from mymodel import model
   >>> model.parameters
and you should get an output such as::

   {'kf': Parameter(name='kf', value=1.04e-06),
    'kr': Parameter(name='kr', value=1.04e-06),
    'kc': Parameter(name='kc', value=1.04e-06)}

Your model now has monomers and parameters specified. In the next
section we will specify rules, which specify the interaction between
monomers and parameters. 

Rules
=====

Rules, as described in this section, comprise the basic elements of
procedural instructions that encode biochemical interactions. In its
simplest form a rule is a chemical reaction that can be made general
to a range of monomer states or very specific to only one kind of
monomer in one kind of state. We follow the style for writing rules as
described in `BioNetGen`_ but the style proposed by `Kappa`_ is quite
similar with only some differences related to the implementation
details (e.g. mass-action vs. stochastic simulations, compartments or
no compartments, etc). We will write two rules to represent the
interaction between the reactants and the products in a two-step
manner as described in the `Basic rule-based modeling and PySB`_
section. 

The general pattern for a rule consists of the statement *Rule* and in
parenthesis a series of statements separated by commas, namely the
rule name (string), the rule interactions, and the rule
parameters. The rule interactions make use of the following
operators::
   *+* operator to represent complexation 
   *<>* operator to represent backward/forward reaction
   *>>* operator to represent forward-only reaction
   *%* operator to represent a binding interaction between two species

To illustrate the use of the operators and the rule syntax we write
the complex formation reaction with labels illustrating the parts of
the rule::

   Rule('C8_Bid_bind', C8(b=None) + Bid(b=None, S=None) <> C8(b=1) % Bid(b=1, S=None), *[kf, kr]) 
	     |              |     |           |         |     |    |     |           |
             |              |     |           |         |     |    |     |          parameter list
	     |              |     |           |         |     |    |     |
	     |              |     |           |         |     |    |    Whenbound species
	     |              |     |           |         |     |    |
	     |		    |     |           |         |     |   binding operator
	     |              |     |           |         |     |
	     |              |     |           |         |    bound species
	     |              |     |           |         |
	     |		    |     |           |        forward/backward operator
	     |              |     |           |
	     |		    |     |          unbound species
	     |              |     |
	     |		    |    complexation / addition operator
	     |              |
	     |		   unbound species
	    rule name

The *rule name* can be any string and should be enclosed in single (')
or double (") quotation marks. The species are *instances* of the
mononmers in a specific state. In this case we are requiring that *C8*
and *Bid* are both unbound, as we would not want any binding to occur
with species that are previously bound. The *complexation* or
*addition* operator tells the program that the two species are being
added, that is, undergoing a transition, to form a new species as
specified on the right side of the rule. The forward/backward
operator states that the reaction is reversible. Finally the *binding*
operator indicates that there is a bond formed between two or more
species. This is indicated by the matching integer (in this case *1*)
in the bonding site of both species along with the *binding*
operator. If a non-reversible rule is desired, then the *forward-only*
operator can be relplaced for the *forward/backward* operator. 

In order to actually change the state of the Bid protein we must now
edit the monomer so that have an acutal state site as follows::

   Monomer('Bid', ['b', 'S'], {'S':['u', 't']})

Having added the state site we can now further specify the state of
the Bid protein whe it undergoes rule-based interactions and
explicitly indicate the changes of the protein state.  

With this state site added, we can now go ahead and write the rules
that will account for the binding step and the unbinding step as
follows::

   Rule('C8_Bid_bind', C8(b=None) + Bid(b=None, S='u') <>C8(b=1) % Bid(b=1, S='u'), *[kf, kr])
   Rule('tBid_from_C8-Bid', C8(b=1) % Bid(b=1, S='u') >> C8(b=None) % Bid(b=None, S='t'), kc)

As shown, the initial reactants, *C8* and *Bid* initially in the
unbound state and, for Bid, in the 'u' state, undergo a complexation
reaction and further a dissociation reaction to return the original
*C8* protein and the *Bid* protein but now in the 't' state,
indicating its truncation. Make these additions to your
:file:`mymodel.py` file. After you are done, your file should look
like this::

   from pysb import *

   Model()

   Monomer('C8', ['b'])
   Monomer('Bid', ['b', 'S'], {'S':['u', 't']})

   Parameter('kf', 1.04e-06)
   Parameter('kr', 1.04e-06)
   Parameter('kc', 1.04e-06)

   Rule('C8_Bid_bind', C8(b=None) + Bid(b=None, S=None) <> C8(b=1) % Bid(b=1, S=None), *[kf, kr]) 
   Rule('tBid_from_C8Bid', C8(b=1) % Bid(b=1, S='u') >> C8(b=None) + Bid(b=None, S='t'), kc)

Once you are done editing your file, start your *ipython* (or
*python*) interpreter and type the commands at the prompts below. Once
you load your model you should be able to probe and check that you
have the correct monomers, parameters, and rules. Your output should
be very similar to the one presented.::

   >>> from mymodel import model
   >>> model.monomers
   {'C8': Monomer(name='C8', sites=['b'], site_states={}),
   'Bid': Monomer(name='Bid', sites=['b', 'S'], site_states={'S': ['u', 't']})}
   >>> model.parameters
   {'kf': Parameter(name='kf', value=1.04e-06),
    'kr': Parameter(name='kr', value=1.04e-06),
    'kc': Parameter(name='kc', value=1.04e-06)}
   >>> model.rules
   {'C8_Bid_bind': Rule(name='C8_Bid_bind', reactants=C8(b=None) +
   Bid(b=None, S=None), products=C8(b=1) % Bid(b=1, S=None),
   rate_forward=Parameter(name='kf', value=1.04e-06),
   rate_reverse=Parameter(name='kr', value=1.04e-06)),
   'tBid_from_C8Bid': Rule(name='tBid_from_C8Bid', reactants=C8(b=1) %
   Bid(b=1, S=u), products=C8(b=None) + Bid(b=None, S=t),
   rate_forward=Parameter(name='kc', value=1.04e-06))}

With this we are almost ready to run a simulation, all we need now is
to specify the initial conditions of the system.

Initial conditions
==================
Having specified the *monomers*, the *parameters* and the *rules* we
have the basics of what is needed to generate a set of ODEs and run a
model. From a mathematical perspective a system of ODEs can only be
solved if a bound is placed on the ODEs for integration. In our case,
these bounds are the initial conditions of the system that indicate
how much non-zero initial species are present at time *t=0s* in the
system. In our system, we only have two initial species, namely *C8*
and *Bid* so we need to specify their initial concentrations. To do
this we enter the following lines of code into the :file:`mymodel.py`
file::

   Initial(C8(b=None), 1000)
   Initial(Bid(b=None, S='u'), 10000)




Observables
===========

Simulation and analysis
=======================

Higher-order rules
==================

Compartments
============



.. rubric:: Footnotes

.. [#func] Technically speaking it's a constructor, not just any old
   function.

.. [#mod] Python allows users to write python code such as PySB code
   to a file and use this code later as an executable script or
   from an interactive instance. Such files are called *modules* and
   can be imported into a Python instance. See `Python modules
   <http://docs.python.org/tutorial/modules.html>'_ for details.

.. [#mkw] The astute Python programmer will recognize this as the
   ``repr`` of the monomer object, using keyword arguments in the
   constructor call.

.. _BioNetGen: http://bionetgen.org/index.php/Documentation

.. _Kappa: http://www.kappalanguage.org/documentation

.. _extrinsic apoptosis signaling: http://www.plosbiology.org/article/info%3Adoi%2F10.1371%2Fjournal.pbio.0060299
