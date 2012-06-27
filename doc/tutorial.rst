`
==========
 Tutorial
==========

This tutorial will walk you through the creation of your first **Pysb**
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
   is essential for the use of **Pysb**. Although the user can go through
   this tutorial and develop an understanding of the **Pysb** tools,
   advanced programming with **Pysb** will require understanding of
   Python. Some useful tutorials/guides include the `Official Python
   Tutorial <http://docs.python.org/tutorial/>`_, `Dive into Python
   <http://www.diveintopython.net/>`_, `Numerical Python (NumPy)
   <http://numpy.scipy.org/>`_, and `Scientific Python (SciPy)
   <http://scipy.org/Getting_Started>`_.


Basic rule-based modeling and **Pysb**
======================================
In rules-based modeling, units that undergo transformations such as
proteins, small molecules, protein complexes, etc are termed
*species*. The interactions among these *species* are then represented
using structured objects that describe the interactions between the
*species* and constitute what we describe as *rules*. The specific
details of how species and rules are specified can vary across
different rules-based modeling approaches. In **Pysb** we have chosen to
ascribe to the approaches found in `BioNetGen`_ and `Kappa`_, but
other approaches are certainly possible for advanced users interested
in modifying the source code. Each rule, describing the interaction
between *species* or sets of *species* must be assigned a set of
*parameters* associated with the nature of the *rule*. Given that
`BioNetGen`_ and `Kappa`_ both describe interactions using a
mass-action kinetics formalism, the *parameters* will necessarily
consist of reaction rates. In what follows we describe how a model can
be instantiated in **Pysb**, how *species* and *rules* are specified, and
how to run a simple simulation.

The key components that every model in **Pysb** needs are:

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
follows.

.. math::
   C8 + Bid \underset{kr}{\overset{kf}{\leftrightharpoons}} C8:Bid \quad {\longleftarrow \mbox{Complex formation step}} \\
   C8:Bid \overset{kc}{\rightarrow} C8 + tBid \quad {\longleftarrow \mbox{Complex dissociation step}}

Where tBid is the truncated Bid. The parameters *kf*, *kr*, and *kc*
represent the forward, reverse, and catalytic rates that dictate the
consumption of Bid via catalysis by C8 and the formation of tBid. For
completeness we write the ODEs that represent this system below:

.. math::
   \frac{d[C8]}{dt}     &= -kf[C8]*[Bid] + kr*[C8:Bid] + kc*[C8:Bid] \\
   \frac{d[Bid]}{dt}    &= -kf*[C8]*[Bid] + kr*[C8:Bid] \\
   \frac{d[C8:Bid]}{dt} &=  kf*[C8]*[Bid] - kr*[C8:Bid] - kc*[C8:Bid] \\
   \frac{dt[tBid]}{dt}  &=  kc*[C8:Bid] 
   :label: ODEs
   
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
:file:`mymodel.py`. The first lines of a **Pysb** program must contain
these lines so you can type them or paste them in your editor as shown
below. Comments in the *Python* language are denoted by a hash (``#``)
in the first column.

.. literalinclude:: examples/mymodel0.py

Now we have the simplest possible model -- the empty model!

To verify that your model is valid and your **Pysb** installation is
working, run :file:`mymodel.py` through the Python interpreter by
typing the following command at your command prompt::

   python mymodel.py

If all went well, you should not see any output. This is to be
expected, because this **Pysb** script *defines* a model but does not
execute any contents. We will revisit these concepts once we have
added some components to our model.

Monomers
========

Chemical *species* in **Pysb**, whether they are small molecules,
proteins, or representations of many molecules are all composed of
*Monomers*. *Monomers* are the subunit that defines how a *species*
can be defined and used. A *Monomer* is defined using the keyword
``Monomer`` followed by the desired *monomer* name and the *sites*
relevant to that monomer. In **Pysb**, like in `BioNetGen`_ or `Kappa`_,
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

Now our model file should look like this:

.. literalinclude:: examples/mymodel1.py

We can run ``python mymodel.py`` again and verify there are no errors,
but you should still have not output given that we have not *done*
anything with the monomers. Now we can do something with them.

Run the *ipython* (or *python*) interpreter with no arguments to enter
interactive mode (be sure to do this from the same directory where
you've saved :file:`mymodel.py`) and run the following code::

   >>> from mymodel import model
   >>> model.monomers

You should see the following output::

    Monomer(name='C8', sites=['b'], site_states={})
    Monomer(name='Bid', sites=['b', 'S'], site_states={})

In the first line, we treat :file:`mymodel.py` as a *module* [#mod]_
and import its symbol ``model``.  In the second and third lines, we
loop over the ``monomers`` attribute of ``model``, printing each
element of that list.  The output for each monomer is a more verbose,
explicit representation of the same call we used to define it. [#mkw]_

Here we can start to see how **Pysb** is different from other modeling
tools.  With other tools, text files are typically created with a
certain syntax, then passed through an execution tool to perform a
task and produce an output, whether on the screen or to an output
file.  In **Pysb** on the other hand we write Python code defining our
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
corresponding to the parameters so that your file looks like this:

.. literalinclude:: examples/mymodel2.py

Once this is done start the *ipython* (or *python*) intepreter and
enter the following commands:: 

   >>> from mymodel import model
   >>> model.parameters
and you should get an output such as::

   {'kf': Parameter(name='kf', value=1.0e-07),
    'kr': Parameter(name='kr', value=1.0e-03),
    'kc': Parameter(name='kc', value=1.0    )}

Your model now has monomers and parameters specified. In the next
section we will specify rules, which specify the interaction between
species and parameters. 

.. Warning:: 

   **Pysb** or the integrators that we suggest for use for numerical
   manipulation do not keep track of units for the user. As such, the
   user is responsible for keeping track of the model in units that
   make sense to the user! For example, the forward rates are
   typically in :math:`M^{-1}s^{-1}`, the reverse rates in
   :math:`s^{-1}`, and the catalytic rates in :math:`s^{-1}`. For the
   present examples we have chosen to work in a volume size of
   :math:`1.0 pL` corresponding to the volume of a cell and to specify
   the Parameters and `Initial conditions`_ in numbers of molecules
   per cell. If you wish to change the units you must change *all* the
   parameter values accordingly.

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
like this:

.. literalinclude:: examples/mymodel3.py

Once you are done editing your file, start your *ipython* (or
*python*) interpreter and type the commands at the prompts below. Once
you load your model you should be able to probe and check that you
have the correct monomers, parameters, and rules. Your output should
be very similar to the one presented (output shown below the ``'>>>'``
python prompts).::

   >>> from mymodel import model
   >>> model.monomers
      {'C8': Monomer(name='C8', sites=['b'], site_states={}),
      'Bid': Monomer(name='Bid', sites=['b', 'S'], site_states={'S': ['u', 't']})}
   >>> model.parameters
      {'kf': Parameter(name='kf', value=1.0e-07),
       'kr': Parameter(name='kr', value=1.0e-03),
       'kc': Parameter(name='kc', value=1.0    )}
   >>> model.rules
      {'C8_Bid_bind': Rule(name='C8_Bid_bind', reactants=C8(b=None) +
      Bid(b=None, S=None), products=C8(b=1) % Bid(b=1, S=None),
      rate_forward=Parameter(name='kf', value=1.0e-07),
      rate_reverse=Parameter(name='kr', value=1.0e-03)),
      'tBid_from_C8Bid': Rule(name='tBid_from_C8Bid', reactants=C8(b=1) %
      Bid(b=1, S=u), products=C8(b=None) + Bid(b=None, S=t),
      rate_forward=Parameter(name='kc', value=1.0))}

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

   Parameter('C8_0', 1000)
   Parameter('Bid_0', 10000)
   Initial(C8(b=None), C8_0)
   Initial(Bid(b=None, S='u'), Bid_0)

A parameter object must be declared to specify the initial condition
rather than just giving a value as shown above. Once the parameter
object is declared (i.e. *C8_0* and *Bid_0*) it can be fed to the
*Initial* definition. Now that we have specified the initial
conditions we are basically ready to run simulations. We will add an
*observables* call in the next section prior to running the
simulation.

Observables
===========

In our model we have two initial species (*C8* and *Bid*) and one
output species (*tBid*). As shown in the :eq:`ODEs` derived from the
reactions above, there are four mathematical species needed to
describe the evolution of the system (i.e. *C8*, *Bid*, *tBid*, and
*C8:Bid*). Although this system is rather small, there are situations
when we will have many more species than we care to monitor or
characterize throughout the time evolution of the :eq:`ODEs`. In
addition, it will often happen that the desirable species are
combinations or sums of many other species. For this reason the
rules-based engines we currently employ implemented the *Observables*
call which automatically collects the necessary information and
returns the desired species. In our case, we will monitor the amount
of free *C8*, unbound *Bid*, and active *tBid*. To specify the
observables enter the following lines in your :file:`mymodel.py` file
as follows::

   Observable('C8', C8(b=None))
   Observable('Bid', Bid(b=None, S='u')
   Observable('tBid', Bid(b=None, S='t')

As shown,the observable can be a species. As we will show later the
observable can also contain wild-cards and given the "don't care don't
write" approach to rule-writing it can be a very powerful approach to
observe activated complexes.  

Simulation and analysis
=======================
By now your :file:`mymodel.py` file should look something like this::

.. literalinclude:: examples/mymodel4.py

You can use a few commands to check that your model is defined
properly. Start your *ipython* (or *python*) interpreter and enter the
commands as shown below. Notice the output should be similar to the
one shown (output shown below the ``'>>>'``` prompts)::

   >>> from mymodel import model
   >>> model.monomers
      {'C8': Monomer(name='C8', sites=['b'], site_states={}),
       'Bid': Monomer(name='Bid', sites=['b', 'S'], site_states={'S': ['u', 't']})}
   >>> model.parameters
      {'kf': Parameter(name='kf', value=1.0e-07),
       'kr': Parameter(name='kr', value=1.0e-03),
       'kc': Parameter(name='kc', value=1.0    ),
       'C8_0': Parameter(name='C8_0', value=1000),
       'Bid_0': Parameter(name='Bid_0', value=10000)}
   >>> model.observables
      {'obsC8': <pysb.core.Observable object at 0x104b2c4d0>,
       'obsBid': <pysb.core.Observable object at 0x104b2c5d0>,
       'obstBid': <pysb.core.Observable object at 0x104b2c6d0>}
   >>> model.initial_conditions
      [(C8(b=None), Parameter(name='C8_0', value=1000)), (Bid(b=None, S=u), Parameter(name='Bid_0', value=10000))]
   >>> model.rules
      {'C8_Bid_bind': Rule(name='C8_Bid_bind', reactants=C8(b=None) +
      Bid(b=None, S=None), products=C8(b=1) % Bid(b=1, S=None),
      rate_forward=Parameter(name='kf', value=1.0e-07),    rate_reverse=Parameter(name='kr', value=1.0e-03)),
       'tBid_from_C8Bid': Rule(name='tBid_from_C8Bid', reactants=C8(b=1)
       % Bid(b=1, S=u), products=C8(b=None) + Bid(b=None, S=t),    rate_forward=Parameter(name='kc', value=1.0))}

With this we are now ready to run a simulation! The parameter values
for the simulation were taken directly from typical values in the
paper about `extrinsic apoptosis signaling`_. To run the simulation we
must use a numerical integrator. Common examples include LSODA, VODE,
CVODE, Matlab's ode15s, etc. We will use two *python* modules that are
very useful for numerical manipulation. We have adapted the
integrators in the *SciPy*[#sp]_ module to function seamlessly with
**Pysb** for integration of ODE systems. We will also be using the *PyLab*
[#pl]_ package for graphing and plotting from the command line. 

We will begin our simulation by loading the model from the *ipython*
(or *python*) interpreter as shown below::

   >>> from mymodel import model
   >>> model.monomers

Now, we will import the *PyLab* and **Pysb** integrator module. Enter
the commands as shown below::

   >>> from pysb.integrate import odesolve
   >>> from pylab import *

We have now loaded the integration engine and the graph engine into
the interpreter environment. You may get some feedback as some things
can be compiled at runtime, depending on your operating
system. The next thing we need is to tell the integrator the time
domain over which we wish to integrate the equations. For our case we
will use :math:`20000s` of simulation time. To do this we generate an
array using the *linspace* function. Enter the command below::

   >>> t = linspace(0, 20000)

This command assigns an array in the range :math:`[0..20000]` to the
variable *t*. You can type the name of the variable at any time to see
the content of the variable. Typing the variable *t* results in the
following::

   >>> t
   array([     0.        ,    408.16326531,    816.32653061,   1224.48979592,
            1632.65306122,   2040.81632653,   2448.97959184,   2857.14285714,
            3265.30612245,   3673.46938776,   4081.63265306,   4489.79591837,
            4897.95918367,   5306.12244898,   5714.28571429,   6122.44897959,
            6530.6122449 ,   6938.7755102 ,   7346.93877551,   7755.10204082,
            8163.26530612,   8571.42857143,   8979.59183673,   9387.75510204,
            9795.91836735,  10204.08163265,  10612.24489796,  11020.40816327,
           11428.57142857,  11836.73469388,  12244.89795918,  12653.06122449,
           13061.2244898 ,  13469.3877551 ,  13877.55102041,  14285.71428571,
           14693.87755102,  15102.04081633,  15510.20408163,  15918.36734694,
           16326.53061224,  16734.69387755,  17142.85714286,  17551.02040816,
           17959.18367347,  18367.34693878,  18775.51020408,  19183.67346939,
           19591.83673469,  20000.        ])

These are the points at which we will get data for each ODE from the
integrator. With this, we can now run our simulation. Enter the
following commands to run the simulation::

   >>> yout = odesolve(model, t)
   >>> yout['obsBid']
   array([ 10000.        ,   9601.77865674,   9224.08135988,   8868.37855506,
            8534.45591732,   8221.19944491,   7927.08884234,   7650.48970981,
            7389.81105408,   7143.5816199 ,   6910.47836131,   6689.32927828,
            6479.10347845,   6278.89607041,   6087.91189021,   5905.45001654,
            5730.89003662,   5563.68044913,   5403.32856328,   5249.39176146,
            5101.47069899,   4959.20384615,   4822.26262101,   4690.34720441,
            4563.18294803,   4440.51745347,   4322.11815173,   4207.77021789,
            4097.27471952,   3990.44698008,   3887.11517373,   3787.11923497,
            3690.30945136,   3596.54594391,   3505.69733323,   3417.64025401,
            3332.25897699,   3249.44415872,   3169.09326717,   3091.10923365,
            3015.40034777,   2941.87977234,   2870.4652525 ,   2801.07879018,
            2733.64632469,   2668.09744369,   2604.36497901,   2542.38554596,
            2482.09776367,   2423.44473279])

As you may recall we named some observables in the `Observables`_
section above. The variable *yout* contains an array of all the ODE
outputs from the integrators along with the named observables
(i.e. *obsBid*, *obstBid*, and *obsC8*) which can be called by their
names. We can therefore plot this data to visualize our output. Using
the commands imported from the *PyLab* module we can create a graph
interactively. Enter the commands as shown below::

   >>>ion()
   >>>figure()
   >>>plot(t, yout['obsBid'], label="Bid")
   >>>plot(t, yout['obstBid'], label="tBid")
   >>>plot(t, yout['obsC8'], label="C8")
   >>>legend()
   >>>xlabel("seconds")
   >>>ylabel("Molecules/cell")
   >>>show()

You should now have a figure in your screen showing the number of
*Bid* molecules decreaing from the initial amount decreasing over
time, the number of *tBid* molecules increasing over time, and the
number of free *C8* molecules decrease to about half. For help with
the above commands and to see more commands related to *PyLab* check
the documentation [#pl]_.

Congratulations! You have created your first model and run a
simulation!

=================
Advanced modeling
=================
In this section we continue with the above tutorial and touch on some
advanced techniques for modeling using compartments (`BioNetGen`_
only), the definition of higher order rules using functions, and model
calibration using the PySB utilities. Although we provide the
functions and utilities we have found useful for the community, we
encourage users to customize the modeling tools to their needs and
add/contribute to the **PySB** modeling community.


Higher-order rules
==================


Compartments
============
We will continue building on your :file:`mymodel.py` file and add one
more species and a compartment. In extrinsic apoptosis, once *tBid* is
activated it translocates to the outer mitochondrial membrane where it
interacts with the protein *Bak* (residing in the membrane). 



.. rubric:: Footnotes

.. [#func] Technically speaking it's a constructor, not just any old
   function.

.. [#mod] Python allows users to write python code such as **Pysb** code
   to a file and use this code later as an executable script or
   from an interactive instance. Such files are called *modules* and
   can be imported into a Python instance. See `Python modules
   <http://docs.python.org/tutorial/modules.html>'_ for details.

.. [#mkw] The astute Python programmer will recognize this as the
   ``repr`` of the monomer object, using keyword arguments in the
   constructor call.

.. [#sp] SciPy: http://www.scipy.org

.. [#pl] PyLab: http://www.scipy.org/PyLab

.. _BioNetGen: http://bionetgen.org/index.php/Documentation

.. _Kappa: http://www.kappalanguage.org/documentation

.. _extrinsic apoptosis signaling: http://www.plosbiology.org/article/info%3Adoi%2F10.1371%2Fjournal.pbio.0060299
