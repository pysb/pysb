Introduction to modeling
========================

The premise to all modeling is that, based on experimental
observations, we believe that a set of rules gobvern the behavior of a
system. Modeling the behavior of such a system involves the
*elucidation* of the rules that govern the system to understand our
observations and the *use* of such rules to further predict the
behavior of a system under a range of conditions. Thus, models can be
used to both *explain* or *predict* the behavior of a system given a
set of conditions. The best models can perform both tasks
satisfactorily. A simple example is the colloquial story of Newton's
apple. The observation was that the apple fell on Newton's head. He
derived the simple yet incredibly powerful :math:`F=ma` whereby he
observed that the Force, :math:`F`, applied to an object of mass,
:mass:`m`, resulted in an acceleration, :math:`a`. We now know that
this model holds for most conditions in every-day activities but we
know that it fails for e.g. relativistic effects. Therefore a model
has a domain of application and a limited usefulness. However, a
successful model can be employed accurately for both the explanation
and the prediction of a system. In the case of cell-molecular biology,
we aim to develop models that describe the behavior of cellular
systems. The model can guide us to understand what are our gaps in the
observations that prevent us from generalizing a theory and, when they
capture the key significant aspects of the behavior of a system,
predict the outcome of the behavrio of a system under a given set of
conditions. 

PySB as modeling tool
---------------------

PySB is a set of software tools that enables users to develop,
implement, and execute biological models in the Python programming
environment. One of the main advantages of PySB is that it leverages
the power of a very powerful programming language to express
biological concepts as parts of a program. The properties of the
programming environment are therefore the same properties found in
PySB. Python is an object-oriented programming language that provides
a useful environment for programming techniques such as data
abstraction, encapsulation, modularity, message-passing, polymorphism,
and inheritance to name a few. In addition to these technical
advantages, we chose Python due to its readable and clear syntax. In
our view, one of the most difficult issues with current biological
modeling is shareability and transparency, both of which are
addresssed, at least in part, by a clear syntax and a programmatic
flow of ideas. PySB can handle simple models, modular models, and
multiple instances of models, as shown in the tutorial. We invite
users to contribute and share their innovations and ideas to make PySB
a better open-source tool for the programming community. 


A quick example
---------------

Using and running PySB can be as simple as typing the following
commands in your Python shell. Go to the directory containing the file
simplemodel.py (usually pysb/examples) and try this at your shell!::

   [host] > python earm_figures.py

You will see some feedback from the machine, depending on your
operating system (and assuming PySB is correctly installed). After a
few seconds of calculations you should get two figures. The first
figure shows the experimental death time determined form experiments
(as dots with error bars) followed by the model-predicted average
(solid line) and the standard deviation ranges (dashed lines). The
second graph will show you the model signatures of three species,
namely initiator caspase (IC) substrate, effector caspase (EC)
substrate, and mitochondrial outer membrane permeabilization (MOMP) as
indicated by release of Smac to the cytosol. You have now run a model!
Feel free to open the files :file:`earm_1_0.py` to see a simple model
instantiation and :file:`earm_figures.py` to see how the model is run
and the figures are generated. If you want to learn how to build
biological models in a systematic (and we think fun) way, visit our
:doc:`tutorial`.

Conversion from other modeling tools
------------------------------------

**What should we say here?**
Here we give some pointers for people coming from SBML, BNG/Kappa,
Matlab, etc. in order to start converting their models and scripts to
work with PySB.  Mostly just a list of resources, not full
explanations.  This will help assure users of those tools that they
can relatively easily carry over their current modeling investments.
 
