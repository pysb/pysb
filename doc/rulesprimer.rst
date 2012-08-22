.. _rules_primer:

About Rules
===========

Overview
--------

In rules-based modeling, units that undergo transformations such as
proteins, small molecules, protein complexes, etc are termed
*species*. The interactions among these *species* are then represented
using structured objects that describe the interactions between the
*species* and constitute what we describe as *rules*. The specific
details of how species and rules are specified can vary across
different rules-based modeling approaches. In PySB we have chosen to
ascribe to the approaches found in `BioNetGen`_ and `Kappa`_,
but other approaches are certainly possible for advanced users
interested in modifying the source codbe. Each rule, describing the
interaction between *species* or sets of *species* must be assigned a
set of *parameters* associated with the nature of the *rule*. Given
that `BioNetGen`_ and `Kappa`_ both describe interactions
using a mass-action kinetics formalism, the *parameters* will
necessarily consist of reaction rates. In what follows we describe how
a model can be instantiated in PySB, how *species* and *rules* are
specified, and how to run a simple simulation.


Reference to Rules-based languages
----------------------------------

PySB uses the rules languages grammar of `BioNetGen`_ and `Kappa`_ almost
verbatim with the differences being mostly synctactic. This has been
done on purpose to keep compatibility with these languages and
leverage their available simulating tools. We invited interested users
to explore the `BioNetGen Tutorial
<http://bionetgen.org/index.php/BioNetGen_Tutorial>`_ or the
`Introduction to Kappa Syntax <http://kappalanguage.org/syntax>`_ pages
for further information. Understanding of any of these languages will
make working with PySB rules a very straightforward exercise for the user.

.. _BioNetGen: http://bionetgen.org/index.php/Documentation

.. _Kappa: http://www.kappalanguage.org/documentation
