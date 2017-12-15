==========================
Frequently Asked Questions
==========================

Below are some of PySB's frequently asked questions. If your question is
not answered here, you can try our
`Gitter channel <https://gitter.im/pysb/pysb>`_.
For more general Python related questions, we recommend `Stack
Overflow <https://www.stackoverflow.com>`_.

General
=======

* Do you support Python 3?

    **Yes, with one caveat, which will be fixed in the next release**. PySB
    currently uses the `<>` operator for reversible rules, e.g.::

       Rule('bindA', A(b=None) + A(b=None) <> A(b=1) % A(b=1), kf, kr)

    PySB "overloads" this operator from base Python, but this operator
    (originally "not equals", a synonym for `!=`) was removed in Python 3.
    Thus, to use Python 3 with the current PySB release, you will have to
    write the above rule as the following::

        Rule('bindA', A(b=None) + A(b=None) != A(b=1) % A(b=1), kf, kr)

    However, following discussions among the core team, we have decided to
    replace the `<>` reversible rule operator in the next release of PySB
    with |. Thus, from the next release, reversible rules will be written
    as follows::

        Rule('bindA', A(b=None) + A(b=None) | A(b=1) % A(b=1), kf, kr)

Rule and Reaction Rate Laws
===========================

* Can I specify a non-mass action rate law?

    **Yes**. PySB has a special entity for this, called Expressions.
    Expressions can be used in place of Parameters for rule rates.
    Expressions can contain mathematical expressions and can utilize other
    Expressions, Parameters, and Observables. Here's a contrived
    example for demonstration purposes::

         Parameter('A_multiplier', 2.0)
         Observable('A_total', A())
         Expression('kf_A`, A_total * A_multiplier)
         Rule('bindA', A(b=None) + A(b=None) >> A(b=1) % A(b=1), kf_A)

    Like Parameters, note that Expressions are multiplied by
    reactant species concentrations within a rule to get the final rate.

* Can I use a discontinuous rate law, like a `Heaviside
  <https://en.wikipedia.org/wiki/Heaviside_step_function>`_ function?

    **Yes.** For simple examples like the Heaviside function, one could
    simply write a rate Expression like the following::

        Observable('A_total', A())
        Parameter('p1', 1.0)
        Expression('e1', (A_total > 100) * p1))

    The inequality in parentheses evaluates to 1 if True and 0 if False.
    Thus, the Expression will be equal to `p1` when `A_total > 100` and 0
    otherwise.

    For more complex piecewise expressions, sympy's Piecewise can be used::

        Expression('kf_A', Piecewise((0, A_total < 400.0),
                                     (0.001, A_total < 500.0),
                                     (0.01, True)))

    Piecewise takes a list of (value, condition) tuples. The Expression's
    value will come from the first condition which evaluates to True. Thus,
    for the Expression to always have a value, the last condition should
    default to True.

Simulation
==========

* How can I speed up my `ScipyOdeSimulator` simulation?

    **Check the weave library is installed.** `weave` is a Python library
    which converts your system of ordinary differential equations (ODEs) to
    C code, which is faster to execute than pure Python code. You can check
    if `weave` is installed by trying to import it at the Python prompt::

        import weave

    If no `ImportError` appears, `weave` is available. Note that `weave` is
    only available for Python 2. We are working on an alternative for Python
    3 using `Cython <http://cython.org>`_.

    When running large numbers of simulations, consider using the
    `CupSodaSimulator` if you have an NVIDIA graphics card (GPU) available.
    It is a GPU-based simulator which can run lots of simulations in parallel.
    See the :doc:`Simulator module documentation</modules/simulator>` for
    details.
