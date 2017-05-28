from pysb import *
from pysb.testing import *
from pysb.bng import *
from pysb.integrate import Solver
import numpy as np
from sympy import Piecewise


@with_model
def test_ic_expression_with_two_parameters():
    Monomer('A')
    Parameter('k1', 1)
    Parameter('k2', 2)
    Expression('e1', k1*k2)
    Rule('A_deg', A() >> None, k1)
    Initial(A(), e1)
    generate_equations(model)
    t = np.linspace(0, 1000, 100)
    sol = Solver(model, t, use_analytic_jacobian=True)
    sol.run()

@with_model
def test_ic_expression_with_one_parameter():
    Monomer('A')
    Parameter('k1', 1)
    Expression('e1', k1)
    Rule('A_deg', A() >> None, k1)
    Initial(A(), e1)
    generate_equations(model)
    t = np.linspace(0, 1000, 100)
    sol = Solver(model, t, use_analytic_jacobian=True)
    sol.run()

@with_model
def test_expressions_with_one_observable():
    Monomer('A')
    Parameter('k1', 1)
    Observable('o1', A())
    Expression('e1', o1)
    Rule('A_deg', A() >> None, k1)
    Initial(A(), k1)
    generate_equations(model)
    t = np.linspace(0, 1000, 100)
    sol = Solver(model, t, use_analytic_jacobian=True)
    sol.run()

@with_model
def test_nested_expression():
    Monomer(u'A')
    Monomer(u'B')
    Parameter(u'k1', 1)
    Observable(u'o1', B())
    Expression(u'e1', o1*k1)
    Expression(u'e2', e1+5)
    Initial(A(), Parameter(u'A_0', 1000))
    Initial(B(), Parameter(u'B_0', 1))
    Rule(u'A_deg', A() >> None, e2)
    generate_equations(model)
    t = np.linspace(0, 1000, 100)
    sol = Solver(model, t, use_analytic_jacobian=True)
    sol.run()

@with_model
def test_piecewise_expression():
    Monomer('A')
    Observable('A_total', A())
    Expression('A_deg_expr', Piecewise((0, A_total < 400.0),
                                       (0.001, A_total < 500.0),
                                       (0.01, True)))
    Initial(A(), Parameter('A_0', 1000))
    Rule('A_deg', A() >> None, A_deg_expr)
    generate_equations(model)
    t = np.linspace(0, 1000, 100)
    sol = Solver(model, t, use_analytic_jacobian=True)
    sol.run()
