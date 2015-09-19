from pysb import *
from pysb.testing import *
from pysb.bng import *
from pysb.integrate import Solver
import numpy as np

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
