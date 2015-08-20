from pysb.testing import *
from pysb.core import *
from pysb.macros import *
from pysb.integrate import odesolve
from pylab import linspace, plot, xlabel, ylabel, show
from sympy import sympify

from pysb import *
from pysb.integrate import *
from pysb.bng import run_ssa
from pysb.macros import synthesize
import matplotlib.pyplot as plt
import numpy as np
from unittest import TestCase

@with_model
def test_integrate_with_expression():

    Monomer('s1')
    Monomer('s9')
    Monomer('s16')
    Monomer('s20')

    Parameter('ka',2e-5)
    Parameter('ka20', 1e5)
    
    Initial(s9(), Parameter('s9_0', 10000))
    
    Observable('s1_obs', s1())
    Observable('s9_obs', s9())
    Observable('s16_obs', s16())
    Observable('s20_obs', s20())
    
    Expression('keff', sympify("ka*ka20/(ka20+s9_obs)"))

    Rule('R1', None >> s16(), ka)
    Rule('R2', None >> s20(), ka)
    Rule('R3', s16() + s20() >> s16() + s1(), keff)

    time = linspace(0, 40, 100)
    x = odesolve(model, time, verbose=True)


class SolverTests(TestCase):
    @with_model
    def setUp(self):
        """ Setup for Solver() tests """

        Monomer('A', ['a'])
        Monomer('B', ['b'])

        Parameter('ksynthA', 100)
        Parameter('ksynthB', 100)
        Parameter('kbindAB', 100)

        Parameter('A_init', 0)
        Parameter('B_init', 0)

        Initial(A(a=None), A_init)
        Initial(B(b=None), B_init)

        Observable("A_free", A(a=None))
        Observable("B_free", B(b=None))
        Observable("AB_complex", A(a=1) % B(b=1))

        synthesize(A(a=None), ksynthA)
        synthesize(B(b=None), ksynthB)
        Rule('AB_bind', A(a=None) + B(b=None) >> A(a=1) % B(b=1), kbindAB)

        time = np.linspace(0, 0.005, 101)
        self.solver = Solver(model, time, verbose=False)

    def test_solver_run(self):
        """ Test solver.run() with no arguments """
        self.solver.run()

    def test_y0_as_dictionary_defined_species(self):
        """ Test solver.run() with y0 as a dictionary """
        self.solver.run(y0={"A(a=None)": 10, "B(b=1) % A(a=1)": 0,
                        "B(b=None)": 0})
        assert np.abs(self.solver.y[0,0] == 10)
    
    def test_y0_as_dictionary_with_undefined_species(self):
        """ Test solver.run() with y0 as a dictionary """
        self.solver.run(y0={"A(a=None)": 0, "B(b=1) % A(a=1)": 100,
                        "B(b=None)": 0})
        assert np.abs(self.solver.y[0, 3] == 100)
        

    @raises(IndexError)
    def test_y0_invalid_dictionary_key(self):
        """ Test solver.run() using y0 dictionary with invalid monomer name """
        self.solver.run(y0={"C(c=None)": 1})

    @raises(ValueError, TypeError)
    def test_y0_non_numeric_value(self):
        """ Test solver.run() using y0 dictionary with a non-numeric value """
        self.solver.run(y0={"A(a=None)": 'eggs'})

    def test_param_values_as_dictionary(self):
        """ Test solver.run() with param_values as a dictionary """
        self.solver.run(param_values={'kbindAB': 0})
        # kbindAB=0 should ensure no AB_complex is produced
        assert all(self.solver.yobs["AB_complex"] < 1e-8)

    @raises(IndexError)
    def test_param_values_invalid_dictionary_key(self):
        """ Test solver.run() using param_values dictionary with invalid
               monomer name """
        self.solver.run(param_values={'spam': 150})

    @raises(ValueError, TypeError)
    def test_param_values_non_numeric_value(self):
        """ Test solver.run() using param_values dict with a non-numeric
               value """
        self.solver.run(param_values={'ksynthA': 'eggs'})