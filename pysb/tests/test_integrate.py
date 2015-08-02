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
    
@with_model
def test_y0_as_dict():

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
    solver = Solver(model, time, verbose=True)
    solver.run(y0={A(a=None):100, B(b=1) % A(a=1):100, B(b=None):50})

@with_model
def test_param_values_as_dict():

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
    solver = Solver(model, time, verbose=True)
    solver.run(param_values={'ksynthA':150})
