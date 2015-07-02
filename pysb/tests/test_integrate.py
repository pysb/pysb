from pysb.testing import *
from pysb.core import *
from pysb.macros import *
from pysb.integrate import odesolve
from pylab import linspace, plot, xlabel, ylabel, show
from sympy import sympify

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
    