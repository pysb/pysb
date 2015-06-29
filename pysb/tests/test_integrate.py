from pysb.testing import *
from pysb.core import *
from pysb.macros import *
from pysb.integrate import odesolve
from pylab import linspace, plot, xlabel, ylabel, show

@with_model
def test_integrate_with_expression():
    #Declaring the monomers
    Monomer('s1')
    Monomer('s16')
    Monomer('s20')

    #Declaring parameters
    Parameter('ka',2e-05)
    Parameter('ka20', 10^5)

    #Declaring expression
    Expression('keff', ka*ka20/(ka20+10000))

    #Declaring rules
    Rule('None_s16', None >> s16(), ka)
    Rule('None_s20', None >> s20(), ka)
    Rule('s16_and_s20', s16() + s20() >> s16() + s1(), keff)

    #Declaring observables
    Observable('s1_obs', s1())
    Observable('s16_obs', s16())
    Observable('s20_obs', s20())


    time = linspace(0, 40, 100)
    x = odesolve(model, time, verbose=True)