from pylab import *
from pysb.integrate import odesolve

from bax_tetramer import model


t = linspace(0, 5e4)
x = odesolve(model, t)

p = plot(t, c_[x['BAX4'], x['BAX4_inh']])
figlegend(p, ['BAX4', 'BAX4_inh'], 'upper left')
show()
