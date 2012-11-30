#!/usr/bin/env python

from pysb.integrate import odesolve
from pylab import linspace, plot, legend, show

from kinase_cascade import model


tspan = linspace(0, 1200)
print "Simulating..."
yfull = odesolve(model, tspan)
plot(tspan, yfull['ppMEK'], label='ppMEK')
plot(tspan, yfull['ppERK'], label='ppERK')
legend(loc='upper left')
show()
