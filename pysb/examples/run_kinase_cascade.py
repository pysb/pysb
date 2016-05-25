#!/usr/bin/env python

from __future__ import print_function
from pysb.simulator import ScipyOdeSimulator
from pylab import linspace, plot, legend, show

from kinase_cascade import model


tspan = linspace(0, 1200)
print("Simulating...")
yfull = ScipyOdeSimulator.execute(model, tspan=tspan)
plot(tspan, yfull['ppMEK'], label='ppMEK')
plot(tspan, yfull['ppERK'], label='ppERK')
legend(loc='upper left')
show()
