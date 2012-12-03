#!/usr/bin/env python
"""Integrates Robertson's example, as defined in robertson.py, and plots the
trajectories.
"""

from pylab import *
from pysb.integrate import odesolve

from robertson import model

# We will integrate from t=0 to t=40
t = linspace(0, 40)
# Simulate the model
print "Simulating..."
y = odesolve(model, t, rtol=1e-4, atol=[1e-8, 1e-14, 1e-6])
# Gather the observables of interest into a matrix
yobs = array([y[obs] for obs in ('A_total', 'B_total', 'C_total')]).T
# Plot normalized trajectories
plot(t, yobs / yobs.max(0))
legend(['y1', 'y2', 'y3'], 'lower right')
show()
