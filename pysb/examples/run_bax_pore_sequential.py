#!/usr/bin/env python
"""Simulate the bax_pore_sequential model and plot the results."""

from __future__ import print_function
from pylab import *
from pysb.integrate import odesolve

from bax_pore_sequential import model, max_size


# System is very stiff, and using logspace instead of linspace to produce the
# vector of time points happens to help with the integration
t = logspace(-3, 5) # 1e-3 to 1e5
print("Simulating...")
x = odesolve(model, t)

# Plot trajectory of each pore
for i in range(1, max_size + 1):
    observable = 'Bax%d' % i
    # Map pore size to the central 50% of the YlOrBr color map
    color = cm.YlOrBr(float(i) / max_size / 2 + 0.25)
    plot(t, x[observable], c=color, label=observable)
# Plot Smac species
plot(t, x['mSmac'], c='magenta', label='mSmac')
plot(t, x['cSmac'], c='cyan', label='cSmac')

# Expand the limits a bit to show the min/max levels clearly
ylim([-0.01e5, 1.01e5])
# Show time on a log scale 
xscale('log')
legend(loc='upper right')
show()
