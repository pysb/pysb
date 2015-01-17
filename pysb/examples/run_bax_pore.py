#!/usr/bin/env python
"""Simulate the bax_pore model and plot the results."""

from __future__ import print_function
from pylab import *
from pysb.integrate import odesolve

from bax_pore import model


t = linspace(0, 100)
print("Simulating...")
x = odesolve(model, t)

p = plot(t, c_[x['BAX4'], x['BAX4_inh']])
figlegend(p, ['BAX4', 'BAX4_inh'], 'upper left')
show()
