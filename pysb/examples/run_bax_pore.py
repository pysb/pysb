"""Simulate the bax_pore model and plot the results."""

from __future__ import print_function
import matplotlib.pyplot as plt
from numpy import linspace
from pysb.simulator import ScipyOdeSimulator

from bax_pore import model


t = linspace(0, 100)
print("Simulating...")
x = ScipyOdeSimulator(model).run(tspan=t).all

plt.plot(t, x['BAX4'])
plt.plot(t, x['BAX4_inh'])
plt.legend(['BAX4', 'BAX4_inh'], loc='upper left')
plt.show()
