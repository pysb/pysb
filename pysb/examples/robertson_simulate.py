# Integrates Robertson's example, as defined in robertson.py, and
# plots the trajectories.

from pylab import *
from pysb.integrate import Solver

from robertson import model

# solve from t=0 to t=40
t = linspace(0, 40)

solver = Solver(model, t, rtol=1e-4, atol=[1e-8, 1e-14, 1e-6])
solver.run()
y = solver.y
# plot trajectories, each normalized to the range 0-1
p = plot(t, y / y.max(0))
figlegend(p, ['y1', 'y2', 'y3'], 'upper right')
show()
