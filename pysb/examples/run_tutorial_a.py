from __future__ import print_function
from pysb.integrate import Solver
from tutorial_a import model

t = [0, 10, 20, 30, 40, 50, 60]
solver = Solver(model, t)
solver.run()
print(solver.y[:, 1])
