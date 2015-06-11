from pysb.integrate import Solver
from pysb.examples.robertson import model
import numpy as np

def test_robertson_integration():
    t = np.linspace(0, 100)
    sol = Solver(model, t)
    sol.run()
    assert sol.y.shape[0] == t.shape[0]
