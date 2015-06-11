from pysb.integrate import Solver
from pysb.examples import robertson, earm_1_0
import numpy as np

def test_robertson_integration():
    t = np.linspace(0, 100)
    # Run with and without inline
    Solver._use_inline = False
    sol = Solver(robertson.model, t, use_jacobian=True)
    sol.run()
    assert sol.y.shape[0] == t.shape[0]
    # Run with and without inline
    Solver._use_inline = True
    sol = Solver(robertson.model, t, use_jacobian=True)
    sol.run()

def test_earm_integration():
    t = np.linspace(0, 1e4, 1000)
    # Run with and without inline
    Solver._use_inline = False
    sol = Solver(earm_1_0.model, t, use_jacobian=True)
    sol.run()
    Solver._use_inline = True
    sol = Solver(earm_1_0.model, t, use_jacobian=True)
    sol.run()

