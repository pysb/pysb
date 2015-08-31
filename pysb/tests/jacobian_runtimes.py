"""Run EARM and Robertson example models and compare runtimes with and
without inline and with and without the analytically-derived Jacobian."""

import timeit
from pysb.integrate import Solver
from pysb.examples import robertson, earm_1_0
import numpy as np

def check_runtime(model, tspan, iterations, use_inline, use_analytic_jacobian):
    Solver._use_inline = use_inline
    sol = Solver(model, tspan, use_analytic_jacobian=use_analytic_jacobian)
    start_time = timeit.default_timer()
    for i in range(iterations):
        sol.run()
    elapsed = timeit.default_timer() - start_time
    print("use_inline=%s, use_analytic_jacobian=%s, %d iterations" %
          (use_inline, use_analytic_jacobian, iterations))
    print("Time: %f sec\n" % elapsed)

if __name__ == '__main__':
    arg_list = [(0, 0), (0, 1), (1, 0), (1, 1)]

    print("-- EARM --")
    earm_tspan = np.linspace(0, 1e4, 1000)
    for args in arg_list:
        check_runtime(earm_1_0.model, earm_tspan, 1000, *args)

    print("-- Robertson --")
    rob_tspan = np.linspace(0, 100)
    for args in arg_list:
        check_runtime(robertson.model, rob_tspan, 5000, *args)
