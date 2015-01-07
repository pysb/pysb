from __future__ import print_function
from scipy.integrate._ode import IntegratorBase
from pysundials import cvode as _cvode, nvecserial
import ctypes

def cvode_rhs_func(t, y, ydot, f_data):
    # take the values that cvode passes its rhs, and adapt them to what
    # scipy.integrate.ode's rhs function expects
    y = y.asarray()
    f_data = ctypes.py_object.from_address(f_data).value
    (f, f_params) = f_data[:]
    ydot[:] = f(t, y, *f_params)[:]
    return 0

class cvode(IntegratorBase):
    valid_methods = {
        'adams': _cvode.CV_ADAMS,
        'bdf': _cvode.CV_BDF,
        }
    valid_iterations = {
        'functional': _cvode.CV_FUNCTIONAL,
        'newton': _cvode.CV_NEWTON,
        }

    def __init__(self, method='adams', iteration='functional', rtol=1.0e-6, atol=1.0e-12):
        if method not in cvode.valid_methods:
            raise Exception("%s is not a valid value for method -- please use one of the following: %s" %
                            (method, [m for m in cvode.valid_methods]))
        if iteration not in cvode.valid_iterations:
            raise Exception("%s is not a valid value for iteration -- please use one of the following: %s" %
                            (iteration, [m for m in cvode.valid_iterations]))
        self.method = method
        self.iteration = iteration
        self.rtol = rtol
        self.atol = atol
        self.first_step = True

    def reset(self, n, has_jac):
        if has_jac:
            raise Exception("has_jac not yet supported")
        self.success = 1
        self.y = _cvode.NVector([0] * n)
        self.first_step = True
        # initialize the cvode memory object
        self.cvode_mem = _cvode.CVodeCreate(cvode.valid_methods[self.method],
                                            cvode.valid_iterations[self.iteration])
        # allocate memory for cvode (even though we need to call CVodeReAlloc right away on the
        # first call to run(), as it seems to be in the spirit of the scipy.integrate.ode design
        # that memory allocation happens here)
        # note that we don't know t0 here, so we pass 0.0 and reinit it later.
        _cvode.CVodeMalloc(self.cvode_mem, cvode_rhs_func, 0.0, self.y, _cvode.CV_SS, self.rtol, self.atol)
        # initialize the dense linear solver
        _cvode.CVDense(self.cvode_mem, n)

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.first_step:
            # copy initial state from input ndarray y0 to our nvector y
            self.y[:] = y0[:]
            # reinitialize cvode, now that we have func and t0, and actual values for y0
            _cvode.CVodeReInit(self.cvode_mem, cvode_rhs_func, t0, self.y, _cvode.CV_SS, self.rtol, self.atol)
            # tell cvode about our rhs function's user data
            f_data = ctypes.cast(ctypes.pointer(ctypes.py_object((f, f_params))), ctypes.c_void_p)
            _cvode.CVodeSetFdata(self.cvode_mem, f_data)
        tret = _cvode.realtype()
        flag = _cvode.CVode(self.cvode_mem, t1, self.y, tret, _cvode.CV_NORMAL)
        if flag < 0:
            self.success = 0
            print("cvodes error: %d (see SUNDIALS manual for more information)" % flag)
        return (self.y, t1)

IntegratorBase.integrator_classes.append(cvode)
