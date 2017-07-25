from pysb.bng import generate_equations
from .egfr_extended import model


class BngNetworkGeneration(object):
    timeout = 300

    def setup(self):
        model.reset_equations()

    def time_egfr_equations_max_iter_8(self):
        generate_equations(model, max_iter=8)
        # Check model ODEs are generated
        [x for x in model.odes]
