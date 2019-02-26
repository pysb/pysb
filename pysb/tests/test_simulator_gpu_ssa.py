from pysb.simulator.gpu_ssa import GPUSimulator
from pysb.examples.schlogl import model
import numpy as np
from nose.plugins.attrib import attr


@attr('gpu')
class TestGpu(object):

    def setUp(self):
        self.tspan = np.linspace(0, 100, 101)
        model.parameters['X_0'].value = 400
        self.simulator = GPUSimulator(model)

    def test_multiple_call_host(self):
        self.simulator.run(self.tspan, number_sim=100)


if __name__ == '__main__':
    t = TestGpu()
    t.setUp()
    t.test_multiple_call_host()
    t.test_one_single_call_host()
