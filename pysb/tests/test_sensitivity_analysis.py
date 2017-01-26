from pysb.tools.sensitivity_analysis import \
    InitialConcentrationSensitivityAnalysis
from pysb.examples.tyson_oscillator import model
import numpy as np
from pysb.simulator.scipyode import ScipyOdeSimulator
import numpy.testing as npt
import os


class TestSensitivityAnalysis(object):
    def setUp(self):
        self.tspan = np.linspace(0, 200, 5001)
        self.observable = 'Y3'
        self.savename = 'sens_out_test'
        self.output_dir = 'sens_out'
        self.vals = np.linspace(.8, 1.2, 5)
        self.sens = InitialConcentrationSensitivityAnalysis(
            model, self.tspan, self.vals, self.obj_func_cell_cycle,
            self.observable, )

        self.p_simulated = np.array(
            [[0., 0., 0., 0., 0., 5.0301, 2.6027, 0., -2.5118, -4.5758],
             [0., 0., 0., 0., 0., 5.0301, 2.5832, 0., -2.5313, -4.5823],
             [0., 0., 0., 0., 0., 5.0301, 2.5767, 0., -2.5313, -4.5953],
             [0., 0., 0., 0., 0., 5.0301, 2.5767, 0., -2.5313, -4.6082],
             [0., 0., 0., 0., 0., 5.0301, 2.5767, 0., -2.5313, -4.60829],
             [5.0301, 5.0301, 5.0301, 5.0301, 5.0301, 0., 0., 0., 0., 0.],
             [2.6027, 2.5832, 2.5767, 2.5767, 2.5767, 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [-2.5118, -2.5313, -2.5313, -2.5313, -2.5313, 0., 0., 0., 0., 0.],
             [-4.5758, -4.5823, -4.5953, -4.6082, -4.6082, 0., 0., 0., 0.,
              0.]])
        self.sens.run(self.run_solver, save_name=self.savename,
                      out_dir=self.output_dir)

    def obj_func_cell_cycle(self, out):
        timestep = self.tspan[:-1]
        y = out[:-1] - out[1:]
        freq = 0
        local_times = []
        prev = y[0]
        for n in range(1, len(y)):
            if y[n] > 0 > prev:
                local_times.append(timestep[n])
                freq += 1
            prev = y[n]

        local_times = np.array(local_times)
        local_freq = np.average(local_times) / len(local_times) * 2
        return local_freq

    def run_solver(self, matrix):
        size_of_matrix = len(matrix)
        solver = ScipyOdeSimulator(model, self.tspan, integrator='lsoda',
                                   integrator_options={'rtol': 1e-8,
                                                       'atol': 1e-8,
                                                       'mxstep': 20000})
        sensitivity_matrix = np.zeros((len(self.tspan), size_of_matrix))
        for k in range(size_of_matrix):
            traj = solver.run(initials=matrix[k, :])
            sensitivity_matrix[:, k] = traj.observables[self.observable]
        return sensitivity_matrix

    def test_p_matrix_shape(self):
        assert self.sens.p_matrix.shape == (10, 10)

    def test_p_matrix(self):
        npt.assert_almost_equal(self.sens.p_matrix, self.p_simulated,
                                decimal=3)

    def test_num_simulations(self):
        assert len(self.sens.simulations) == 25

    def test_pmatrix_outfile_exists(self):
        outfile = os.path.join(self.output_dir,
                               '{}_p_matrix.csv'.format(self.savename))
        assert os.path.exists(outfile)

    def test_create_png(self):
        self.sens.create_boxplot_and_heatplot(save_name='test',
                                              out_dir=self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir, 'test.png'))
