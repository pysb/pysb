from pysb.tools.sensitivity_analysis import \
    PairwiseSensitivity, InitialsSensitivity
from pysb.examples.tyson_oscillator import model
import numpy as np
import numpy.testing as npt
import os
from pysb.simulator.scipyode import ScipyOdeSimulator
import tempfile
import shutil
from nose.tools import raises


class TestSensitivityAnalysis(object):
    def setUp(self):
        self.tspan = np.linspace(0, 200, 5001)
        self.observable = 'Y3'
        self.savename = 'sens_out_test'
        self.output_dir = tempfile.mkdtemp()
        self.vals = np.linspace(.8, 1.2, 5)
        self.vals = [.8, .9, 1., 1.1, 1.2]
        self.model = model
        self.solver = ScipyOdeSimulator(self.model,
                                        tspan=self.tspan,
                                        integrator='lsoda',
                                        integrator_options={'rtol': 1e-8,
                                                            'atol': 1e-8,
                                                            'mxstep': 20000})
        self.sens = PairwiseSensitivity(
            solver=self.solver,
            values_to_sample=self.vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable
        )

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

        self.sens.run()

    def tearDown(self):
        shutil.rmtree(self.output_dir)

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

    def test_run(self):

        sens_vode = PairwiseSensitivity(
            solver=self.solver,
            values_to_sample=self.vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable
        )
        sens_vode.run()
        npt.assert_almost_equal(self.sens.p_matrix, self.p_simulated,
                                decimal=3)

    def test_old_class_naming(self):
        sens_vode = InitialsSensitivity(
            solver=self.solver,
            values_to_sample=self.vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable
        )
        sens_vode.run()
        npt.assert_almost_equal(self.sens.p_matrix, self.p_simulated,
                                decimal=3)

    def test_p_matrix_shape(self):
        assert self.sens.p_matrix.shape == (10, 10)

    def test_p_matrix(self):
        npt.assert_almost_equal(self.sens.p_matrix, self.p_simulated,
                                decimal=3)

    def test_pmatrix_outfile_exists(self):
        self.sens.run(save_name=self.savename,
                      out_dir=self.output_dir)
        assert os.path.exists(os.path.join(
            self.output_dir, '{}_p_matrix.csv'.format(self.savename)
        ))
        assert os.path.exists(os.path.join(
            self.output_dir, '{}_p_prime_matrix.csv'.format(self.savename)
        ))

    def test_create_png(self):
        self.sens.create_boxplot_and_heatplot(save_name='test',
                                              out_dir=self.output_dir)

        assert os.path.exists(os.path.join(self.output_dir, 'test.png'))
        assert os.path.exists(os.path.join(self.output_dir, 'test.eps'))
        assert os.path.exists(os.path.join(self.output_dir, 'test.svg'))

        self.sens.create_individual_pairwise_plots(save_name='test2',
                                                   out_dir=self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir,
                                           'test2_subplots.png'))

        self.sens.create_plot_p_h_pprime(save_name='test3',
                                         out_dir=self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir,
                                           'test3_P_H_P_prime.png'))

    def test_unique_simulations_only(self):
        vals = [.8, .9, 1.1, 1.2, 1.3]
        sens = PairwiseSensitivity(
            values_to_sample=vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable,
            solver=self.solver
        )
        sens.run()
        self.sens.create_plot_p_h_pprime(save_name='test4',
                                         out_dir=self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir,
                                           'test4_P_H_P_prime.png'))

    def test_param_pair(self):
        vals = [.9, 1.0, 1.1]
        sens = PairwiseSensitivity(
            values_to_sample=vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable,
            solver=self.solver,
            sample_list=['k1', 'cdc0']
        )
        sens.run()
        self.sens.create_plot_p_h_pprime(save_name='test4',
                                         out_dir=self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir,
                                           'test4_P_H_P_prime.png'))

    def test_all_params(self):
        vals = [.9, 1.1]

        sens = PairwiseSensitivity(
            values_to_sample=vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable,
            solver=self.solver,
            sens_type='all'
        )
        sens.run()
        self.sens.create_plot_p_h_pprime(save_name='test4',
                                         out_dir=None)
        assert os.path.exists('test4_P_H_P_prime.png')

    @raises(ValueError)
    def test_param_not_in_model(self):
        vals = [.8, .9, 1.1, 1.2, 1.3]
        solver = ScipyOdeSimulator(self.model,
                                   tspan=self.tspan,
                                   integrator='lsoda',
                                   integrator_options={'rtol': 1e-8,
                                                       'atol': 1e-8,
                                                       'mxstep': 20000})
        sens = PairwiseSensitivity(
            values_to_sample=vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable,
            solver=solver, sample_list=['a0']
        )

    @raises(ValueError)
    def test_sens_type_and_list_none(self):
        vals = [.8, .9, 1.1, 1.2, 1.3]
        sens = PairwiseSensitivity(
            values_to_sample=vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable,
            solver=self.solver, sample_list=None, sens_type=None
        )

    @raises(TypeError)
    def test_bad_solver(self):
        vals = [.8, .9, 1.1, 1.2, 1.3]
        sens = PairwiseSensitivity(
            values_to_sample=vals,
            objective_function=self.obj_func_cell_cycle,
            observable=self.observable,
            solver=None, sample_list=None, sens_type=None
        )
