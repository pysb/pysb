from pysb.tools.sensitivity_analysis import \
    InitialConcentrationSensitivityAnalysis
from pysb.examples.tyson_oscillator import model
import numpy as np
from pysb.simulator.scipyode import ScipyOdeSimulator
import numpy.testing as npt




def obj_func_cell_cycle(out):
    timestep = tspan[:-1]
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

tspan = np.linspace(0, 200, 5001)
observable = 'Y3'
savename = 'test'
output_dir = 'test'
vals = np.linspace(.8, 1.2, 5)
sens = InitialConcentrationSensitivityAnalysis(model, tspan, vals,
                                               obj_func_cell_cycle,
                                               observable, )
p_simulated = np.array(
        [[0., 0., 0., 0., 0., 5.0301, 2.6027, 0., -2.5118, -4.5758],
         [0., 0., 0., 0., 0., 5.0301, 2.5832, 0., -2.5313, -4.5823],
         [0., 0., 0., 0., 0., 5.0301, 2.5767, 0., -2.5313, -4.5953],
         [0., 0., 0., 0., 0., 5.0301, 2.5767, 0., -2.5313, -4.6082],
         [0., 0., 0., 0., 0., 5.0301, 2.5767, 0., -2.5313, -4.60829],
         [5.0301, 5.0301, 5.0301, 5.0301, 5.0301, 0., 0., 0., 0., 0.],
         [2.6027, 2.5832, 2.5767, 2.5767, 2.5767, 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [-2.5118, -2.5313, -2.5313, -2.5313, -2.5313, 0., 0., 0., 0., 0.],
         [-4.5758, -4.5823, -4.5953, -4.6082, -4.6082, 0., 0., 0., 0., 0.]])


def run_solver(matrix):
    size_of_matrix = len(matrix)
    solver = ScipyOdeSimulator(model, tspan, integrator='lsoda',
                               mxstep=20000, rtol=1e-8, atol=1e-8, )
    sensitivity_matrix = np.zeros((len(tspan), size_of_matrix))
    for k in range(size_of_matrix):
        traj = solver.run(initials=matrix[k, :])
        sensitivity_matrix[:, k] = traj.observables[observable]
    return sensitivity_matrix


def test_p_matrix_shape():
    sens.run(run_solver, save_name=savename, out_dir=output_dir)
    assert sens.p_matrix.shape == (10, 10)


def test_p_matrix():
    sens.run(run_solver, save_name=savename, out_dir=output_dir)
    npt.assert_almost_equal(sens.p_matrix, p_simulated, decimal=4)


def test_num_simulations():
    sens.run(run_solver, save_name=savename, out_dir=output_dir)
    assert len(sens.simulations) == 25


test_p_matrix()
test_num_simulations()
