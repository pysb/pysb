from pysb.simulator import KappaSimulator
from pysb.kappa import run_simulation
from pysb.examples import michment
from pysb.bng import generate_equations
import numpy as np
from nose.tools import raises


KAPPA_SEED = 123


def _compare_kappa_sims(run_simulation_output, kappasimulator_output):
    sim1 = run_simulation_output.view('<f8')[:, 1:]
    sim2 = kappasimulator_output.observables.view('f8').reshape(sim1.shape)

    assert np.allclose(sim1, sim2, atol=1e-16, rtol=1e-16)


def test_kappa_sim_michment():
    perturbation = '%init: 12 E(s[.])'

    orig_sim = run_simulation(michment.model, time=100, points=100,
                              seed=KAPPA_SEED, perturbation=perturbation)

    sim = KappaSimulator(michment.model, tspan=np.linspace(0, 100, 101))
    x = sim.run(seed=KAPPA_SEED, perturbation=perturbation)

    _compare_kappa_sims(orig_sim, x)


@raises(ValueError)
def test_kappa_sim_invalid_arg():
    sim = KappaSimulator(michment.model, tspan=range(10))
    sim.run(spam='eggs')


def test_kappa_2sims():
    sim = KappaSimulator(michment.model, tspan=np.linspace(0, 100, 101))
    res = sim.run(n_runs=2, seed=KAPPA_SEED)
    assert res.nsims == 2


def test_kappa_2initials():
    sim = KappaSimulator(michment.model, tspan=np.linspace(0, 100, 101))
    res = sim.run(initials={
        michment.model.initials[0].pattern: [10, 100],
        michment.model.initials[1].pattern: [100, 1000]
    }, seed=KAPPA_SEED)
    assert res.nsims == 2


def test_kappa_2params():
    base_param_values = np.array(
        [p.value for p in michment.model.parameters] * 2
    ).reshape(2, len(michment.model.parameters))
    sim = KappaSimulator(michment.model, tspan=np.linspace(0, 100, 101))

    res = sim.run(param_values=base_param_values, seed=KAPPA_SEED)
    assert res.nsims == 2
    assert res.dataframe.loc[0].equals(res.dataframe.loc[1])


def test_kappa_1timepoint():
    michment.model.reset_equations()
    sim = KappaSimulator(michment.model, tspan=[0, 1])
    # This set of parameter values causes Kappa to abort after 1st time point
    res = sim.run(param_values=[100, 100, 10, 10, 100, 100], seed=KAPPA_SEED)
    assert res.dataframe.shape == (1, 4), res.dataframe.shape


def test_kappa_1timepoint_with_netgen():
    generate_equations(michment.model)
    sim = KappaSimulator(michment.model, tspan=[0, 1])
    # This set of parameter values causes Kappa to abort after 1st time point
    res = sim.run(param_values=[100, 100, 10, 10, 100, 100], seed=KAPPA_SEED)
    assert res.dataframe.shape == (1, 4), res.dataframe.shape