import numpy as np
from pysb.integrate import Solver
from itertools import compress


def pre_equilibration(model, time_search, tolerance=1e-6, ligand=None, ligand_value=None, parameters=None):
    """

    :param model: PySB model
    :param time_search: time span arry to be used to find the equilibrium
    :param tolerance: (tolerance, -tolerance) Range within equilibrium is considered as reached
    :param ligand: Species whose value want to be changed.
    :param ligand_value: Initial condition of ligand (usually zero)
    :param parameters: Model parameters
    :return:
    """
    if parameters is not None:
        # accept vector of parameter values as an argument
        if len(parameters) != len(model.parameters):
            raise Exception("param_values must be the same length as model.parameters")
        if not isinstance(parameters, np.ndarray):
            parameters = np.array(parameters)
    else:
        # create parameter vector from the values in the model
        parameters = np.array([p.value for p in model.parameters])

    param_dict = dict((p.name, parameters[i]) for i, p in enumerate(model.parameters))

    # Check if ligand and value of ligand to be used for pre equilibration are provided
    if ligand is not None and ligand_value is not None:
        if isinstance(ligand, str):
            param_dict[ligand] = ligand_value
    elif ligand is not None and ligand_value is None:
        if isinstance(ligand, str):
            param_dict[ligand] = 0

    # Solve system for the time span provided
    solver = Solver(model, time_search)
    solver.run(param_dict)
    y = solver.y.T
    dt = time_search[1] - time_search[0]

    time_to_equilibration = [0, 0]
    for idx, sp in enumerate(y):
        sp_eq = False
        derivative = np.diff(sp) / dt
        derivative_range = ((derivative < tolerance) & (derivative > -tolerance))
        # Indexes of values less than tolerance and greater than -tolerance
        derivative_range_idxs = list(compress(xrange(len(derivative_range)), derivative_range))
        for i in derivative_range_idxs:
            # Check if derivative is close to zero in the time points ahead
            if (derivative[i + 3] < tolerance) | (derivative[i + 3] > -tolerance):
                sp_eq = True
                if time_search[i] > time_to_equilibration[0]:
                    time_to_equilibration[0] = time_search[i]
                    time_to_equilibration[1] = i
            if not sp_eq:
                raise Exception('Equilibrium can not be reached within the time_search input')
            if sp_eq:
                break
        else:
            raise Exception('Species s{0} has not reached equilibrium'.format(idx))

    conc_eq = y[:, time_to_equilibration[1]]
    return time_to_equilibration, conc_eq
