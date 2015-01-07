"""Produce contact map for Figure 5D from the PySB publication"""

from __future__ import print_function
import pysb.integrate
import pysb.util
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import sys
import inspect

from earm.lopez_embedded import model


# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']

# Load experimental data file
data_path = os.path.join(os.path.dirname(__file__), 'fig6_data.csv')
exp_data = np.genfromtxt(data_path, delimiter=',', names=True)

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_data = np.array([9810.0, 180.0, 1.0])
momp_var = np.array([7245000.0, 3600.0, 1e-9])

# Build time points for the integrator, using the same time scale as the
# experimental data but with greater resolution to help the integrator converge.
ntimes = len(exp_data['Time'])
# Factor by which to increase time resolution
tmul = 10
# Do the sampling such that the original experimental timepoints can be
# extracted with a slice expression instead of requiring interpolation.
tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1],
                    (ntimes-1) * tmul + 1)
# Initialize solver object
solver = pysb.integrate.Solver(model, tspan, rtol=1e-5, atol=1e-5)

# Get parameters for rates only
rate_params = model.parameters_rules()
# Build a boolean mask for those params against the entire param list
rate_mask = np.array([p in rate_params for p in model.parameters])
# Build vector of nominal parameter values from the model
nominal_values = np.array([p.value for p in model.parameters])
# Set the radius of a hypercube bounding the search space
bounds_radius = 2


def objective_func(x, rate_mask, lb, ub):

    caller_frame, _, _, caller_func, _, _ = inspect.stack()[1]
    if caller_func in {'anneal', '_minimize_anneal'}:
        caller_locals = caller_frame.f_locals
        if caller_locals['n'] == 1:
            print(caller_locals['best_state'].cost, caller_locals['current_state'].cost)
        
    # Apply hard bounds
    if np.any((x < lb) | (x > ub)):
        print("bounds-check failed")
        return np.inf

    # Simulate model with rates taken from x (which is log transformed)
    param_values = np.array([p.value for p in model.parameters])
    param_values[rate_mask] = 10 ** x
    solver.run(param_values)

    # Calculate error for point-by-point trajectory comparisons
    e1 = 0
    for obs_name, data_name, var_name in zip(obs_names, data_names, var_names):
        # Get model observable trajectory (this is the slice expression
        # mentioned above in the comment for tspan)
        ysim = solver.yobs[obs_name][::tmul]
        # Normalize it to 0-1
        ysim_norm = ysim / np.nanmax(ysim)
        # Get experimental measurement and variance
        ydata = exp_data[data_name]
        yvar = exp_data[var_name]
        # Compute error between simulation and experiment (chi-squared)
        e1 += np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)

    # Calculate error for Td, Ts, and final value for IMS-RP reporter
    # =====
    # Normalize trajectory
    ysim_momp = solver.yobs[momp_obs]
    ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
    # Build a spline to interpolate it
    st, sc, sk = scipy.interpolate.splrep(solver.tspan, ysim_momp_norm)
    # Use root-finding to find the point where trajectory reaches 10% and 90%
    t10 = scipy.interpolate.sproot((st, sc-0.10, sk))[0]
    t90 = scipy.interpolate.sproot((st, sc-0.90, sk))[0]
    # Calculate Td as the mean of these times
    td = (t10 + t90) / 2
    # Calculate Ts as their difference
    ts = t90 - t10
    # Get yfinal, the last element from the trajectory
    yfinal = ysim_momp_norm[-1]
    # Build a vector of the 3 variables to fit
    momp_sim = [td, ts, yfinal]
    # Perform chi-squared calculation against mean and variance vectors
    e2 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3

    # Calculate error for final cPARP value (ensure all PARP is cleaved)
    cparp_final = model.parameters['PARP_0'].value
    cparp_final_var = .01
    cparp_final_sim = solver.yobs['cPARP'][-1]
    e3 = (cparp_final - cparp_final_sim) ** 2 / (2 * cparp_final_var)

    error = e1 + e2 + e3
    return error


def estimate(start_values=None):

    """Estimate parameter values by fitting to data.

    Parameters
    ==========
    parameter_values : numpy array of floats, optional
        Starting parameter values. Taken from model's nominal parameter values
        if not specified.

    Returns
    =======
    numpy array of floats, containing fitted parameter values.

    """

    # Set starting position to nominal parameter values if not specified
    if start_values is None:
        start_values = nominal_values
    else:
        assert start_values.shape == nominal_values.shape
    # Log-transform the starting position
    x0 = np.log10(start_values[rate_mask])
    # Displacement size for annealing moves
    dx = .02
    # The default 'fast' annealing schedule uses the 'lower' and 'upper'
    # arguments in a somewhat counterintuitive way. See
    # http://projects.scipy.org/scipy/ticket/1126 for more information. This is
    # how to get the search to start at x0 and use a displacement on the order
    # of dx (note that this will affect the T0 estimation which *does* expect
    # lower and upper to be the absolute expected bounds on x).
    lower = x0 - dx / 2
    upper = x0 + dx / 2
    # Log-transform the rate parameter values
    xnominal = np.log10(nominal_values[rate_mask])
    # Hard lower and upper bounds on x
    lb = xnominal - bounds_radius
    ub = xnominal + bounds_radius

    # Perform the annealing
    args = [rate_mask, lb, ub]
    (xmin, Jmin, Tfinal, feval, iters, accept, retval) = \
        scipy.optimize.anneal(objective_func, x0, full_output=True,
                              maxiter=4000, quench=0.5,
                              lower=lower, upper=upper,
                              args=args)
    # Construct vector with resulting parameter values (un-log-transformed)
    params_estimated = start_values.copy()
    params_estimated[rate_mask] = 10 ** xmin

    # Display annealing results
    for v in ('xmin', 'Jmin', 'Tfinal', 'feval', 'iters', 'accept', 'retval'):
        print("%s: %s" % (v, locals()[v]))

    return params_estimated


def display(params_estimated):

    # Simulate model with nominal parameters and construct a matrix of the
    # trajectories of the observables of interest, normalized to 0-1.
    solver.run()
    obs_names_disp = ['mBid', 'aSmac', 'cPARP']
    obs_totals = [model.parameters[n].value for n in ('Bid_0', 'Smac_0', 'PARP_0')]
    sim_obs = solver.yobs[obs_names_disp].view(float).reshape(len(solver.yobs), -1)
    sim_obs_norm = (sim_obs / obs_totals).T

    # Do the same with the estimated parameters
    solver.run(params_estimated)
    sim_est_obs = solver.yobs[obs_names_disp].view(float).reshape(len(solver.yobs), -1)
    sim_est_obs_norm = (sim_est_obs / obs_totals).T

    # Plot data with simulation trajectories both before and after fitting

    color_data = '#C0C0C0'
    color_orig = '#FAAA6A'
    color_est = '#83C98E'

    plt.subplot(311)
    plt.errorbar(exp_data['Time'], exp_data['norm_ICRP'],
                 yerr=exp_data['nrm_var_ICRP']**0.5, c=color_data, linewidth=2,
                 elinewidth=0.5)
    plt.plot(solver.tspan, sim_obs_norm[0], color_orig, linewidth=2)
    plt.plot(solver.tspan, sim_est_obs_norm[0], color_est, linewidth=2)
    plt.ylabel('Fraction of\ncleaved IC-RP/Bid', multialignment='center')
    plt.axis([0, 20000, -0.2, 1.2])

    plt.subplot(312)
    plt.vlines(momp_data[0], -0.2, 1.2, color=color_data, linewidth=2)
    plt.plot(solver.tspan, sim_obs_norm[1], color_orig, linewidth=2)
    plt.plot(solver.tspan, sim_est_obs_norm[1], color_est, linewidth=2)
    plt.ylabel('Td / Fraction of\nreleased Smac', multialignment='center')
    plt.axis([0, 20000, -0.2, 1.2])

    plt.subplot(313)
    plt.errorbar(exp_data['Time'], exp_data['norm_ECRP'],
                 yerr=exp_data['nrm_var_ECRP']**0.5, c=color_data, linewidth=2,
                 elinewidth=0.5)
    plt.plot(solver.tspan, sim_obs_norm[2], color_orig, linewidth=2)
    plt.plot(solver.tspan, sim_est_obs_norm[2], color_est, linewidth=2)
    plt.ylabel('Fraction of\ncleaved EC-RP/PARP', multialignment='center')
    plt.xlabel('Time (s)')
    plt.axis([0, 20000, -0.2, 1.2])

    plt.show()


if __name__ == '__main__':

    params_estimated = None
    try:
        earm_path = sys.modules['earm'].__path__[0]
        fit_file = os.path.join(earm_path, '..', 'EARM_2_0_M1a_fitted_params.txt')
        params_estimated = np.genfromtxt(fit_file)[:,1].copy()
    except IOError:
        pass
    if params_estimated is None:
        np.random.seed(1)
        params_estimated = estimate()
    display(params_estimated)
