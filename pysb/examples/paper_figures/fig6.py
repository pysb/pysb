import pysb.integrate
import pysb.util
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import inspect

from earm.lopez_embedded import model


# List of model observables and corresponding data file columns for
# point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']

# Load experimental data file
earm_path = os.path.dirname(__file__)
data_path = os.path.join(earm_path, 'xpdata', 'forfits',
                         'EC-RP_IMS-RP_IC-RP_data_for_models.csv')
exp_data = np.genfromtxt(data_path, delimiter=',', names=True)

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory)
momp_data = np.array([9810.0, 180.0, 1.0])
#momp_var = np.array([7245000.0, 3600.0, 1e-9])
momp_var = np.array([72450.0, 3600.0, 1e-9])

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
bounds_radius = 2  # TODO remove bounds checking entirely?


def objective_func(x, rate_mask, lb, ub):

    caller_frame, _, _, caller_func, _, _ = inspect.stack()[1]
    if caller_func == 'anneal':
        caller_locals = caller_frame.f_locals
        if caller_locals['n'] == 1:
            print caller_locals['best_state'].cost, caller_locals['current_state'].cost
        
    # Apply hard bounds
    if np.any((x < lb) | (x > ub)):
        print "bounds-check failed"
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
        print "%s: %s" % (v, locals()[v])

    return params_estimated


def display(params_estimated):

    # Construct matrix of experimental data and variance columns of interest
    exp_obs_norm = exp_data[data_names].view(float).reshape(len(exp_data), -1).T
    var_norm = exp_data[var_names].view(float).reshape(len(exp_data), -1).T
    std_norm = var_norm ** 0.5

    # Simulate model with new parameters and construct a matrix of the
    # trajectories of the observables of interest, normalized to 0-1.
    solver.run(params_estimated)
    obs_names_disp = obs_names + ['aSmac']
    sim_obs = solver.yobs[obs_names_disp].view(float).reshape(len(solver.yobs), -1)
    sim_obs_norm = (sim_obs / sim_obs.max(0)).T

    # Plot experimental data and simulation on the same axes
    colors = ('r', 'b')
    for exp, exp_err, sim, c in zip(exp_obs_norm, std_norm, sim_obs_norm, colors):
        plt.plot(exp_data['Time'], exp, color=c, marker='.', linestyle=':')
        plt.errorbar(exp_data['Time'], exp, yerr=exp_err, ecolor=c,
                     elinewidth=0.5, capsize=0, fmt=None)
        plt.plot(solver.tspan, sim, color=c)
    plt.plot(solver.tspan, sim_obs_norm[2], color='g')
    plt.vlines(momp_data[0], -0.05, 1.05, color='g', linestyle=':')
    plt.show()


if __name__ == '__main__':

    print 'Estimating rates for model:', model.name

    np.random.seed(1)
    params_estimated = estimate()

    # Write parameter values to a file
    fit_filename = 'fit_%s.txt' % model.name.replace('.', '_')
    fit_filename = os.path.join(earm_path, fit_filename)
    print 'Saving parameter values to file:', fit_filename
    pysb.util.write_params(model, params_estimated, fit_filename)

    display(params_estimated)
