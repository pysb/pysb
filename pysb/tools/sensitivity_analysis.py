import os
from itertools import product
import collections
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from pysb.bng import generate_equations
import pysb.simulator.base
from pysb.logging import get_logger


class InitialsSensitivity(object):
    """
    Pairwise sensitivity analysis of initial conditions

    This class calculates the sensitivity of a specified model Observable to
    changes in pairs of initial species concentrations. The results are
    stored in matrices described in Attributes.

    .. warning::
        The interface for this class is considered experimental and may
        change without warning as PySB is updated.

    Parameters
    ----------
    solver : pysb.simulator.Simulator
        Simulator instance used to perform models. Must be initialized with
        tspan argument set.
    values_to_sample : vector_like
        Values to sample for each initial concentration of the
        model.parameters values.
    objective_function : function
        A function that returns a scalar value. Used to calculate fraction
        of changed that is used for calculating sensitivity. See Example.
    observable : str
        Observable used in the objective_function.

    Attributes
    ----------
    b_matrix: numpy.ndarray
        Matrix of 2-tuples containing (perturbation, species index)
    b_prime_matrix: numpy.ndarray
        Same as b_matrix, only where one of the species concentrations is
        unchanged (i.e. with the single variable contribution removed)
    p_matrix: numpy.ndarray
        Pairwise sensitivity matrix
    p_prime_matrix: numpy.ndarray
        Normalized pairwise sensitivity matrix (in the sense that it
        contains changes from the baseline, unperturbed case)

    References
    ----------
    Harris et al. Bioinformatics (2017), under review.

    Examples
    --------
    Sensitivity analysis on the Tyson cell cycle model

    >>> from pysb.examples.tyson_oscillator import model
    >>> import numpy as np
    >>> from pysb.simulator.scipyode import ScipyOdeSimulator
    >>> np.set_printoptions(precision=4)
    >>> tspan=np.linspace(0, 200, 201)
    >>> observable = 'Y3'
    >>> values_to_sample = [.8, 1.2]
    >>> def obj_func_cell_cycle(out):
    ...     timestep = tspan[:-1]
    ...     y = out[:-1] - out[1:]
    ...     freq = 0
    ...     local_times = []
    ...     prev = y[0]
    ...     for n in range(1, len(y)):
    ...         if y[n] > 0 > prev:
    ...             local_times.append(timestep[n])
    ...             freq += 1
    ...         prev = y[n]
    ...     local_times = np.array(local_times)
    ...     local_freq = np.average(local_times)/len(local_times)*2
    ...     return local_freq
    >>> solver = ScipyOdeSimulator(model, tspan)
    >>> sens = InitialsSensitivity(\
            values_to_sample=values_to_sample,\
            observable=observable,\
            objective_function=obj_func_cell_cycle,\
            solver=solver\
        )
    >>> print(sens.b_matrix)
    [[((0.8, 'cdc0'), (0.8, 'cdc0')) ((0.8, 'cdc0'), (1.2, 'cdc0'))
      ((0.8, 'cdc0'), (0.8, 'cyc0')) ((0.8, 'cdc0'), (1.2, 'cyc0'))]
     [((1.2, 'cdc0'), (0.8, 'cdc0')) ((1.2, 'cdc0'), (1.2, 'cdc0'))
      ((1.2, 'cdc0'), (0.8, 'cyc0')) ((1.2, 'cdc0'), (1.2, 'cyc0'))]
     [((0.8, 'cyc0'), (0.8, 'cdc0')) ((0.8, 'cyc0'), (1.2, 'cdc0'))
      ((0.8, 'cyc0'), (0.8, 'cyc0')) ((0.8, 'cyc0'), (1.2, 'cyc0'))]
     [((1.2, 'cyc0'), (0.8, 'cdc0')) ((1.2, 'cyc0'), (1.2, 'cdc0'))
      ((1.2, 'cyc0'), (0.8, 'cyc0')) ((1.2, 'cyc0'), (1.2, 'cyc0'))]]
    >>> sens.run()
    >>> print(sens.p_matrix)#doctest: +NORMALIZE_WHITESPACE
    [[ 0.      0.      5.0243 -4.5381]
     [ 0.      0.      5.0243 -4.5381]
     [ 5.0243  5.0243  0.      0.    ]
     [-4.5381 -4.5381  0.      0.    ]]

    >>> print(sens.p_prime_matrix) #doctest: +NORMALIZE_WHITESPACE
     [[ 0.      0.      5.0243 -4.5381]
      [ 0.      0.      5.0243 -4.5381]
      [ 0.      0.      0.      0.    ]
      [ 0.      0.      0.      0.    ]]
    >>> print(sens.p_matrix - sens.p_prime_matrix) \
            #doctest: +NORMALIZE_WHITESPACE
     [[ 0.      0.      0.      0.    ]
      [ 0.      0.      0.      0.    ]
      [ 5.0243  5.0243  0.      0.    ]
      [-4.5381 -4.5381  0.      0.    ]]
    >>> sens.create_boxplot_and_heatplot() #doctest: +SKIP
    """

    def __init__(self, solver, values_to_sample, objective_function,
                 observable):
        self._model = solver.model
        self._logger = get_logger(__name__, model=self._model)
        self._logger.info('%s created for observable %s' % (
            self.__class__.__name__, observable))
        generate_equations(self._model)
        self._ic_params_of_interest_cache = None
        self._values_to_sample = values_to_sample
        if solver is None or not isinstance(solver,
                                            pysb.simulator.base.Simulator):
            raise(TypeError, "solver must be a pysb.simulator object")
        self._solver = solver
        self.objective_function = objective_function
        self.observable = observable

        # Outputs
        self.b_matrix = []
        self.b_prime_matrix = []
        self.p_prime_matrix = np.zeros(self._size_of_matrix)
        self.p_matrix = np.zeros(self._size_of_matrix)

        self._original_initial_conditions = np.zeros(len(self._model.species))
        self._index_of_species_of_interest = self._create_index_of_species()
        self.simulation_initials = self._setup_simulations()
        # Stores the objective function value for the original unperturbed
        # model
        self._objective_fn_standard = None

    @property
    def _ic_params_of_interest(self):
        if self._ic_params_of_interest_cache is None:
            self._ic_params_of_interest_cache = list(
                (i[1].name for i in self._model.initial_conditions))
            self._ic_params_of_interest_cache = \
                sorted(self._ic_params_of_interest_cache)

        return self._ic_params_of_interest_cache

    @property
    def _n_species(self):
        return len(self._ic_params_of_interest)

    @property
    def _n_sam(self):
        return len(self._values_to_sample)

    @property
    def _nm(self):
        return self._n_species * self._n_sam

    @property
    def _size_of_matrix(self):
        return self._nm ** 2

    @property
    def _shape_of_matrix(self):
        return self._nm, self._nm

    @property
    def sensitivity_multiset(self):
        """
        Sensitivity analysis multiset (also called "Q" matrix)

        Returns
        -------
        list
            List of lists containing the sensitivity analysis multiset
        """
        sens_ij_nm = []
        sens_matrix = self.p_matrix - self.p_prime_matrix

        # separate each species sensitivity
        for j in range(0, self._nm, self._n_sam):
            per_species1 = []
            for i in range(0, self._nm, self._n_sam):
                if i != j:
                    tmp = sens_matrix[j:j + self._n_sam,
                          i:i + self._n_sam].copy()
                    per_species1.append(tmp)
            sens_ij_nm.append(per_species1)
        return sens_ij_nm

    def _calculate_objective(self, function_value):
        """
        Calculate fraction of change for obj value and standard

        Parameters
        ----------
        function_value : scalar
            scalar value provided by objective function
        """
        return (self.objective_function(function_value)
                - self._objective_fn_standard) / \
               self._objective_fn_standard * 100.

    def run(self, save_name=None, out_dir=None):
        """
        Run sensitivity analysis

        Parameters
        ----------
        save_name : str, optional
            prefix of saved files
        out_dir : str, optional
            location to save output if required
        """
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
        self._solver.initials = None
        self._solver.param_values = None
        sim_results = self._solver.run(param_values=None, initials=None)
        self._objective_fn_standard = self.objective_function(
            np.array(sim_results.observables[self.observable]))

        traj = self._solver.run(initials=self.simulation_initials)
        sensitivity_output = np.array(traj.observables)[self.observable].T
        p_matrix = np.zeros(self._size_of_matrix)
        p_prime_matrix = np.zeros(self._size_of_matrix)
        counter = 0
        # places values in p matrix that are unique
        for i in range(len(p_matrix)):
            if i in self.b_index:
                tmp = self._calculate_objective(sensitivity_output[:, counter])
                p_matrix[i] = tmp
                counter += 1
        # places values in p matrix that are duplicated
        for i in range(len(p_matrix)):
            if i in self.b_prime_not_in_b:
                tmp = self._calculate_objective(
                    sensitivity_output[:, self.b_prime_not_in_b[i]])
                p_prime_matrix[i] = tmp
            elif i in self.b_prime_in_b:
                p_prime_matrix[i] = p_matrix[self.b_prime_in_b[i]]
        # Reshape
        p_matrix = p_matrix.reshape(self._shape_of_matrix)
        # Project the mirrored image
        self.p_matrix = p_matrix + p_matrix.T
        self.p_prime_matrix = p_prime_matrix.reshape(self._shape_of_matrix)

        # save output if desired
        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            p_name = os.path.join(out_dir, '{}_p_matrix.csv'.format(save_name))
            p_prime_name = os.path.join(
                out_dir, '{}_p_prime_matrix.csv'.format(save_name))
            self._logger.debug("Saving p matrix and p' matrix to {} and {}".
                               format(p_name, p_prime_name))

            np.savetxt(p_name, self.p_matrix)
            np.savetxt(p_prime_name, self.p_prime_matrix)

    def _create_index_of_species(self):
        """ create dictionary of initial conditions by index """
        index_of_init_condition = {}
        for ic_sp, ic_param in self._model.initial_conditions:
            for sp_idx, sp in enumerate(self._model.species):
                if ic_sp.is_equivalent_to(sp):
                    self._original_initial_conditions[sp_idx] = ic_param.value
                    if ic_param.name in self._ic_params_of_interest:
                        index_of_init_condition[ic_param.name] = sp_idx
        index_of_species_of_interest = collections.OrderedDict(
            sorted(index_of_init_condition.items()))
        return index_of_species_of_interest

    def _g_function(self):
        """ creates sample matrix, index of samples values, and shows bij """
        counter = -1
        sampled_values_index = set()
        bij_unique = dict()
        sampling_matrix = np.zeros(
            (self._size_of_matrix, len(self._model.species)))
        sampling_matrix[:, :] = self._original_initial_conditions
        matrix = self.b_matrix
        for j in range(len(matrix)):
            for i in matrix[j, :]:
                sigma_i, index_i = i[0]
                sigma_j, index_j = i[1]
                s_1 = str(i[0]) + str(i[1])
                s_2 = str(i[1]) + str(i[0])
                counter += 1

                if index_i == index_j:
                    continue
                elif s_1 in bij_unique or s_2 in bij_unique:
                    continue
                else:
                    x = self._index_of_species_of_interest[index_i]
                    y = self._index_of_species_of_interest[index_j]
                    sampling_matrix[counter, x] *= sigma_i
                    sampling_matrix[counter, y] *= sigma_j
                    bij_unique[s_1] = counter
                    bij_unique[s_2] = counter
                    sampled_values_index.add(counter)

        return sampling_matrix, sampled_values_index, bij_unique

    def _setup_simulations(self):
        """
        Create initial conditions matrix for sensitivity analysis

        Returns
        -------
        numpy.ndarray
            Matrix of initial conditions
        """
        # create matrix (cartesian product of sample vals vs index of species
        a_matrix = cartesian_product(self._values_to_sample,
                                     self._index_of_species_of_interest)
        # reshape to flatten
        a_matrix = a_matrix.T.reshape(self._n_sam * self._n_species)
        # creates matrix b
        self.b_matrix = cartesian_product(a_matrix, a_matrix)

        # create matrix a'
        a_prime = cartesian_product(np.ones(self._n_sam),
                                    self._index_of_species_of_interest)
        a_prime = a_prime.T.reshape(self._n_sam * self._n_species)

        # creates matrix b prime
        self.b_prime_matrix = cartesian_product(a_prime, a_matrix)

        b_to_run, self.b_index, in_b = self._g_function()

        n_b_index = len(self.b_index)

        b_prime = np.zeros((self._size_of_matrix, len(self._model.species)))
        b_prime[:, :] = self._original_initial_conditions
        counter = -1

        bp_not_in_b_raw = set()
        bp_dict = dict()
        bp_not_in_b_dict = dict()
        bp_not_in_b_visited = dict()
        new_sim_counter = -1
        # checks for and removes duplicates of simulations initial conditions
        for j in range(len(self.b_prime_matrix)):
            for i in self.b_prime_matrix[j, :]:
                sigma_i, index_i = i[0]
                sigma_j, index_j = i[1]
                s_1 = str(i[0]) + str(i[1])
                counter += 1
                # no need for doing if same index
                if index_i == index_j:
                    continue
                # pointing to same simulation if already in b
                elif s_1 in in_b:
                    bp_dict[counter] = in_b[s_1]
                elif s_1 in bp_not_in_b_visited:
                    bp_not_in_b_dict[counter] = bp_not_in_b_visited[s_1]
                else:
                    new_sim_counter += 1
                    x = self._index_of_species_of_interest[index_i]
                    y = self._index_of_species_of_interest[index_j]
                    b_prime[new_sim_counter, x] *= sigma_i
                    b_prime[new_sim_counter, y] *= sigma_j
                    bp_not_in_b_visited[s_1] = new_sim_counter + n_b_index
                    bp_not_in_b_dict[counter] = new_sim_counter + n_b_index
                    bp_not_in_b_raw.add(new_sim_counter)

        self.b_prime_in_b = bp_dict
        self.b_prime_not_in_b = bp_not_in_b_dict
        x = b_to_run[list(self.b_index)]
        y = b_prime[list(bp_not_in_b_raw)]
        simulations = np.vstack((x, y))
        self._logger.debug("Number of simulations to run = %s" % len(
                           simulations))
        return simulations

    def create_plot_p_h_pprime(self, save_name=None, out_dir=None, show=False):
        """
        Plot of P, H(B), and P'

        See :class:`InitialsSensitivity` for descriptions of these matrices

        Parameters
        ----------
        save_name : str, optional
            name to save figure as
        out_dir : str, optional
            location to save figure
        show : bool
            show the plot if True

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object for further adjustments, if required
        """
        colors = 'seismic'
        sens_matrix = self.p_matrix - self.p_prime_matrix
        v_max = max(np.abs(self.p_matrix.min()), self.p_matrix.max())
        v_min = -1 * v_max

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.imshow(self.p_matrix, interpolation='nearest', origin='upper',
                   cmap=plt.get_cmap(colors), vmin=v_min, vmax=v_max,
                   extent=[0, self._nm, 0, self._nm])

        ax2.imshow(self.p_prime_matrix, interpolation='nearest',
                   origin='upper', cmap=plt.get_cmap(colors),
                   vmin=v_min, vmax=v_max, extent=[0, self._nm, 0, self._nm])

        ax3.imshow(sens_matrix, interpolation='nearest', origin='upper',
                   cmap=plt.get_cmap(colors), vmin=v_min, vmax=v_max)

        for ax in (ax1, ax2, ax3):
            ax.set_xticks([])
            ax.set_yticks([])

        ax1.set_title('P', fontsize=22)
        ax2.set_title('H(B\')', fontsize=22)
        ax3.set_title('P\' = P - H(B\')', fontsize=22)
        fig.subplots_adjust(wspace=0, hspace=0.0)
        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            fig.savefig(os.path.join(out_dir,
                                     '{}_P_H_P_prime.png'.format(save_name)),
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        return fig

    def create_individual_pairwise_plots(self, save_name=None, out_dir=None,
                                         show=False):
        """
        Single plot containing heat plot of each specie pair

        Parameters
        ----------
        save_name : str, optional
            name ot save figure as
        out_dir : str, optional
            output directory
        show : bool
            show figure

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object for further adjustments, if required
        """
        colors = 'seismic'
        sens_matrix = self.p_matrix - self.p_prime_matrix
        v_max = max(np.abs(self.p_matrix.min()), self.p_matrix.max())
        v_min = -1 * v_max
        fig = plt.figure(figsize=(self._n_species + 6, self._n_species + 6))
        gs = gridspec.GridSpec(self._n_species, self._n_species)
        # creates a plot of each species vs each species
        # adds space between plots so you can zoom in on output pairs
        for n, j in enumerate(range(0, self._nm, self._n_sam)):
            for m, i in enumerate(range(0, self._nm, self._n_sam)):
                ax2 = plt.subplot(gs[n, m])
                if n == 0:
                    ax2.set_xlabel(self._ic_params_of_interest[m], fontsize=20)
                    ax2.xaxis.set_label_position('top')
                if m == 0:
                    ax2.set_ylabel(self._ic_params_of_interest[n], fontsize=20)
                plt.xticks([])
                plt.yticks([])
                if i != j:
                    tmp = sens_matrix[j:j + self._n_sam,
                          i:i + self._n_sam].copy()
                    ax2.imshow(tmp, interpolation='nearest', origin='upper',
                               cmap=plt.get_cmap(colors), vmin=v_min,
                               vmax=v_max)
                else:
                    ax2.imshow(np.zeros((self._n_sam, self._n_sam)),
                               interpolation='nearest', origin='upper',
                               cmap=plt.get_cmap(colors), vmin=v_min,
                               vmax=v_max)
        plt.tight_layout()

        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            plt.savefig(os.path.join(out_dir,
                                     '{}_subplots.png'.format(save_name)),
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        return fig

    def create_boxplot_and_heatplot(self, x_axis_label=None, save_name=None,
                                    out_dir=None, show=False):
        """
        Heat map and box plot of sensitivities

        Parameters
        ----------
        x_axis_label : str, optional
            label for x asis
        save_name : str, optional
            name of figure to save
        out_dir : str, option
            output directory to save figures
        show : bool
            Show plot if True

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object for further adjustments, if required
        """

        colors = 'seismic'
        sens_ij_nm = self.sensitivity_multiset

        # Create heatmap and boxplot of data
        fig = plt.figure(figsize=(14, 10))
        plt.subplots_adjust(hspace=0.1)

        # use gridspec to scale colorbar nicely
        outer = gridspec.GridSpec(2, 1, width_ratios=[1.],
                                  height_ratios=[0.03, 1])

        gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                               hspace=.35)
        ax0 = plt.subplot(gs1[0])
        ax1 = plt.subplot(gs2[0])

        # scale the colors to minimum or maximum of p matrix
        v_max = max(np.abs(self.p_matrix.min()), self.p_matrix.max())
        v_min = -1 * v_max

        # create heatmap of sensitivities
        im = ax1.imshow(self.p_matrix, interpolation='nearest', origin='upper',
                        cmap=plt.get_cmap(colors), vmin=v_min,
                        vmax=v_max, extent=[0, self._nm, 0, self._nm])

        shape_label = np.arange(self._n_sam / 2, self._nm, self._n_sam)
        plt.xticks(shape_label, self._ic_params_of_interest,
                   rotation='vertical', fontsize=12)
        plt.yticks(shape_label, reversed(self._ic_params_of_interest),
                   fontsize=12)
        x_ticks = ([i for i in range(0, self._nm, self._n_sam)])
        ax1.set_xticks(x_ticks, minor=True)
        ax1.set_yticks(x_ticks, minor=True)
        plt.grid(True, which='minor', linestyle='--')
        color_bar = plt.colorbar(im, cax=ax0, orientation='horizontal',
                                 use_gridspec=True)
        color_bar.set_label('% change', y=1, labelpad=5)
        color_bar.ax.xaxis.set_label_position('top')
        ticks = np.linspace(v_min, v_max, 5, dtype=int)
        color_bar.set_ticks(ticks)
        color_bar.ax.set_xticklabels(ticks)

        # create boxplot of single parameter sensitivities
        ax2 = plt.subplot(gs2[1])
        x = [np.array(mat).flatten() for mat in sens_ij_nm[::-1]]
        ax2.boxplot(x, vert=False, labels=None, showfliers=True, whis='range')
        ax2.set_xlim(v_min - 2, v_max + 2)
        if x_axis_label is not None:
            ax2.set_xlabel(x_axis_label, fontsize=12)
        plt.setp(ax2, yticklabels=reversed(self._ic_params_of_interest))
        ax2.yaxis.tick_left()
        ax2.set_aspect(1. / ax2.get_data_ratio(), adjustable='box')
        if save_name is not None:
            if out_dir is None:
                out_dir = '.'
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            plt.savefig(os.path.join(out_dir, save_name + '.png'),
                        bbox_inches='tight')
            plt.savefig(os.path.join(out_dir, save_name + '.eps'),
                        bbox_inches='tight')
            plt.savefig(os.path.join(out_dir, save_name + '.svg'),
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

        return fig


def cartesian_product(array_1, array_2):
    """
    Cartesian product between two lists

    Parameters
    ----------
    array_1 : list_like
    array_2 : list_like

    Returns
    -------
    out : np.array
        array of shape (len(array_1) x len(array_2))
    """
    a = list(product(array_1, array_2))
    a = np.asarray(a, dtype=','.join(['object'] * len(a[0])))
    return a.reshape(len(array_1), len(array_2))
