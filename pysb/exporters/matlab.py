"""
A class for converting a PySB model to a set of ordinary differential
equations for integration in MATLAB.

Note that for use in MATLAB, the name of the ``.m`` file must match the name of
the exported MATLAB class (e.g., ``robertson.m`` for the example below).

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.

Output for the Robertson example model
======================================

Information on the form and usage of the generated MATLAB class is contained in
the documentation for the MATLAB model, as shown in the following example for
``pysb.examples.robertson``::

    asdf
"""

import pysb
import pysb.bng
import sympy
import re
from StringIO import StringIO
from pysb.export import Export, pad

class ExportMatlab(Export):
    """A class for returning the ODEs for a given PySB model for use in
    MATLAB.

    Inherits from :py:class:`pysb.export.Export`, which implements
    basic functionality for all exporters.
    """
    def export(self):
        """Generate a MATLAB class definition containing the ODEs for the PySB
        model associated with the exporter.

        Returns
        -------
        string
            String containing the MATLAB code for an implementation of the
            model's ODEs.
        """
        output = StringIO()
        pysb.bng.generate_equations(self.model)

        # Substitute underscores for any dots in the model name
        model_name = self.model.name.replace('.', '_')

        # -- Parameters and Initial conditions -------
        # Declare the list of parameters as a struct
        params_str = 'self.parameters = struct( ...\n'+' '*16
        params_str_list = []
        for i, p in enumerate(self.model.parameters):
            # Add parameter to struct along with nominal value
            cur_p_str = "'%s', %g" % (p.name, p.value)
            # Decide whether to continue or terminate the struct declaration:
            if i == len(self.model.parameters) - 1:
                cur_p_str += ');'    # terminate
            else:
                cur_p_str += ', ...' # continue

            params_str_list.append(cur_p_str)

        # Format and indent the params struct declaration
        params_str += ('\n'+' '*16).join(params_str_list)

        # Fill in an array of the initial conditions based on the named
        # parameter values
        initial_values_str = ('initial_values = zeros(1,%d);\n'+' '*12) % \
                             len(self.model.species)
        initial_values_str += ('\n'+' '*12).join(
                ['initial_values(%d) = self.parameters.%s; %% %s' %
                 (i+1, ic[1].name, ic[0])
                 for i, ic in enumerate(self.model.initial_conditions)])

        # -- Build observables declaration --
        observables_str = 'self.observables = struct( ...\n'+' '*16
        observables_str_list = []
        for i, obs in enumerate(self.model.observables):
            # Associate species and coefficient lists with observable names,
            # changing from zero- to one-based indexing
            cur_obs_str = "'%s', [%s; %s]" % \
                          (obs.name,
                           ' '.join([str(sp+1) for sp in obs.species]),
                           ' '.join([str(c) for c in obs.coefficients]))
            # Decide whether to continue or terminate the struct declaration:
            if i == len(self.model.observables) - 1:
                cur_obs_str += ');'    # terminate
            else:
                cur_obs_str += ', ...' # continue

            observables_str_list.append(cur_obs_str)
        # Format and indent the observables struct declaration
        observables_str += ('\n'+' '*16).join(observables_str_list)

        # -- Build ODEs -------
        # Build a stringified list of species
        species_list = ['%% %s;' % s for i, s in enumerate(self.model.species)]
        # Build the ODEs as strings from the model.odes array
        odes_list = ['y(%d,1) = %s;' % (i+1, sympy.ccode(self.model.odes[i])) 
                     for i in range(len(self.model.odes))] 
        # Zip the ODEs and species string lists and then flatten them
        # (results in the interleaving of the two lists)
        odes_species_list = [item for sublist in zip(species_list, odes_list)
                                  for item in sublist]
        # Flatten to a string and add correct indentation
        odes_str = ('\n'+' '*12).join(odes_species_list)

        # Change species names from, e.g., 's(0)' to 'y0(1)' (note change
        # from zero-based indexing to 1-based indexing)
        odes_str = re.sub(r's(\d+)', \
                          lambda m: 'y0(%s)' % (int(m.group(1))+1), odes_str)
        # Change C code 'pow' function to MATLAB 'power' function
        odes_str = re.sub(r'pow\(', 'power(', odes_str)
        # Prepend 'self.' to named parameters
        for i, p in enumerate(self.model.parameters):
            odes_str = re.sub(r'\b(%s)\b' % p.name, 'p.%s' % p.name,
                              odes_str)

        # -- Build final output --
        output.write(pad(r"""
            classdef %(model_name)s
                %% A class implementing the ordinary differential equations
                %% for the %(model_name)s model.
                %%
                %% Save as %(model_name)s.m.
                %%
                %% Generated by pysb.exporters.matlab.ExportMatlab.
                %%
                %% Properties
                %% ----------
                %% observables : struct
                %%     A struct containing the names of the observables from the
                %%     PySB model as field names. Each field in the struct
                %%     maps the observable name to a matrix with two rows:
                %%     the first row specifies the indices of the species
                %%     associated with the observable, and the second row
                %%     specifies the coefficients associated with the species.
                %%     For any given timecourse of model species resulting from
                %%     integration, the timecourse for an observable can be
                %%     retrieved using the get_observable method, described
                %%     below.
                %%
                %% All model parameters are also exposed as properties by name,
                %% using the names from the original PySB model. The nominal
                %% values are set by the constructor and their values can be
                %% overriden explicitly once an instance has been created.
                %%
                %% Methods
                %% -------
                %% %(model_name)s.odes(tspan, y0)
                %%     The right-hand side function for the ODEs of the model,
                %%     for use with MATLAB ODE solvers (see Examples).
                %%
                %% %(model_name)s.get_initial_values()
                %%     Returns a vector of initial values for all species,
                %%     specified in the order that they occur in the original
                %%     PySB model (i.e., in the order found in model.species).
                %%     Non-zero initial conditions are specified using the
                %%     named parameters included as properties of the instance.
                %%     Hence initial conditions other than the defaults can be
                %%     used by assigning a value to the named parameter and then
                %%     calling this method. The vector returned by the method
                %%     is used for integration by passing it to the MATLAB
                %%     solver as the y0 argument.
                %%
                %% %(model_name)s.get_observable(x, obs_name)
                %%     Given a matrix of timecourses for all model species
                %%     (i.e., resulting from an integration of the model),
                %%     get the trajectory corresponding to the observable with
                %%     the given name (i.e., the sum of the appropriate
                %%     species).
                %%
                %% Examples
                %% --------
                %% Example integration using default initial and parameter
                %% values:
                %%
                %% >> m = %(model_name)s();
                %% >> tspan = [0 100];
                %% >> [t x] = ode15s(@m.odes, tspan, m.get_initial_values());
                %%
                properties
                    observables
                    parameters
                end

                methods
                    function self = %(model_name)s()
                        %% Assign default parameter values
                        %(params_str)s

                        %% Define species indices (first row) and coefficients
                        %% (second row) of named observables
                        %(observables_str)s
                    end

                    function initial_values = get_initial_values(self)
                        %% Return the vector of initial conditions for all
                        %% species based on the values of the parameters
                        %% as currently defined in the instance.

                        %(initial_values_str)s
                    end

                    function y = odes(self, tspan, y0)
                        %% Right hand side function for the ODEs

                        %% Shorthand for the struct of model parameters
                        p = self.parameters;

                        %(odes_str)s
                    end

                    function out = get_observable(self, y, obs_name)
                        %% Retrieve the trajectory for the named observable
                        %% from the trajectories of all model species.

                        obs = self.observables.(obs_name);
                        species = obs(1, :);
                        coefficients = obs(2, :);
                        out = y(:, species) * coefficients';
                    end
                end
            end
            """, 0) %
            {'model_name': model_name,
             'params_str':params_str,
             'initial_values_str': initial_values_str,
             'observables_str': observables_str,
             'params_str': params_str,
             'odes_str': odes_str})

        return output.getvalue()

