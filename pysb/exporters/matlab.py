"""
A class for converting a PySB model to a set of ordinary differential
equations for integration in MATLAB.

Note that for use in MATLAB, the name of the ``.m`` file must match the name of
the ODE function (e.g., ``robertson_odes.m`` for the example below).

Output for the Robertson example model
======================================

The MATLAB code produced will follow the form as given below for
``pysb.examples.robertson``::

    % MATLAB model definition file
    % save as robertson_odes.m
    function out = robertson_odes(t, input, param)

    param(1) = 0.04 % k1;
    param(2) = 30000000.0 % k2;
    param(3) = 10000.0 % k3;
    param(4) = 1.0 % A_0;
    param(5) = 0.0 % B_0;
    param(6) = 0.0 % C_0;

    % A();
    out(1,1) = -param(1)*input(1) + param(3)*input(2)*input(3);
    % B();
    out(2,1) = param(1)*input(1) - param(2)*power(input(2), 2) - param(3)*input(2)*input(3);
    % C();
    out(3,1) = param(2)*power(input(2), 2);

    end

The MATLAB output consists of a single function defining the right-hand side of
the ODEs, for use with any of the MATLAB solvers (e.g., ``ode15s``).  The name
of the function is ``[model_name]_odes``, where ``model_name`` is the name of
the file containing the model definition, stripped of the ``.py`` extension.

The first block of code maps the named parameters of the model to entries in a
MATLAB parameter array. The second block of code lists the ODEs, with the
comment above each ODE indicating the corresponding species.
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
        # Declare the list of parameters. Ignore any initial condition
        # parameters, as these will get incorporated into the initial
        # values vector
        params_str = ('\n'+' '*8).join(
                     ['%s' % p.name for p in self.model.parameters
                      if p not in self.model.parameters_initial_conditions()])

        # Get a list of tuples for all species in the model, consisting
        # of the (zero or non-zero) initial condition for the species
        # and the species' string representation
        ic_values = ['0'] * len(self.model.odes)
        for i, ic in enumerate(self.model.initial_conditions):
            ic_values[self.model.get_species_index(ic[0])] = ic[1].value
        ic_tuples = zip(ic_values, self.model.species)

        # Build the list of initial condition declarations, assigning the
        # values and adding a comment to indicate the identity of the species
        initial_values_str = ('\n'+' '*12).join(
                ['self.default_initial_values(%d) = %s; %% %s' %
                 (i+1, ic_tuple[0], ic_tuple[1])
                 for i, ic_tuple in enumerate(ic_tuples)])

        # Assign nominal values to parameters, ignoring any initial condition
        # parameters
        param_values_str = ('\n'+' '*12).join(
                       ['self.%s = %g;' % (p.name, p.value)
                        for p in self.model.parameters
                        if p not in self.model.parameters_initial_conditions()])

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
            odes_str = re.sub(r'\b(%s)\b' % p.name, 'self.%s' % p.name,
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
                %% default_initial_values : vector of doubles
                %%     The vector of default initial values for all species,
                %%     specified in the order that they occur in the original
                %%     PySB model (i.e., in the order found in model.species).
                %%     The default values can be used by passing this vector to
                %%     the MATLAB solver as the y0 argument; however, if
                %%     desired, a vector of alternative initial conditions can
                %%     be passed to the solver instead (see Examples below).
                %%     The order of species can be found by inspecting the
                %%     source code for the constructor, in which the default
                %%     values are assigned.
                %%
                %% All rate parameters are also exposed as properties by name,
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
                %% Examples
                %% --------
                %% Example integration using default initial and parameter
                %% values:
                %%
                %% >> m = %(model_name)s();
                %% >> tspan = [0 100];
                %% >> [t x] = ode15s(@m.odes, tspan, m.default_initial_values);
                %%
                %% Example using overriden initial values:
                %%
                %% >> m = %(model_name)s();
                %% >> tspan = [0 100];
                %% >> [t x] = ode15s(@m.odes, tspan, new_initial_values);
                %%
                properties
                    default_initial_values
                    %(params_str)s
                end

                methods
                    function self = %(model_name)s()
                        %% Default initial values
                        %(initial_values_str)s

                        %% Assign default parameter values
                        %(param_values_str)s
                    end

                    function y = odes(self, tspan, y0)
                        %% Right hand side function for the ODEs

                        %(odes_str)s
                    end
                end
            end
            """, 0) %
            {'model_name': model_name,
             'params_str':params_str,
             'initial_values_str': initial_values_str,
             'param_values_str': param_values_str,
             'odes_str': odes_str})

        return output.getvalue()



