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
from pysb.export import Export

class ExportMatlab(Export):
    def export(self):
        """Export the model as a set of ODEs for use in MATLAB.

        Returns
        -------
        string
            String containing the MATLAB code for the ODE right-hand side
            function.
        """

        output = StringIO()
        pysb.bng.generate_equations(self.model)

        model_name = self.model.name.replace('.', '_')

        # Header comment
        output.write("% MATLAB model definition file\n")
        output.write('%% save as %s_odes.m\n' % model_name)
        # The name of the ODE function
        output.write('function out = %s_odes(t, input, param)\n\n' % model_name)

        # Initialize the parameter array with parameter values
        params_str = '\n'.join(['param(%d) = %s %% %s;' % (i+1, p.value, p.name)
                                 for i, p in enumerate(self.model.parameters)])

        # Convert the species list to MATLAB comments
        species_list = ['%% %s;' % s for i, s in enumerate(self.model.species)]
        # Build the ODEs as strings from the model.odes array
        odes_list = ['out(%d,1) = %s;' % (i+1, sympy.ccode(self.model.odes[i])) 
                     for i in range(len(self.model.odes))] 
        # Zip the ODEs and species string lists and then flatten them
        # (results in the interleaving of the two lists)
        odes_species_list = [item for sublist in zip(species_list, odes_list)
                                  for item in sublist]
        odes_str = '\n'.join(odes_species_list)

        # Change species names from, e.g., 's(0)' to 'input(1)' (note change
        # from zero-based indexing to 1-based indexing)
        odes_str = re.sub(r's(\d+)', \
                          lambda m: 'input(%s)' % (int(m.group(1))+1), odes_str)
        # Change C code 'pow' function to MATLAB 'power' function
        odes_str = re.sub(r'pow\(', 'power(', odes_str)

        # Convert named parameters to, e.g. 'param(1)'
        for i, p in enumerate(self.model.parameters):
            odes_str = re.sub(r'\b(%s)\b' % p.name, 'param(%d)' % (i+1),
                              odes_str)

        # List the mapping of complexes to indices in the species array
        # species_list = '\n'.join(['%% input(%d) = %s;' % (i+1, s)
        #                            for i, s in enumerate(self.model.species)])

        output.write(params_str + "\n\n")
        output.write(odes_str + "\n\n")
        output.write("end\n")
        return output.getvalue()


