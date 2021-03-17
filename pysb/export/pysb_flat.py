"""
A module containing a class that exports a PySB model to a single Python source
file that, when imported, will recreate the same model. This is intended for
saving a dynamically generated model so that it can be reused without re-running
the dynamic generation process. Note that any macro calls and other program
structure in the original model are "flattened" in the process.

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.

Structure of the Python code
============================

The standalone Python code calls ``Model()``, then defines Monomers, Parameters,
Expressions (constant), Compartments, Observables, Expressions (dynamic), Rules
and initial conditions in that order. This can be considered a sort of "repr()"
for a full model.

If the output is saved as ``foo.py`` then one may load the model with the
following line::

    from foo import model

"""

from pysb.export import Exporter
from io import StringIO

class PysbFlatExporter(Exporter):
    """A class for generating PySB "flat" model source code from a model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """
    def export(self):
        """Export PySB source code from a model.

        Returns
        -------
        string
            String containing the Python code.
        """

        output = StringIO()

        # Convenience function for writing out a componentset.
        def write_cset(cset): 
            for c in cset:
                output.write(repr(c))
                output.write("\n")
            if cset:
                output.write("\n")

        if self.docstring:
            output.write('"""')
            output.write(self.docstring)
            output.write('"""\n\n')
        output.write("# exported from PySB model '%s'\n" % self.model.name)
        output.write("\n")
        output.write("from pysb import Model, Monomer, Parameter, Expression, "
                     "Compartment, Rule, Observable, Initial, MatchOnce, "
                     "Annotation, MultiState, Tag, ANY, WILD\n")
        output.write("\n")
        output.write("Model()\n")
        output.write("\n")
        write_cset(self.model.monomers)
        write_cset(self.model.parameters)
        write_cset(self.model.expressions_constant())
        write_cset(self.model.compartments)
        write_cset(self.model.observables)
        write_cset(self.model.tags)
        write_cset(self.model.expressions_dynamic())
        write_cset(self.model.rules)
        for ic in self.model.initials:
            output.write("%s\n" % ic)
        output.write("\n")
        write_cset(self.model.annotations)

        return output.getvalue()
