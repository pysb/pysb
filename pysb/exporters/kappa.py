"""
Module containing a class for returning the Kappa equivalent for a given PySB model.

Serves as a wrapper around :py:class:`pysb.generator.kappa.KappaGenerator`.

"""

from pysb.generator.kappa import KappaGenerator
from pysb.export import Export

class ExportKappa(Export):
    """A class for returning the Kappa for a given PySB model.

    Inherits from :py:class:`pysb.export.Export`, which implements
    basic functionality for all exporters.
    """

    def export(self, dialect='kasim'):
        """Generate the corresponding Kappa for the PySB model associated with
        the exporter. A wrapper around
        :py:class:`pysb.generator.kappa.KappaGenerator`.

        Parameters
        ----------
        dialect : (optional) string, either 'kasim' (default) or 'complx'
            The Kappa file syntax for the Kasim simulator is slightly
            different from that of the complx analyzer. This argument
            specifies which type of Kappa to produce ('kasim' is the default).

        Returns
        -------
        string
            The Kappa output.
        """
        kappa_str = ''
        if self.docstring:
            kappa_str += '# ' + self.docstring.replace('\n', '\n# ') + '\n'
        gen = KappaGenerator(self.model, dialect=dialect)
        kappa_str += gen.get_content()
        return kappa_str

