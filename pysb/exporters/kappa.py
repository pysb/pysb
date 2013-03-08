"""
A class for returning the Kappa equivalent for a given PySB model. Serves as
a wrapper around ``pysb.generator.kappa.KappaGenerator``.

Note that the Kappa syntax for the KaSim simulator is slightly different
from that for the complx analyzer; this script generates syntax for the
KaSim simulator.
"""

from pysb.generator.kappa import KappaGenerator
from pysb.export import Export

class ExportKappa(Export):
    def export(self, dialect='kasim'):
        """Generate the corresponding Kappa for the given PySB model.
        A wrapper around ``pysb.generator.kappa.KappaGenerator``.

        dialect : string, either 'kasim' or 'complx'
            The Kappa file syntax for the Kasim simulator is slightly
            different from that of the complx analyzer. This argument
            specifies which type of Kappa to produce ('kasim' is the default).

        Returns
        -------
        string
            The Kappa output.
        """

        gen = KappaGenerator(self.model, dialect=dialect)
        return gen.get_content()

