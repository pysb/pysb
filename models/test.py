from pysb import *

Model()

Compartment('extracellular', dimension=3)
Compartment('membrane', parent=extracellular, dimension=2)

Monomer('egf', ['R'])
Monomer('egfr', ['L', 'D', 'C'])

Parameter('Kf_egfr_egf', 1.2)
Parameter('Kr_egfr_egf', 1.1)
Rule('egfr_egf',
     egfr(L=None) + egf(R=None) <>
     egfr(L=1)    % egf(R=1),
     Kf_egfr_egf, Kr_egfr_egf)

Observe('free_egf', egf(R=None))
Observe('free_egfr', egfr(L=None))
Observe('bound', egf(R=1) % egfr(L=1))

Parameter('egf_0', 6.0)
Parameter('egfr_0', 10.0)
Initial(egf(R=None) ** extracellular, egf_0)
Initial(egfr(L=None, D=None, C=None) ** membrane, egfr_0)


if __name__ == '__main__':
    from pysb.generator.bng import BngGenerator
    gen = BngGenerator(model)
    print gen.get_content()
    print "begin actions"
    print "  generate_network({overwrite=>1});"
    print "end actions"

