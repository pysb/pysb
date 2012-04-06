from pysb import *


rnas = []

def Nuc(name, sites=[], site_states={}, **kwargs):
    return Monomer(name, sites + ['r5', 'r3'], site_states, **kwargs)

def Rna(*args, **kwargs):
    rna = Nuc(*args, **kwargs)
    rnas.append(rna)
    return rna

def degrade_rna(rna):
    Rule('%s_deg' % rna.name, rna(r3=None) >> NONE(), rna_deg)


Model()

Parameter('rna_deg', 1e-6)
Parameter('a_deg', 1e-3)
Parameter('mRNA_1_init', 50)
Parameter('mRNA_2_init', 10)

Rna('mRNA_1')
Rna('mRNA_2')
Nuc('A')
Monomer('NONE')

for rna in rnas:
    degrade_rna(rna)
Rule('deadenylate', A(r3=None) >> NONE(), a_deg, delete_molecules=True)

Initial(mRNA_1(r5=None, r3=1) % A(r5=1, r3=2) % A(r5=2, r3=None), mRNA_1_init)
Initial(mRNA_2(r5=None, r3=1) % A(r5=1, r3=2) % A(r5=2, r3=3) % A(r5=3, r3=None), mRNA_2_init)




if __name__ == '__main__':
    from pysb.tools.export_bng import run as run_export
    print run_export(model)
