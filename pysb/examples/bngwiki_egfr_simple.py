"""Translation of the BioNetGen example "egfr_simple" from the BNG wiki.

http://bionetgen.org/index.php/Egfr_simple
"""

from __future__ import print_function
from pysb import *

Model()


Parameter('NA', 6.02e23)     # Avogadro's number (molecules/mol)
Parameter('f', 1)            # Fraction of the cell to simulate
Expression('Vo', f*1.0e-10)  # Extracellular volume=1/cell_density (L)
Expression('V', f*3.0e-12)   # Cytoplasmic volume (L)

# Initial amount of ligand (20 nM) converted to copies per cell
Expression('EGF_init', 20*1e-9*NA*Vo)

# Initial amounts of cellular components (copies per cell)
Expression('EGFR_init', f*1.8e5)
Expression('Grb2_init', f*1.5e5)
Expression('Sos1_init', f*6.2e4)

# Rate constants
# Divide by NA*V to convert bimolecular rate constants from /M/sec to
# /(molecule/cell)/sec
Expression('kp1', 9.0e7/(NA*Vo))  # ligand-monomer binding
Parameter('km1', 0.06)            # ligand-monomer dissociation
Expression('kp2', 1.0e7/(NA*V))   # aggregation of bound monomers
Parameter('km2', 0.1)             # dissociation of bound monomers
Parameter('kp3', 0.5)             # dimer transphosphorylation
Parameter('km3', 4.505)           # dimer dephosphorylation
Expression('kp4', 1.5e6/(NA*V))   # binding of Grb2 to receptor
Parameter('km4', 0.05)            # dissociation of Grb2 from receptor
Expression('kp5', 1.0e7/(NA*V))   # binding of Grb2 to Sos1
Parameter('km5', 0.06)            # dissociation of Grb2 from Sos1
Parameter('deg', 0.01)            # degradation of receptor dimers


Monomer('EGF', ['R'])
Monomer('EGFR', ['L','CR1','Y1068'], {'Y1068':('U','P')})
Monomer('Grb2', ['SH2','SH3'])
Monomer('Sos1', ['PxxP'])


Initial(EGF(R=None), EGF_init)
Initial(EGFR(L=None, CR1=None, Y1068='U'), EGFR_init)
Initial(Grb2(SH2=None, SH3=None), Grb2_init)
Initial(Sos1(PxxP=None), Sos1_init)


Observable('EGFR_tot', EGFR())
Observable('Lig_free', EGF(R=None))
Observable('Dim', EGFR(CR1=ANY), match='species')
Observable('RP', EGFR(Y1068=('P',WILD)))
Observable('Grb2Sos1', Grb2(SH2=None, SH3=1) % Sos1(PxxP=1))
Observable('Sos1_act', EGFR(Y1068=1) % Grb2(SH2=1, SH3=2) % Sos1(PxxP=2))


# Ligand-receptor binding
Rule('egf_bind_egfr',
     EGFR(L=None, CR1=None) + EGF(R=None) <> EGFR(L=1, CR1=None) % EGF(R=1),
     kp1, km1)

# Receptor-aggregation
Rule('egfr_dimerize',
     EGFR(L=ANY, CR1=None) + EGFR(L=ANY, CR1=None) <>
     EGFR(L=ANY, CR1=1) % EGFR(L=ANY, CR1=1),
     kp2, km2)

# Transphosphorylation of EGFR by RTK
Rule('egfr_transphos',
     EGFR(CR1=ANY, Y1068='U') >> EGFR(CR1=ANY, Y1068='P'), kp3)

# Dephosphorylation
Rule('egfr_dephos',
     EGFR(Y1068='P') >> EGFR(Y1068='U'), km3)

# Grb2 binding to pY1068
Rule('grb2_bind_egfr',
     EGFR(Y1068='P') + Grb2(SH2=None) <> EGFR(Y1068=('P',1)) % Grb2(SH2=1),
     kp4, km4)

# Grb2 binding to Sos1
Rule('sos1_bind_grb2',
     Grb2(SH3=None) + Sos1(PxxP=None) <> Grb2(SH3=1) % Sos1(PxxP=1),
     kp5, km5)

# Receptor dimer internalization/degradation
Rule('egfr_dimer_degrade',
     EGF(R=1) % EGF(R=2) % EGFR(L=1, CR1=3) % EGFR(L=2, CR1=3) >> None,
     deg)


if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")
