from Pysb import *
import generator.bng as bng

Model('simple_egfr')


Parameter('EGF_tot',      1.2e6)
Parameter('EGFR_tot',     1.8e5)
Parameter('Grb2_tot',     1.5e5)
Parameter('Sos_tot',      6.2e4)

# Biomolecular rate constants are in (# per cell)^-1 s^-1,
#  obtained by dividing constants in M^-1 s^-1 by Na*V,
#  where Na is Avogadro's number and V is the volume
#  of the relevant compartment (the cytoplasm for all cases here).
# Unimolecular rate constants are in s^-1
Parameter('kp1',      1.667e-06) # ligand-monomer binding
Parameter('km1',           0.06) # ligand-monomer dissociation

Parameter('kp2',      5.556e-06) # aggregation of bound monomers
Parameter('km2',            0.1) # dissociation of bound monomers

Parameter('kp3',            0.5) # dimer transphosphorylation   
Parameter('km3',          4.505) # dimer dephosphorylation        

Parameter('kp4',      8.333e-07) # binding of Grb2 to receptor
Parameter('km4',           0.05) # dissociation of Grb2 from receptor

Parameter('kp5',      5.556e-06) # binding of Grb2 to Sos
Parameter('km5',           0.06) # dissociation of Grb2 from Sos

Parameter('kdeg',          0.01)


Monomer('EGF', 'r')
Monomer('EGFR',
        ['l','r','Y1068','Y1148'],
        { 'Y1068': ['U','P'],
          'Y1148': ['U','P'] }
        )
Monomer('Grb2', ['SH2','SH3'])
Monomer('Shc', ['PTB','Y317'], { 'Y317': ['U','P'] } )
Monomer('Sos', 'PR')
Monomer('NULL')


# Ligand-receptor binding      
#EGFR(l,r) + EGF(r) <-> EGFR(l!1,r).EGF(r!1) kp1, km1
Rule('ligand_receptor_binding',
     EGFR(l=None, r=None) + EGF(r=None),
     EGFR(l=1, r=None)    + EGF(r=1),
     kp1)
ligand_receptor_binding.reversible(km1)

# Receptor-aggregation 
#EGFR(l!+,r) + EGFR(l!+,r) <-> EGFR(l!+,r!1).EGFR(l!+,r!1) kp2,km2
Rule('receptor_aggregation',
     EGFR(l=ANY, r=None) + EGFR(l=ANY, r=None),
     EGFR(l=ANY, r=1)    + EGFR(l=ANY, r=1),
     kp2)
receptor_aggregation.reversible(km2)

# Transphosphorylation of EGFR by RTK
#EGFR(r!+,Y1068~U) -> EGFR(r!+,Y1068~P)  kp3
# FIXME: implement once we support states in MonomerPattern

# Dephosphorylayion
#EGFR(Y1068~P) -> EGFR(Y1068~U)  km3

# Grb2 binding to pY1068
#EGFR(Y1068~P) + Grb2(SH2)   <-> EGFR(Y1068~P!1).Grb2(SH2!1)   kp4,km4

# Grb2 binding to Sos
#Grb2(SH2,SH3) + Sos(PR) <-> Grb2(SH2,SH3!1).Sos(PR!1) kp5,km5

# Receptor dimer internalization/degradation
#EGFR().EGFR() -> NULL() kdeg DeleteMolecules



gen = bng.BngGenerator(model=simple_egfr)
print gen.content
