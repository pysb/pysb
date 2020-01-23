"""An implementation of the model from:

Model-based dissection of CD95 signaling dynamics reveals both a pro- and
antiapoptotic role of c-FLIPL.  Fricker N, Beaudouin J, Richter P, Eils R,
Krammer PH, Lavrik IN. J Cell Biol. 2010 Aug 9;190(3):377-89.
doi:10.1083/jcb.201002060

http://jcb.rupress.org/content/190/3/377.long

Implemented by: Jeremie Roux, Will Chen, Jeremy Muhlich
"""

from __future__ import print_function
from pysb import *

Model()

# Non-zero initial conditions (in molecules per cell):
Parameter('L_0'        , 1500e3); # baseline level of ligand for most experiments (corresponding to 50 ng/ml SuperKiller TRAIL)
Parameter('pR_0'       , 170.999e3);  # TRAIL receptor (for experiments not involving siRNA)
Parameter('FADD_0'     , 133.165e3);
Parameter('flipL_0'    , 0.49995e3);  # FlipL 1X = 0.49995e3
Parameter('flipS_0'    , 0.422e3);  # Flip
Parameter('pC8_0'      , 200.168e3);  # procaspase-8 (pro-C8)
Parameter('Bid_0'       , 100e3);  # Bid

Monomer('L', ['b'])
Monomer('pR', ['b', 'rf'])
Monomer('FADD', ['rf', 'fe'])
Monomer('flipL', ['b', 'fe', 'ee', 'D384'],
        {'D384': ['U','C']}
        )
Monomer('flipS', ['b', 'fe', 'ee'])
Monomer('pC8', ['fe', 'ee', 'D384', 'D400'],
        {'D384': ['U','C'],
	 'D400': ['U','C']}
        )
Monomer('Bid') #called Apoptosis substrat in Lavrik's model
Monomer('tBid')

flip_monomers = (flipL, flipS);

# L + R <--> L:R
Parameter('kf1', 70.98e-03) #70.98e-03
Parameter('kr1', 0)
Rule('R_L_Binding', L (b=None) + pR (b=None, rf=None) >> L (b=1) % pR (b=1, rf=None), kf1)

# FADD binds
Parameter('kf29', 84.4211e-03) #84.4211e-03
Rule('RL_FADD_Binding', pR (b=ANY, rf=None) + FADD (rf=None, fe=None) >> pR (b=ANY, rf=2) % FADD (rf=2, fe=None), kf29)

#C8 binds to L:R:FADD
Parameter('kf30', 3.19838e-03) #3.19838e-03
Parameter('kr30', 0.1) #0.1
Rule('RLFADD_C8_Binding', FADD (rf=ANY, fe=None) + pC8 (fe=None, ee=None, D384='U') | FADD (rf=ANY, fe=1) % pC8 (fe=1, ee=None, D384='U'), kf30, kr30)

#FLIP(variants) bind to L:R:FADD
Parameter('kf31', 69.3329e-03)
Parameter('kr31', 0.0)
Parameter('kf32', 69.4022e-03)
Parameter('kr32', 0.08)
# FIXME: this pattern requires a dummy kr31 which is ultimately ignored
for flip_m, kf, kr, reversible in (zip(flip_monomers, (kf31,kf32), (kr31,kr32), (False,True))):
    rule = Rule('RLFADD_%s_Binding' % flip_m.name, FADD (rf=ANY, fe=None) + flip_m (fe=None, ee=None) | FADD (rf=ANY, fe=1) % flip_m (fe=1, ee=None), kf, kr)
    if reversible is False:
        rule.is_reversible = False
        rule.rule_expression.is_reversible = False
        rule.rate_reverse = None

pC8_HomoD   = pC8 (fe=ANY, ee=1, D384='U') % pC8   (fe=ANY, ee=1, D384='U')
pC8_HeteroD = pC8 (fe=ANY, ee=1, D384='U') % flipL (fe=ANY, ee=1, D384='U')
p43_HomoD   = pC8 (fe=ANY, ee=1, D384='C') % pC8   (fe=ANY, ee=1, D384='C')
p43_HeteroD = pC8 (fe=ANY, ee=1, D384='C') % flipL (fe=ANY, ee=1, D384='C')

#L:R:FADD:C8 dimerizes
Parameter('kf33', 2.37162)
Parameter('kr33', 0.1)
Parameter('kc33', 1e-05)
Rule('RLFADD_C8_C8_Binding', pC8 (fe=ANY, ee=None, D384='U') + pC8 (fe=ANY, ee=None, D384='U') | pC8_HomoD, kf33, kr33)

#L:R:FADD:C8 L:R:FADD:FLIP(variants) dimerizes
Parameter('kf34', 4.83692)
Parameter('kr34', 0)
Parameter('kf35', 2.88545)
Parameter('kr35', 1)
# FIXME: this pattern requires a dummy kr31 which is ultimately ignored
for flip_m, kf, kr, reversible in (zip(flip_monomers, (kf34,kf35), (kr34,kr35), (False,True))):
    rule = Rule('RLFADD_C8_%s_Binding' % flip_m.name, pC8 (fe=ANY, ee=None, D384='U') + flip_m (fe=ANY, ee=None) | pC8 (fe=ANY, ee=1, D384='U') % flip_m (fe=ANY, ee=1), kf, kr)
    if reversible is False:
        rule.is_reversible = False
        rule.rule_expression.is_reversible = False
        rule.rate_reverse = None

Parameter('kc36', 0.223046e-3)
#Homodimer catalyses Homodimer ?: no p18 is released. Only this "cleaved" p43 homoD is the product that will transform into a p18 + L:R:FADD in later reaction.
Rule('HomoD_cat_HomoD', pC8_HomoD + pC8_HomoD >> pC8_HomoD + p43_HomoD, kc36)
#Homodimer catalyses Heterodimer ?????
Rule('HomoD_cat_HeteroD', pC8_HomoD + pC8_HeteroD >> pC8_HomoD + p43_HeteroD, kc36)

Parameter('kc37', 0.805817e-3)
#Heterodimer catalyses Heterodimer ?????
Rule('HeteroD_cat_HeteroD', pC8_HeteroD + pC8_HeteroD >> pC8_HeteroD + p43_HeteroD, kc37)
#Heterodimer catalyses Homodimer ?????
Rule('HeteroD_cat_HomoD', pC8_HeteroD + pC8_HomoD >> pC8_HeteroD + p43_HomoD, kc37)

Parameter('kc38', 1.4888e-3)
#Cleaved Homodimer catalyses Homodimer ?????
Rule('Cl_HomoD_cat_HomoD', p43_HomoD + pC8_HomoD >> p43_HomoD + p43_HomoD, kc38)
#Cleaved HomoD catalyses Heterodimer ?????
Rule('Cl_HomoD_cat_HeteroD', p43_HomoD + pC8_HeteroD >> p43_HomoD + p43_HeteroD, kc38)

Parameter('kc39', 13.098e-3)
#Cleaved HeteroD catalyses Homodimer ?????
Rule('Cl_heteroD_cat_HomoD', p43_HeteroD + pC8_HomoD >> p43_HeteroD + p43_HomoD, kc39)
#Cleaved HeteroD catalyses Heterodimer ?????
Rule('Cl_heteroD_cat_HeteroD', p43_HeteroD + pC8_HeteroD >> p43_HeteroD + p43_HeteroD, kc39)

#Cleaved HomoD catalyses Cleaved HomoD to p18 and release L:R:FADD
Parameter('kc40', 0.999273e-3)
Rule('Cl_HomoD_cat_Cl_HomoD', pC8 (fe=ANY, ee=1, D384='C', D400='U') % pC8 (fe=ANY, ee=1, D384='C', D400='U') +
     FADD (rf=ANY, fe=2) % pC8 (fe=2, ee=3, D384='C', D400='U') % FADD (rf=ANY, fe=4) % pC8 (fe=4, ee=3, D384='C', D400='U') >>
     pC8 (fe=ANY, ee=1, D384='C', D400='U') % pC8 (fe=ANY, ee=1, D384='C', D400='U') +
     FADD (rf=ANY, fe=None) + FADD (rf=ANY, fe=None) + pC8 (fe=None, ee=1, D384='C',D400='C') % pC8 (fe=None, ee=1, D384='C',D400='C'), 
     kc40)

#Cleaved HeteroD catalyses Cleaved HomoD to p18 and release L:R:FADD
Parameter('kc41', 0.982109e-3)
Rule('Cl_HeteroD_cat_Cl_HomoD', pC8 (fe=ANY, ee=1, D384='C', D400='U') % flipL (fe=ANY, ee=1, D384='C') +
     FADD (rf=ANY, fe=2) % pC8 (fe=2, ee=3, D384='C', D400='U') % FADD (rf=ANY, fe=4) % pC8 (fe=4, ee=3, D384='C', D400='U') >>
     pC8 (fe=ANY, ee=1, D384='C', D400='U') % flipL (fe=ANY, ee=1, D384='C') +
     FADD (rf=ANY, fe=None) + FADD (rf=ANY, fe=None) + pC8 (fe=None, ee=1, D384='C',D400='C') % pC8 (fe=None, ee=1, D384='C',D400='C'), 
     kc41)
 
#Cleaved HomoD cleaves Bid ?????
Parameter('kc42', 0.0697394e-3)
Rule('Cl_Homo_cat_Bid', pC8 (fe=ANY, ee=1, D384='C', D400='U') % pC8 (fe=ANY, ee=1, D384='C', D400='U') + Bid () >>
     pC8 (fe=ANY, ee=1, D384='C', D400='U') % pC8 (fe=ANY, ee=1, D384='C', D400='U') + tBid (), kc42)

#Cleaved HeteroD cleaves Bid ?????
Parameter('kc43', 0.0166747e-3)
Rule('Cl_Hetero_cat_Bid', pC8 (fe=ANY, ee=1, D384='C', D400='U') % flipL (fe=ANY, ee=1, D384='C') + Bid () >>
     pC8 (fe=ANY, ee=1, D384='C', D400='U') % flipL (fe=ANY, ee=1, D384='C') + tBid (), kc43)

#p18 cleaves Bid ?????
Parameter('kc44', 0.0000479214e-3)
Rule('p18_Bid_cat', pC8 (fe=None, ee=1, D384='C',D400='C') % pC8 (fe=None, ee=1, D384='C',D400='C') + Bid () >> 
	pC8 (fe=None, ee=1, D384='C',D400='C') % pC8 (fe=None, ee=1, D384='C',D400='C') + tBid (), kc44) 


# Fig 4B

Observable('p18', pC8(fe=None, ee=1, D384='C',D400='C') % pC8(fe=None, ee=1, D384='C',D400='C'))
Observable('tBid_total', tBid() )




# generate initial conditions from _0 parameter naming convention
for m in model.monomers:
    ic_param = model.parameters.get('%s_0' % m.name, None)
    if ic_param is not None:
        sites = {}
        for s in m.sites:
            if s in m.site_states:
                sites[s] = m.site_states[s][0]
            else:
                sites[s] = None
        Initial(m(sites), ic_param)


####

if __name__ == '__main__':
    print(__doc__, "\n", model)
    print("""
NOTE: This model code is designed to be imported and programatically
manipulated, not executed directly. The above output is merely a
diagnostic aid.""")


####
# some of the model definition from the supplemental materials, for reference:

# ********** MODEL STATES
# %% Protein amounts are given in thousand molecules per cell.
# CD95L(0) = 1,500%% amount ligand
# CD95R(0) = 170.999%% amount CD95
# FADD(0) = 133.165%% amount FADD
# C8(0) = 200.168%% amount Procaspase-8
# FL(0) = 0.49995%% amount FLIP-Long
# FS(0) = 0.422%% amount FLIP-Short
# CD95RL(0) = 0%% amount of CD95-CD95L complexes
# CD95FADD(0) = 0%% amount of CD95-FADD complexes
# FADDC8(0) = 0%% amount Procaspase-8 bound to FADD
# FADDFL(0) = 0%% amount c-FLIPL bound to FADD
# FADDFS(0) = 0%% amount c-FLIPS bound to FADD
# C8heterodimer(0) = 0%% amount Procaspase-8/c-FLIPL heterodimers
# C8homodimer(0) = 0%% amount Procaspase-8 homodimers
# C8FSdimer(0) =0%% amount Procaspase-8/c-FLIPS heterodimers
# p43heterodimer(0) = 0%% amount p43/p41-Procaspase-8/p43-FLIP heterodimers
# p43homodimer(0) = 0%% amount p43/p41-Procaspase-8 homodimers
# p18(0)=0%% amount p18 formed
# apoptosissubstrate(0)=100
# cleavedsubstrate(0) = 0%% amount cleaved apoptosis substrate

# ********** MODEL VARIABLES
# p18total = 2 x p18
# p43Casp8total = 2 x p43homodimer + p43heterodimer
# procaspase8total = C8 + FADDC8 + C8heterodimer + 2 x C8homodimer + C8FSdimer
# c8total = p43Casp8total + procaspase8total + 2 x p18
# cleavedC8 = c8total - procaspase8total
# celldeath = cleavedsubstrate / 0.10875%% Model readout: percentage of dead cells

# ********** MODEL REACTIONS
# RCD95LBindCD95R = 7.0980e-002 x CD95L x CD95R
# RFADDBindCD95RL = 0.0844211 x CD95RL x FADD
# RC8BindCD95FADD = 0.00319838 x CD95FADD x C8
# RFLBindCD95FADD = 0.0693329 x CD95FADD x FL
# RFSBindCD95FADD = 0.0694022 x CD95FADD x FS
# RFADDC8Dissociate = 0.1 x FADDC8
# RFADDFSDissociate = 0.08 x FADDFS
# RFADDC8BindFADDC8 = 1.18581 x FADDC8 x FADDC8
# RFADDFLBindFADDC8 = 4.83692 x FADDC8 x FADDFL
# RFADDFSBindFADDC8 = 2.88545 x FADDC8 x FADDFS
# RC8FSdimerDissociate = 1 x C8FSdimer
# RC8homodimerDissociate = 0.1 x C8homodimer
# RC8homodimerCleaveC8homodimer = 0.000223046 x C8homodimer x C8homodimer
# RC8homodimerCleaveC8heterodimer = 0.000223046 x C8homodimer x C8heterodimer
# RC8heterodimerCleaveC8heterodimer = 0.000805817 x C8heterodimer x C8heterodimer
# RC8heterodimerCleaveC8homodimer = 0.000805817 x C8heterodimer x C8homodimer
# Rp43homodimerCleaveC8homodimer = 0.0014888 x p43homodimer x C8homodimer
# Rp43homodimerCleaveC8heterodimer = 0.0014888 x p43homodimer x C8heterodimer
# Rp43heterodimerCleaveC8homodimer = 0.013098 x p43heterodimer x C8homodimer
# Rp43heterodimerCleaveC8heterodimer = 0.013098 x p43heterodimer x C8heterodimer
# Rp43homodimerCleavep43homodimer = 0.000999273 x p43homodimer x p43homodimer
# Rp43heterodimerCleavep43homodimer = 0.000982109 x p43heterodimer x p43homodimer
# Rp43heterodimerCleaveApoptosisSubstrate = 1.66747e-005 x p43heterodimer x apoptosissubstrate
# Rp43homodimerCleaveApoptosisSubstrate = 6.97394e-005 x p43homodimer x apoptosissubstrate
# Rp18CleaveApoptosisSubstrate = 4.79214e-08 x p18 x apoptosissubstrate
