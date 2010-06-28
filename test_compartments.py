from pysb import *
import logging

logging.basicConfig()
complog = logging.getLogger("compartments_file")
complog.setLevel(logging.DEBUG)

complog.debug("starting Model")
Model()

complog.debug("setting Compartment")
Compartment('extra_cellular', dimension=3, size=1,   parent=None)
Compartment('ec_membrane', dimension=2, size=1.0, parent=extra_cellular)
Compartment('cytoplasm', dimension=3, size=1.0, parent=ec_membrane)
Compartment('endo_membrane', dimension=2, size=1, parent=cytoplasm)
Compartment('endosome', dimension=3, size=1, parent=endo_membrane)


complog.debug("setting Monomers")
Monomer('egf', ['R','test'], {'test':['up', 'down']})
Monomer('egfr', ['L', 'D', 'C'], {'C':['on', 'off']})
Monomer('shc', ['L', 'A'], {'A':['on', 'off']})

Parameter('K_egfr_egf_F', 1.2)
Parameter('K_egfr_egf_R', 1.2)
Rule('egfr_egf',
     egfr(L=None) ** ec_membrane + egf(R=None) <>
     egfr(L=1) ** ec_membrane    % egf(R=1),
     K_egfr_egf_F, K_egfr_egf_R)

Parameter('K_egfr_endo', 0.5)
Rule('egfr_endocytosis',
     (egfr(L=1) % egf(R=1)) ** ec_membrane >>
     (egfr(L=1) % egf(R=1)) ** endo_membrane,
     K_egfr_endo)

Observe('egfr_total', egfr())
Observe('egfr_egf_endo', (egfr(L=1) % egf(R=1)) ** endo_membrane)

Parameter('egf_0', 10000)
Parameter('egfr_0', 500)
Initial(egf(R=None, test='up') ** extra_cellular, egf_0)
Initial(egfr(L=None, D=None, C='off') ** ec_membrane, egfr_0)

print extra_cellular
print ec_membrane
print cytoplasm
print "====="
print egf
print egfr
print shc
print "====="
print egfr_egf
print egfr_endocytosis

#####

from pysb.integrate import odesolve
from pylab import *

t = linspace(0, 3600, 3600/60)  # 1 hour, in minutes
y = odesolve(model, t)

plot(t, array([y['egfr_total'], y['egfr_egf_endo']]).T)
show()


#from pysb.generator.bng import BngGenerator
#gen = BngGenerator(model)
#print gen.get_content()
#print ""
#print "begin actions"
#print "  generate_network({overwrite=>1});"
#print "  simulate_ode({t_end=>21600,n_steps=>360});" # 6 hours, 1-minute steps
#print "end actions"
