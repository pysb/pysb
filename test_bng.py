import pysb
import pysb.bng
import model_simple_egfr

pysb.bng.pkg_path = '/usr/local/share/bionetgen'


ydot = pysb.bng.generate_equations(model_simple_egfr.bng_content)

for s in sorted(ydot.keys()):
    print 'ds%d/dt = %s' % (s, ydot[s])

