import pysb
import pysb.bng
import model_simple_egfr

pysb.bng.pkg_path = '/home/jmuhlich/development/BioNetGen-2.0.48'


pysb.bng.generate_equations(model_simple_egfr.bng_content)


