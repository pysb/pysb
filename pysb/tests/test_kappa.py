from pysb.testing import *
from pysb import *
from pysb.kappa import *
from pysb.bng import generate_network
import subprocess 
from re import split

@with_model
def test_kappa_expressions():
    Monomer('A',['site'],{'site': ['u']})
    Parameter('two',2)
    Parameter('kr',0.1)
    Parameter('num_A',1000)
    Expression('kf',1e-5/two)
    Initial(A(site=('u')),num_A)
    Rule('dimerize',
         A(site='u') + A(site='u') <> A(site=('u', 1)) % A(site=('u',1)),
         kf, kr)
    # Accommodates Expression in kappa simulation
    ok_(run_kasim(model, time=0, cleanup=True))

    Rule('degrade_dimer', A(site=('u', ANY)) >> None, kr)
    Observable('dimer', A(site=('u', ANY)))
    # Accommodates site with explicit state and arbitrary bond
    ok_(run_kasim(model, time=0, cleanup=True))

@with_model
def test_kappa_wild():
    Monomer('A',['site'])
    Monomer('B',['site'])
    Initial(A(site=None), Parameter('A_0', 100))
    Initial(B(site=None), Parameter('B_0', 100))
    Initial(A(site=1) % B(site=1), Parameter('AB_0', 1000))
    Rule('deg_A', A(site=pysb.WILD) >> None, Parameter('k', 1))
    ok_(run_kasim(model, time=0, cleanup=True))

def test_exported_kappa_file():
    m = """from pysb import *
Model()
Monomer('A',['site'],{'site': ['u']})
Parameter('two',2)
Parameter('kr',0.1)
Parameter('num_A',1000)
Expression('kf',1e-5/two)
Initial(A(site=('u')),num_A)
Rule('dimerize',A(site='u')+A(site='u') <> A(site=('u',1))%A(site=('u',1)),kf,kr)
Rule('degrade_dimer',A(site=('u',ANY)) >> None,kr)
Observable('dimer',A(site=('u',ANY)))
Observable('total_A_patterns',A(site=('u',WILD)))"""
    wf = open('test.py','w')
    wf.write(m)
    wf.close()
    commands = 'python -m pysb.export test.py kappa > test.ka; KaSim -i test.ka -e 0'
    # tests kappa WILD pattern and kappa syntax generation
    ok_(subprocess.check_call(commands, shell=True) == 0)
    subprocess.call('rm test.ka; rm test.py*', shell=True)
