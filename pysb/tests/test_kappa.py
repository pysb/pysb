from pysb.testing import *
from pysb import *
from pysb.kappa import *
from pysb.bng import generate_network
import subprocess 
from re import split
import pygraphviz as pgv

@with_model
def test_kappa_expressions():
    Monomer('A',['site'],{'site': ['u']})
    Parameter('two',2)
    Parameter('kr',0.1)
    Parameter('num_A',1000)
    Expression('kf',1e-5/two)
    Initial(A(site=('u')),num_A)
    Rule('dimerize_fwd',
         A(site='u') + A(site='u') >> A(site=('u', 1)) % A(site=('u',1)), kf)
    Rule('dimerize_rev',
         A(site=('u', 1)) % A(site=('u',1)) >>
         A(site='u') + A(site='u'), kr)
    # Accommodates Expression in kappa simulation
    run_simulation(model, time=0)

    Rule('degrade_dimer', A(site=('u', ANY)) >> None, kr)
    Observable('dimer', A(site=('u', ANY)))
    # Accommodates site with explicit state and arbitrary bond
    run_simulation(model, time=0)

@with_model
def test_kappa_simulation_results():
    Monomer('A', ['b'])
    Monomer('B', ['b'])
    Initial(A(b=None), Parameter('A_0', 100))
    Initial(B(b=None), Parameter('B_0', 100))
    Rule('A_binds_B', A(b=None) + B(b=None) >> A(b=1) % B(b=1),
         Parameter('kf', 1))
    Rule('A_binds_B_rev', A(b=1) % B(b=1) >> A(b=None) + B(b=None),
         Parameter('kr', 1))
    Observable('AB', A(b=1) % B(b=1))
    npts = 200
    kres = run_simulation(model, time=100, points=npts)
    ok_(len(kres['time']) == npts + 1)
    ok_(len(kres['AB']) == npts + 1)
    ok_(kres['time'][0] == 0)
    ok_(sorted(kres['time'])[-1] == 100)

@with_model
def test_flux_map():
    """Test kappa simulation with flux map (returns tuple with graph)"""
    #from pysb.examples.robertson import model
    #model.parameters['A_0'].value = 100
    Monomer('A', ['b'])
    Monomer('B', ['a', 'c'])
    Monomer('C', ['b'])
    Parameter('k', 0.001)
    Rule('A_binds_B', A(b=None) + B(a=None) >> A(b=1) % B(a=1), k)
    Rule('C_binds_B', C(b=None) + B(c=None) >> C(b=1) % B(c=1), k)
    Observable('ABC', A(b=1) % B(a=1, c=2) % C(b=2))
    Initial(A(b=None), Parameter('A_0', 100))
    Initial(B(a=None, c=None), Parameter('B_0', 100))
    Initial(C(b=None), Parameter('C_0', 100))
    kasim_result = run_simulation(model, time=10, points=100, flux_map=True,
                                  output_dir='.', cleanup=True, verbose=False)
    ok_(len(kasim_result) == 2)
    simdata = kasim_result[0]
    ok_(len(simdata['time']) == 101)
    ok_(len(simdata['ABC']) == 101)
    ok_(simdata['time'][0] == 0)
    ok_(sorted(simdata['time'])[-1] == 10)
    fluxmap = kasim_result[1]
    ok_(isinstance(fluxmap, pgv.AGraph))

@with_model
def test_kappa_wild():
    Monomer('A',['site'])
    Monomer('B',['site'])
    Initial(A(site=None), Parameter('A_0', 100))
    Initial(B(site=None), Parameter('B_0', 100))
    Initial(A(site=1) % B(site=1), Parameter('AB_0', 1000))
    Rule('deg_A', A(site=pysb.WILD) >> None, Parameter('k', 1))
    Observable('A_', A())
    run_simulation(model, time=0)

@with_model
def test_contact_map_kasa():
    Monomer('A', ['b'])
    Monomer('B', ['b'])
    Rule('A_binds_B', A(b=None) + B(b=None) >> A(b=1) % B(b=1),
         Parameter('k_A_binds_B', 1))
    Observable('AB', A(b=1) % B(b=1))
    contact_map(model, cleanup=False, output_dir='.')

@with_model
def test_influence_map_kasa():
    Monomer('A', [])
    Monomer('B', ['active'], {'active': ['y', 'n']})
    Monomer('C', ['active'], {'active': ['y', 'n']})
    Initial(A(), Parameter('A_0', 100))
    Initial(B(active='n'), Parameter('B_0', 100))
    Initial(C(active='n'), Parameter('C_0', 100))
    Rule('A_activates_B',
         A() + B(active='n') >> A() + B(active='y'),
         Parameter('k_A_activates_B', 1))
    Rule('B_activates_C',
         B(active='y') + C(active='n') >> B(active='y') + C(active='y'),
         Parameter('k_B_activates_C', 1))
    influence_map(model, cleanup=True)

