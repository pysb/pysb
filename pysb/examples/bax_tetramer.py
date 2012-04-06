from pysb import *

# Demonstration of the proper representation of BAX-style tetramers,
# where the subunits dimerize and then tetramerize instead of
# assembling sequentially.

Model()

# Each BAX-BAX bond must always involve a t1 site on one monomer and a
# t2 site on the other.
Monomer('BAX', ['t1', 't2', 'inh'])
Monomer('MCL1', ['b'])

# Two lone monomers form a dimer.
Parameter('kdimf', 1e-6)
Parameter('kdimr', 1e-7)
Rule('bax_dim',
     BAX(t1=None, t2=None) + BAX(t1=None, t2=None) <>
     BAX(t1=1, t2=None) % BAX(t1=None, t2=1),
     kdimf, kdimr)

# Two lone dimers form a tetramer, with a higher rate than the dimerization.
Parameter('ktetf', 1e-3)
Parameter('ktetr', 1e-4)
Rule('bax_tet',
     BAX(t1=1, t2=None) % BAX(t1=None, t2=1) + BAX(t1=2, t2=None) % BAX(t1=None, t2=2) <>
     BAX(t1=1, t2=3) % BAX(t1=4, t2=1) % BAX(t1=2, t2=4) % BAX(t1=3, t2=2),
     ktetf, ktetr)

# An inhibitory protein can bind to a BAX subunit at any time.
Parameter('kbaxmcl1f', 1e-5)
Parameter('kbaxmcl1r', 1e-6)
Rule('bax_inh_mcl1',
     BAX(inh=None) + MCL1(b=None) <>
     BAX(inh=1)    % MCL1(b=1),
     kbaxmcl1f, kbaxmcl1r)

# Initial conditions
Parameter('BAX_0', 1e3)
Initial(BAX(t1=None, t2=None, inh=None), BAX_0)
Parameter('MCL1_0', 1e2)
Initial(MCL1(b=None), MCL1_0)

# We must fully specify all four BAX-BAX bonds, otherwise the pattern
# is too loose, match a given species multiple times (beyond the
# factor of four expected due to the rotational symmetry of the
# tetramer), resulting in erroneously high values.
Observe('BAX4', BAX(t1=1, t2=3) % BAX(t1=4, t2=1) % BAX(t1=2, t2=4) % BAX(t1=3, t2=2))
# Same all-bonds requirement here.  However since the BAX tetramer is
# considered inhibited when even one subunit has an inhibitor bound,
# we only need to explicitly write inh=ANY on one of the monomer
# patterns.
Observe('BAX4_inh', BAX(inh=ANY, t1=1, t2=3) % BAX(t1=4, t2=1) % BAX(t1=2, t2=4) % BAX(t1=3, t2=2))


if __name__ == '__main__':
    from pysb.generator.bng import BngGenerator
    gen = BngGenerator(model)
    print gen.get_content()
    print ""
    print "begin actions"
    print "  generate_network({overwrite=>1});"
    print "  simulate_ode({t_end=>21600,n_steps=>360});" # 6 hours, 1-minute steps
    print "end actions"
