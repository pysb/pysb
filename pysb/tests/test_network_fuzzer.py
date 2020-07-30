import string
import random
import numpy
import sympy


from pysb import (
    Monomer, Model, Rule, Parameter, ReactionPattern, \
    ComplexPattern, MonomerPattern, Compartment, Observable, RuleExpression,\
    Expression, Initial
)

import pysb.bng

ALLOWED_LETTERS = string.ascii_letters

UNARY_FUNS = [
    sympy.functions.exp,
    sympy.functions.log,
    sympy.functions.Abs,
    sympy.functions.cos,
    sympy.functions.sin,
    sympy.functions.tan,
    sympy.functions.sqrt,
    lambda x: sympy.core.Pow(x, -1)  # division
]

BINARY_FUNS = [
    sympy.core.Add,
    sympy.core.Mul,
    sympy.core.Pow
]

NUMBER_TRIALS = 1e4

LENGTH_NAMES = 10

# COMPARTMENTS
LENGTH_COMP_NAME = LENGTH_NAMES
PROB_NEW_COMPARTMENT = 0.2
PROB_COMPARTMENT_PARENT = 0.3

# PARAMETERS
LENGTH_PAR_NAME = LENGTH_NAMES
PROB_NEW_PARAMETER = 0.95

# SITE CONDITIONS
NUMBER_CONDITIONS = 3

# REACTION PATTERS
NUMBER_REACTANTS = 2

# COMPLEX PATTERNS
NUMBER_BONDS = 2
MAX_BONDNUMBERS = 20

# MONOMER PATTERNS
NUMBER_PROTOMERS = 2
PROB_MATCH_ONCE = 0.1

# RULES
LENGTH_RULE_NAME = LENGTH_NAMES
PROB_REVERSIBLE = 0.2
PROB_MOVE_CONNECTED = 0.1
PROB_DELETE_MOL = 0.1

# MONOMERS
LENGTH_MONO_NAME = LENGTH_NAMES
PROB_NEW_MONO = 0.2

# OBSERVABLES
LENGTH_OBS_NAME = LENGTH_NAMES
PROB_NEW_OBS = 0.5
PROB_MATCH_SPECIES = 0.5

# MONOMER SITES
NUMBER_SITES = 5
LENGTH_SITE_NAME = 1

# SITE STATES
PROB_STATES = 0.7
NUMBER_STATES = 3
LENGTH_STATE_NAME = 1

# EXPRESSION
LENGTH_EXPR_NAME = LENGTH_NAMES
PROB_NEW_EXPR = 0.5
EXPR_WEIGHT_PAR = 10
EXPR_WEIGHT_OBS = 10
EXPR_WEIGHT_EXPR = 10
EXPR_WEIGHT_UNARY = 30
EXPR_WEIGHT_BINARY = 20

# RATE
PROB_PARAMETER = 0.9


def poisson(lam):
    return numpy.random.poisson(lam, 1)[0]


def bernoulli(p):
    return numpy.random.binomial(1, p)


def random_name(l_name=LENGTH_NAMES):
    return ''.join(random.choices(ALLOWED_LETTERS, k=max(poisson(l_name), 1)))


def random_name_list(l_names, l_name):
    return [random_name(l_name) for _ in range(poisson(l_names))]


def random_site_states(sites, l_states, p_state):
    return {
        site: random_name_list(l_states, l_name=LENGTH_STATE_NAME)
        for site in sites
        if bernoulli(p_state)
    }


def add_bond_to_random_site(mono, site_conditions, bond_n):
    if not mono.sites:
        raise ValueError('Cannot add bond to monomer without sites.')

    bond_site = random.choice(mono.sites)
    if bond_site in site_conditions:
        condition = site_conditions[bond_site]
        if condition is None:
            site_conditions[bond_site] = bond_n
        if isinstance(condition, str) or isinstance(condition, int):
            site_conditions[bond_site] = (condition, bond_n)
        elif isinstance(condition, tuple) and isinstance(condition[0], int):
            site_conditions[bond_site] = (bond_n, *condition)
        elif isinstance(condition, tuple) and isinstance(condition[0], str) \
                and isinstance(condition[1], int):
            site_conditions[bond_site] = (condition[0], (bond_n, condition[1]))
        elif isinstance(condition, tuple) and isinstance(condition[0], str) \
                and isinstance(condition[1], tuple):
            site_conditions[bond_site] = (condition[0], (bond_n,
                                                         *condition[1]))
        else:
            RuntimeError('Encountered unsupported site conditions ' +
                         str(site_conditions))
    else:
        site_conditions[bond_site] = bond_n


def add_random_bond(mps, bond_mp_idx):
    bond_number = random.randint(0, MAX_BONDNUMBERS)
    for idx in bond_mp_idx:
        mp = mps[idx]
        add_bond_to_random_site(mp['monomer'], mp['site_conditions'],
                                bond_number)


def add_random_bonds(mps, l_bonds=NUMBER_BONDS):
    for _ in range(poisson(l_bonds)):
        mp_idx = random.choices(range(len(mps)), k=2)
        add_random_bond(mps, mp_idx)


def add_connecting_bonds(mps):
    for imp in range(len(mps)-1):
        add_random_bond(mps, [imp, imp+1])


def random_site_conditions(mono, l_cond=NUMBER_CONDITIONS, explicit=False):
    if not mono.site_states:
        return {}

    if explicit:
        site_states = mono.site_states.items()
    else:
        site_states = random.choices(
            list(mono.site_states.items()), k=poisson(l_cond)
        )
    site_conditions = {
        site: random.choice(states)
        for site, states in site_states
    }
    if explicit:
        for site in mono.sites:
            if site not in site_conditions:
                site_conditions[site] = None
    return site_conditions


def random_parameter(model, p_new=PROB_NEW_PARAMETER):
    if not bernoulli(p_new) and model.parameters:
        return random.choice(model.parameters)

    return Parameter(random_name())


def random_monomer_pattern(model, explicit):
    mono = random_monomer(model)
    site_conditions = random_site_conditions(mono, explicit=explicit)
    compartment = random_compartment(model)
    return {'monomer': mono, 'site_conditions': site_conditions,
            'compartment': compartment}


def random_complex_pattern(model, l_monos=NUMBER_PROTOMERS,
                           p_match=PROB_MATCH_ONCE, explicit=False):
    n_monos = poisson(l_monos)
    if explicit:
        n_monos = max(n_monos, 1)
    if n_monos == 0:
        return None

    mps = [random_monomer_pattern(model, explicit=explicit)
           for _ in range(n_monos)]
    if model.compartments:
        compartment = random.choice(model.compartments)
    else:
        compartment = None

    if explicit:
        add_connecting_bonds(mps)
    else:
        add_random_bonds(mps)

    mps = [MonomerPattern(**mp) for mp in mps]

    return ComplexPattern(mps, compartment=compartment,
                          match_once=bernoulli(p_match))


def random_reaction_pattern(model, l_react=NUMBER_REACTANTS):
    n_reactants = poisson(l_react)
    if n_reactants == 0:
        reactants = [None]
    else:
        reactants = [random_complex_pattern(model) for _ in range(n_reactants)]
    return ReactionPattern(reactants)


def random_symbolic(model, w_par=EXPR_WEIGHT_PAR, w_obs=EXPR_WEIGHT_OBS,
                    w_expr=EXPR_WEIGHT_EXPR, w_ufun=EXPR_WEIGHT_UNARY,
                    w_bfun=EXPR_WEIGHT_BINARY):
    operation_weights = {
        'par': (lambda m: random_parameter(m), w_par),
        'obs': (lambda m: random_observable(m), w_obs),
        'expr': (lambda m: random_expression(m), w_expr),
        'ufun': (lambda m: random.choice(UNARY_FUNS)(random_symbolic(m)),
                 w_ufun),
        'bfun': (lambda m: random.choice(BINARY_FUNS)(random_symbolic(m),
                                                      random_symbolic(m)),
                 w_bfun),
    }

    ops = [val[0] for val in operation_weights.values()]
    weights = [val[1] for val in operation_weights.values()]

    return random.choices(ops, weights, k=1)[0](model)


def random_rate(model, p_par=PROB_PARAMETER):
    if bernoulli(p_par):
        return random_parameter(model)
    else:
        return random_expression(model)


def random_rule(model, p_reversible=PROB_REVERSIBLE,
                p_move_connected=PROB_MOVE_CONNECTED,
                p_delete_mol=PROB_DELETE_MOL):
    name = random_name()

    kwargs = {
        'rate_forward': random_rate(model),
        'move_connected': bernoulli(p_move_connected),
        'delete_molecules': bernoulli(p_delete_mol),
    }

    rp = random_reaction_pattern(model)
    pp = random_reaction_pattern(model)

    rule_expression = RuleExpression(rp, pp, bernoulli(p_reversible))
    if rule_expression.is_reversible:
        kwargs['rate_reverse'] = random_rate(model)

    return Rule(name, rule_expression, **kwargs)


def random_compartment(model, p_new=PROB_NEW_COMPARTMENT,
                       p_parent=PROB_COMPARTMENT_PARENT):

    if not bernoulli(p_new) and model.compartments:
        return random.choice(model.compartments)

    name = random_name()

    if bernoulli(p_parent) and model.compartments:
        parent = random.choice(model.compartments)
    else:
        parent = None

    dimension = random.choice([2, 3])

    return Compartment(name, parent, dimension, random_parameter(model))


def random_monomer(model, p_new=PROB_NEW_MONO, l_sites=NUMBER_SITES,
                   l_states=NUMBER_STATES, p_state=PROB_STATES):
    if not bernoulli(p_new) and model.monomers:
        return random.choice(model.monomers)

    name = random_name()
    sites = random_name_list(l_sites, LENGTH_SITE_NAME)
    site_states = random_site_states(sites, l_states, p_state)

    return Monomer(name, sites, site_states)


def random_observable(model, p_new=PROB_NEW_OBS, p_species=PROB_MATCH_SPECIES):
    if not bernoulli(p_new) and model.observables:
        return random.choice(model.observables)

    name = random_name()
    obs = random_reaction_pattern(model)
    if bernoulli(p_species):
        match = 'species'
    else:
        match = 'molecules'
    return Observable(name, obs, match)


def random_expression(model, p_new=PROB_NEW_EXPR):
    if not bernoulli(p_new) and model.expressions:
        return random.choice(model.expressions)
    name = random_name()
    expr = random_symbolic(model)
    return Expression(name, expr)


def random_initial(model):
    pattern = random_complex_pattern(model, explicit=True)
    value = random_rate(model)
    return Initial(pattern, value)


def validate_random_generation(generation_fun, n_trials):
    n_trials = int(n_trials)
    n_valid = 0
    for _ in range(n_trials):
        try:
            pysb.SelfExporter.cleanup()
            m = Model()
            generation_fun(m)
            n_valid += 1
        except ValueError:
            pass

    assert n_valid > 0
    pct_valid = n_valid / n_trials * 100
    print(pct_valid + '% of generations did not fail')


def test_monomers():
    validate_random_generation(lambda m: random_monomer(m, p_new=1),
                               NUMBER_TRIALS)


def test_compartments():
    validate_random_generation(lambda m: random_compartment(m, p_new=1),
                               NUMBER_TRIALS)


def test_observables():
    validate_random_generation(lambda m: random_observable(m, p_new=1),
                               NUMBER_TRIALS)


def test_expressions():
    validate_random_generation(lambda m: random_expression(m, p_new=1),
                               1e3)


def test_initials():
    validate_random_generation(random_initial, 1e3)


def test_rules():
    validate_random_generation(random_rule, 1e3)


def generate_random_model(m):
    for _ in range(3):
        random_rule(m)

    for _ in range(3):
        random_initial(m)

    pysb.bng.generate_equations(m, verbose=True)


def test_models():
    validate_random_generation(generate_random_model, 1e3)


