import inspect
import warnings
import pysb
import sympy

# Alias basestring under Python 3 for forwards compatibility
try:
    basestring
except NameError:
    basestring = str

class BngGenerator(object):

    def __init__(self, model):
        self.model = model
        if self.model.has_synth_deg():
            self.model.enable_synth_deg()
        self.__content = None

    def get_content(self):
        if self.__content == None:
            self.generate_content()
        return self.__content

    def generate_content(self):
        self.__content = "begin model\n"
        self.generate_parameters()
        self.generate_compartments()
        self.generate_molecule_types()
        self.generate_observables()
        self.generate_functions()
        self.generate_species()
        self.generate_reaction_rules()
        self.__content += "end model\n"

    def generate_parameters(self):
        exprs = self.model.expressions_constant()
        if not self.model.parameters and not exprs:
            return
        self.__content += "begin parameters\n"
        max_length = max(len(p.name) for p in
                         self.model.parameters | self.model.expressions)
        for p in self.model.parameters:
            self.__content += (("  %-" + str(max_length) + "s   %e\n") %
                               (p.name, p.value))
        for e in exprs:
            self.__content += (("  %-" + str(max_length) + "s   %s\n") %
                               (e.name, expression_to_muparser(e)))
        self.__content += "end parameters\n\n"

    def generate_compartments(self):
        if not self.model.compartments:
            return
        self.__content += "begin compartments\n"
        for c in self.model.compartments:
            if c.parent is None:
                parent_name = ''
            else:
                parent_name = c.parent.name
            if c.size is None:
                size = "1.0"
            else:
                size = c.size.name
            self.__content += ("  %s  %d  %s  %s\n") % (c.name, c.dimension, size, parent_name)
        self.__content += "end compartments\n\n"        

    def generate_molecule_types(self):
        if not self.model.monomers:
            return
        self.__content += "begin molecule types\n"
        for m in self.model.monomers:
            site_code = ','.join([format_monomer_site(m, s) for s in m.sites])
            self.__content += "  %s(%s)\n" % (m.name, site_code)
        self.__content += "end molecule types\n\n"

    def generate_reaction_rules(self):
        if not self.model.rules:
            warn_caller("Model does not contain any rules")
            return
        self.__content += "begin reaction rules\n"
        max_length = max(len(r.name) for r in self.model.rules) + 1  # +1 for the colon
        for r in self.model.rules:
            label = r.name + ':'
            react_p = r.reactant_pattern
            prod_p = r.product_pattern
            if r.is_synth():
                source_mp = self.model.monomers['__source']()
                react_p = pysb.core.as_reaction_pattern(source_mp)
                prod_p += source_mp
            if r.is_deg():
                sink_mp = self.model.monomers['__sink']()
                prod_p = pysb.core.as_reaction_pattern(sink_mp)
            reactants_code = format_reactionpattern(react_p)
            products_code = format_reactionpattern(prod_p)
            arrow = '->'
            if r.is_reversible:
                arrow = '<->'
            self.__content += ("  %-" + str(max_length) + "s  %s %s %s    %s") % \
                (label, reactants_code, arrow, products_code, r.rate_forward.name)
            if r.is_reversible:
                self.__content += ', %s' % r.rate_reverse.name
            if r.delete_molecules:
                self.__content += ' DeleteMolecules'
            if r.move_connected:
                self.__content += ' MoveConnected'
            self.__content += "\n"
        self.__content += "end reaction rules\n\n"

    def generate_observables(self):
        if not self.model.observables:
            return
        max_length = max(len(name) for name in self.model.observables.keys())
        self.__content += "begin observables\n"
        for obs in self.model.observables:
            observable_code = format_reactionpattern(obs.reaction_pattern,
                                                     for_observable=True)
            self.__content += ("  %s %-" + str(max_length) + "s   %s\n") % \
                              (obs.match.title(), obs.name, observable_code)
        self.__content += "end observables\n\n"

    def generate_functions(self):
        exprs = self.model.expressions_dynamic()
        if not exprs:
            return
        max_length = max(len(e.name) for e in exprs) + 2
        self.__content += "begin functions\n"
        for i, e in enumerate(exprs):
            signature = e.name + '()'
            self.__content += ("  %-" + str(max_length) + "s   %s\n") % \
                (signature, expression_to_muparser(e))
        self.__content += "end functions\n\n"

    def generate_species(self):
        if not self.model.initial_conditions:
            warn_caller("Model does not contain any initial conditions")
            return
        species_codes = [format_complexpattern(cp) for cp, param in self.model.initial_conditions]
        max_length = max(len(code) for code in species_codes)
        self.__content += "begin species\n"
        for i, code in enumerate(species_codes):
            param = self.model.initial_conditions[i][1]
            self.__content += ("  %-" + str(max_length) + "s   %s\n") % (code, param.name)
        self.__content += "end species\n\n"

def format_monomer_site(monomer, site):
    ret = site
    if site in monomer.site_states:
        for state in monomer.site_states[site]:
            ret += '~' + state
    return ret

def format_reactionpattern(rp, for_observable=False):
    if for_observable is False:
        delimiter = ' + '
    else:
        delimiter = ' '
    return delimiter.join([format_complexpattern(cp) for cp in rp.complex_patterns])

def format_complexpattern(cp):
    ret = '.'.join([format_monomerpattern(mp) for mp in cp.monomer_patterns])
    if cp.compartment is not None:
        ret = '@%s:%s' % (cp.compartment.name, ret)
    if cp.match_once:
        ret = '{MatchOnce}' + ret
    return ret

def format_monomerpattern(mp):
    # sort sites in the same order given in the original Monomer
    site_conditions = sorted(mp.site_conditions.items(),
                             key=lambda x: mp.monomer.sites.index(x[0]))
    site_pattern_code = ','.join([format_site_condition(site, state) for (site, state) in site_conditions])
    ret = '%s(%s)' % (mp.monomer.name, site_pattern_code)
    if mp.compartment is not None:
        ret = '%s@%s' % (ret, mp.compartment.name)
    return ret

def format_site_condition(site, state):
    # empty
    if state == None:
        state_code = ''
    # single bond
    elif isinstance(state, int):
        state_code = '!' + str(state)
    # multiple bonds
    elif isinstance(state, list) and all(isinstance(s, int) for s in state):
        state_code = ''.join('!%d' % s for s in state)
    # state
    elif isinstance(state, basestring):
        state_code = '~' + state
    # state AND single bond
    elif isinstance(state, tuple):
        # bond is wildcard (zero or more unspecified bonds)
        if state[1] == pysb.WILD:
            state = (state[0], '?')
        elif state[1] == pysb.ANY:
            state = (state[0], '+')
        state_code = '~%s!%s' % state
    # one or more unspecified bonds
    elif state is pysb.ANY:
        state_code = '!+'
    # anything at all (usually you can leverage don't-care-don't-write, but in
    # some cases such as when a rule explicitly sets the state of site A but
    # conditions on site B, site A on the reactant side must be set to WILD)
    elif state is pysb.WILD:
        state_code = '!?'
    else:
        raise Exception("BNG generator has encountered an unknown element in a rule pattern site condition.")
    return '%s%s' % (site, state_code)

def warn_caller(message):
    caller_frame = inspect.currentframe()
    # walk up through the stack until we are outside of pysb
    stacklevel = 1
    module = inspect.getmodule(caller_frame)
    while module and module.__name__.startswith('pysb.'):
        stacklevel += 1
        caller_frame = caller_frame.f_back
        module = inspect.getmodule(caller_frame)
    warnings.warn(message, stacklevel=stacklevel)

def expression_to_muparser(expression):
    """Render the Expression as a muparser-compatible string."""

    # sympy.printing.sstr is the preferred way to render an Expression as a
    # string (rather than, e.g., str(Expression.expr) or repr(Expression.expr).
    # Note: "For large expressions where speed is a concern, use the setting
    # order='none'"
    code = sympy.printing.sstr(expression.expr, order='none')
    code = code.replace('\n     @', '')
    code = code.replace('**', '^')
    return code
