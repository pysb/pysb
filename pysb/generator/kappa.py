import pysb
import sympy
from re import sub
import warnings

# Alias basestring under Python 3 for forwards compatibility
try:
    basestring
except NameError:
    basestring = str

class KappaGenerator(object):

    # Dialect can be either 'complx' or 'kasim' (default)
    def __init__(self, model, dialect='kasim', _warn_no_ic=True):
        self.model = model
        self.__content = None
        self.dialect = dialect
        self._warn_no_ic = _warn_no_ic

    def get_content(self):
        if self.__content == None:
            self.generate_content()
        return self.__content

    def generate_content(self):
        self.__content = ''
        #self.generate_compartments()

        # Agent declarations appear to be required in kasim
        # but prohibited in complx
        if (self.dialect == 'kasim'):
            self.generate_molecule_types() 
            # Parameters, variables, and expressions are allowed in kasim
            self.generate_parameters()

        self.generate_reaction_rules()
        self.generate_observables()
        self.generate_species()

    def generate_parameters(self):
        names = [x.name for x in self.model.parameters] + \
                [x.name for x in self.model.expressions]
        for p in self.model.parameters:
            self.__content += "%%var: '%s' %e\n" % (p.name, p.value)
        for e in self.model.expressions:
            sym_names = [x.name for x in e.expr.atoms(sympy.Symbol)]
            str_expr = str(expression_to_muparser(e))
            for n in sym_names:
                str_expr = str_expr.replace(n,"'%s'"%n)
            self.__content += "%%var: '%s' %s\n" % (e.name, str_expr)
        self.__content += "\n"

    #def generate_compartments(self):
    #    self.__content += "begin compartments\n"
    #    for c in self.model.compartments:
    #        if c.parent is None:
    #            parent_name = ''
    #        else:
    #            parent_name = c.parent.name
    #        self.__content += ("  %s  %d  %f  %s\n" %
    #                          (c.name, c.dimension, c.size, parent_name))
    #    self.__content += "end compartments\n\n"

    def generate_molecule_types(self):
        for m in self.model.monomers:
            site_code = ','.join([format_monomer_site(m, s) for s in m.sites])
            self.__content += "%%agent: %s(%s)\n" % (m.name, site_code)
        self.__content += "\n"

    def generate_reaction_rules(self):
        # +1 for the colon
        #max_length = max(len(r.name) for r in self.model.rules) + 1
        for r in self.model.rules:
            label = "'" + r.name + "'"
            reactants_code = format_reactionpattern(r.reactant_pattern)
            products_code  = format_reactionpattern(r.product_pattern)
            arrow = '->'

            if isinstance(r.rate_forward,pysb.core.Expression) and \
               self.dialect == 'complx':
                raise KappaException("Expressions are not supported by complx.")
            # Get the rate code depending on the dialect
            if self.dialect == 'kasim':
                f_rate_code = "'" + r.rate_forward.name + "'"
            else:
                f_rate_code = float(r.rate_forward.value)

            self.__content += ("%s %s %s %s @ %s") % \
                (label, reactants_code, arrow, products_code, f_rate_code)
            self.__content += "\n"

            # Add the reverse reaction
            if r.is_reversible:
                if isinstance(r.rate_reverse,pysb.core.Expression) and \
                   self.dialect == 'complx':
                    raise KappaException("Expressions are not supported by "
                                         "complx.")
                # Get the rate code depending on the dialect
                if self.dialect == 'kasim':
                    r_rate_code = "'" + r.rate_reverse.name + "'"
                else:
                    r_rate_code = float(r.rate_reverse.value)

                label = "'" + r.name + '_rev' + "'"
                self.__content += ("%s %s %s %s @ %s") % \
                      (label, products_code, arrow, reactants_code, r_rate_code)
                self.__content += "\n"

        self.__content += "\n"

    def generate_observables(self):
        if not self.model.observables:
            return
        for obs in self.model.observables:
            name = "'" + obs.name + "'"
            observable_code = format_reactionpattern(obs.reaction_pattern)
            # In the near future (KaSim 4.0), the observable syntax will
            # require pipe characters around the expression. However, for the
            # time being we'll stick with the old syntax for backwards
            # compatibility
            #if self.dialect == 'kasim':
            #    self.__content += ("%%obs: %s |%s|\n") % \
            #                      (name, observable_code)
            #else:
            #    self.__content += ("%%obs: %s %s\n") % (name, observable_code)
            self.__content += ("%%obs: %s %s\n") % (name, observable_code)

        self.__content += "\n"

    def generate_species(self):
        if self._warn_no_ic and not self.model.initial_conditions:
            warnings.warn("Warning: No initial conditions.")

        species_codes = [format_complexpattern(cp)
                         for cp, param in self.model.initial_conditions]
        #max_length = max(len(code) for code in species_codes)
        for i, code in enumerate(species_codes):
            param = self.model.initial_conditions[i][1]
            #self.__content += ("%%init:  %-" + str(max_length) + \
            #                  "s   %s\n") % (code, param.name)
            if (self.dialect == 'kasim'):
                self.__content += ("%%init: '%s' %s\n") % (param.name, code)
            else:
                if isinstance(param,pysb.core.Expression):
                    raise KappaException("complx does not support Expressions.")
                # Switched from %g (float) to %d (int) because kappa didn't
                # like scientific notation for large integers
                self.__content += "%%init: %10d * %s\n" % (param.value, code)
        self.__content += "\n"


def format_monomer_site(monomer, site):
    ret = site
    if site in monomer.site_states:
        for state in monomer.site_states[site]:
            ret += '~' + state
    return ret

def format_reactionpattern(rp):
    return ','.join([format_complexpattern(cp) for cp in rp.complex_patterns])

def format_complexpattern(cp):
    ret = ','.join([format_monomerpattern(mp) for mp in cp.monomer_patterns])
    if cp.compartment is not None:
        ret = '@%s:%s' % (cp.compartment.name, ret)
    return ret

def format_monomerpattern(mp):
    # sort sites in the same order given in the original Monomer
    site_conditions = sorted(mp.site_conditions.items(),
                             key=lambda x: mp.monomer.sites.index(x[0]))
    site_pattern_code = ','.join([format_site_condition(site, state)
                                  for (site, state) in site_conditions])
    ret = '%s(%s)' % (mp.monomer.name, site_pattern_code)
    if mp.compartment is not None:
        ret = '%s@%s' % (ret, mp.compartment.name)
    return ret

def format_site_condition(site, state):
    # If the state/bond is unspecified
    if state == None:
        state_code = ''
    # If there is a bond number
    elif isinstance(state, int):
        state_code = '!' + str(state)
    # If there is a lists of bonds to the site (not supported by Kappa)
    elif isinstance(state, list):
        raise KappaException("Kappa generator does not support multiple bonds "
                              "to a single site.")
    # Site with state
    elif isinstance(state, basestring):
        state_code = '~' + state
    # Site with state and a bond
    elif isinstance(state, tuple):
        # If the bond is ANY
        if state[1] == pysb.ANY:
            state_code = '~%s!_' % state[0]
        # If the bond is WILD
        elif state[1] == pysb.WILD:
            state_code = '~%s?' % state[0]
        # If the bond is a number
        elif type(state[1]) == int:
            state_code = '~%s!%s' % state
        # If it's something else, raise an Exception
        else:
            raise KappaException("Kappa generator has encountered an unknown "
                                 "element in a site state/bond tuple: (%s, %s)"
                                 % state)
    # Site bound to ANY
    elif state == pysb.ANY:
        state_code = '!_'
    # Site bound to WILD
    elif state == pysb.WILD:
        state_code = '?'
    # Something else
    else:
        raise KappaException("Kappa generator has encountered an unknown "
                        "element in a rule pattern site condition.")
    return '%s%s' % (site, state_code)

def expression_to_muparser(expression):
    """Render the Expression as a Kappa compatible string."""
    # sympy.printing.sstr is the preferred way to render an Expression as a
    # string (rather than, e.g., str(Expression.expr) or repr(Expression.expr).
    # Note: "For large expressions where speed is a concern, use the setting
    # order='none'"
    code = sympy.printing.sstr(expression.expr, order='none')
    code = code.replace('\n @', '')
    code = code.replace('**', '^')
    # kasim syntax cannot handle Fortran scientific notation (must use 'e'
    # instead of 'd')
    code = sub('(?<=[0-9])d', 'e', code)
    return code

class KappaException(Exception):
    pass
