import pysb
import sympy
import warnings
from sympy.printing import StrPrinter
from sympy.core import S
import collections
import re
import pysb.logging
from pysb.export import CompartmentsNotSupported
# Alias basestring under Python 3 for forwards compatibility
try:
    basestring
except NameError:
    basestring = str


class KappaGenerator(object):

    # Dialect can be either 'complx' or 'kasim' (default)
    def __init__(self, model, dialect='kasim', _warn_no_ic=True):
        if model and model.compartments:
            raise CompartmentsNotSupported()
        self.model = model
        self.__content = None
        self.dialect = dialect
        self._warn_no_ic = _warn_no_ic
        self._renamed_states = collections.defaultdict(dict)
        self._log = pysb.logging.get_logger(__name__)

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
        for p in self.model.parameters:
            self.__content += "%%var: '%s' %e\n" % (p.name, p.value)
        for e in self.model.expressions:
            str_expr = str(expression_to_muparser(e))
            self.__content += "%%var: '%s' %s\n" % (e.name, str_expr)
        self.__content += "\n"

    def generate_molecule_types(self):
        for m in self.model.monomers:
            site_code = ','.join([
                self.format_monomer_site(m, s)
                for s in m.sites])
            self.__content += "%%agent: %s(%s)\n" % (m.name, site_code)
        self.__content += "\n"

    def generate_reaction_rules(self):
        for r in self.model.rules:
            label = "'" + r.name + "'"
            reactants_code = self.format_reactionpattern(r.reactant_pattern)
            products_code = self.format_reactionpattern(r.product_pattern)
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
            observable_code = self.format_reactionpattern(obs.reaction_pattern)
            self.__content += "%%obs: %s |%s|\n" % (name, observable_code)

        self.__content += "\n"

    def generate_species(self):
        if self._warn_no_ic and not self.model.initial_conditions:
            warnings.warn("Warning: No initial conditions.")

        species_codes = [self.format_complexpattern(cp)
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

    def format_monomer_site(self, monomer, site):
        ret = site
        if site in monomer.site_states:
            new_states = []
            for state in monomer.site_states[site]:
                if re.match('_*[0-9]*$', state):
                    new_state = '__state' + state
                    while new_state in new_states:
                        new_state = '_' + new_state
                    self._renamed_states[monomer.name][state] = new_state
                    state = new_state
                new_states.append(state)

            ret += '{%s}' % ' '.join(new_states)
        if self._renamed_states[monomer.name]:
            self._log.info('Monomer "{}" states were renamed as '
                           'follows: {}'.format(
                monomer.name,
                self._renamed_states[monomer.name])
            )
        return ret

    def format_reactionpattern(self, rp):
        if not rp.complex_patterns:
            return '.'
        return ','.join(filter(None,
                               [self.format_complexpattern(cp)
                                for cp in rp.complex_patterns]))

    def format_complexpattern(self, cp):
        if cp is None:
            return '.'
        ret = ','.join(filter(None, [self.format_monomerpattern(mp)
                                     for mp in cp.monomer_patterns]))
        if cp.compartment is not None:
            ret = '@%s:%s' % (cp.compartment.name, ret)
        return ret

    def format_monomerpattern(self, mp):
        # sort sites in the same order given in the original Monomer
        site_conditions = sorted(mp.site_conditions.items(),
                                 key=lambda x: mp.monomer.sites.index(x[0]))
        site_pattern_code = ','.join([self.format_site_condition(
                                        site, state, mp.monomer.name)
                                      for (site, state) in site_conditions])
        ret = '%s(%s)' % (mp.monomer.name, site_pattern_code)
        if mp.compartment is not None:
            ret = '%s@%s' % (ret, mp.compartment.name)
        return ret

    def format_site_condition(self, site, state, mon_name):
        state = self._renamed_states.get(mon_name, {}).get(state, state)
        # If the state/bond is unspecified
        if state is None:
            state_code = '[.]'
        # If there is a bond number
        elif isinstance(state, int):
            state_code = '[%s]' % state
        # If there is a lists of bonds to the site (not supported by Kappa)
        elif isinstance(state, list):
            raise KappaException("Kappa generator does not support multiple "
                                 "bonds to a single site.")
        # Site with state
        elif isinstance(state, basestring):
            state_code = '{%s}[.]' % state
        # Site with state and a bond
        elif isinstance(state, tuple):
            # If the bond is ANY
            if state[1] == pysb.ANY:
                state_code = '{%s}[_]' % state[0]
            # If the bond is WILD
            elif state[1] == pysb.WILD:
                state_code = '{%s}[#]' % state[0]
            # If the bond is a number
            elif type(state[1]) == int:
                state_code = '{%s}[%s]' % state
            # If it's something else, raise an Exception
            else:
                raise KappaException("Kappa generator has encountered an "
                                     "unknown element in a site state/bond "
                                     "tuple: (%s, %s)" % state)
        # Site bound to ANY
        elif state == pysb.ANY:
            state_code = '[_]'
        # Site bound to WILD
        elif state == pysb.WILD:
            state_code = '[#]'
        # Something else
        else:
            raise KappaException("Kappa generator has encountered an unknown "
                                 "element in a rule pattern site condition.")
        return '%s%s' % (site, state_code)


# Old, stateless versions of these functions kept for API compatibility
def format_monomer_site(monomer, site):
    return KappaGenerator(None).format_monomer_site(monomer, site)


def format_reactionpattern(rp):
    return KappaGenerator(None).format_reactionpattern(rp)


def format_complexpattern(cp):
    return KappaGenerator(None).format_complexpattern(cp)


def format_monomerpattern(mp):
    return KappaGenerator(None).format_monomerpattern(mp)


def format_site_condition(site, state):
    return KappaGenerator(None).format_site_condition(site, state, '')
# End API compatibility functions


class KappaPrinter(StrPrinter):
    def __init__(self, **settings):
        super(KappaPrinter, self).__init__(settings)

    def _print_Piecewise(self, expr):
        if expr.args[-1][1] is not sympy.true:
            raise NotImplementedError('Piecewise statements are only '
                                      'supported if convertible to Kappa '
                                      'ternary statements')

        if_stmt = self._print(expr.args[-1][0])
        for pos in range(len(expr.args) - 2, -1, -1):
            if_stmt = '{} [?] {} [:] {}'.format(
                self._print(expr.args[pos][1]),
                self._print(expr.args[pos][0]),
                if_stmt
            )

        return if_stmt

    def _print_Pow(self, expr, rational=False):
        """ Adapted from sympy/printing/str.py """
        if expr.exp is S.Half and not rational:
            return "[sqrt](%s)" % self._print(expr.base)

        if expr.is_commutative:
            if -expr.exp is S.Half and not rational:
                # Note: Don't test "expr.exp == -S.Half" here, because that
                # will match -0.5, which we don't want.
                return "1/[sqrt](%s)" % self._print(expr.base)

        return super(KappaPrinter, self)._print_Pow(expr, rational)\
            .replace('**', '^')

    def _print_Pi(self, expr):
        return '[pi]'

    def _print_Exp1(self, expr):
        return '[E]'

    def _print_Function(self, expr):
        """ Adapted from sympy/printing/str.py """
        if (expr.func in (sympy.log, sympy.exp, sympy.sin, sympy.cos,
                         sympy.tan) and len(expr.args) == 1):
            func_name = '[{}]'.format(expr.func.__name__)
        elif expr.func is sympy.Mod and len(expr.args) == 2:
            return '{} [mod] {}'.format(*map(self._print, expr.args))
        else:
            func_name = expr.func.__name__
        return "{}({})".format(func_name, self.stringify(expr.args, ", "))

    def _print_Max(self, expr):
        """ Adapted from sympy/printing/cxxcode.py """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "[max] {} {}".format(
            self._print(expr.args[0]), self._print(sympy.Max(*expr.args[1:])))

    def _print_Min(self, expr):
        """ Adapted from sympy/printing/cxxcode.py """
        if len(expr.args) == 1:
            return self._print(expr.args[0])
        return "[min] {} {}".format(
            self._print(expr.args[0]), self._print(sympy.Min(*expr.args[1:])))

    def _print_Parameter(self, expr):
        return "'{}'".format(expr.name)


def expression_to_muparser(expression):
    """Render the Expression as a muparser-compatible string."""

    return KappaPrinter(order='none').doprint(expression.expr)


class KappaException(Exception):
    pass
