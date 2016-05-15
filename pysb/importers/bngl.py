from pysb.core import MonomerPattern, ComplexPattern, RuleExpression, \
    ReactionPattern, ANY, WILD
from pysb.builder import Builder
from pysb.bng import BngConsole
import xml.etree.ElementTree
import re
from sympy.parsing.sympy_parser import parse_expr


def _ns(tag_string):
    return tag_string.format('{http://www.sbml.org/sbml/level3}')


class BNGLBuilder(Builder):
    def __init__(self, filename):
        super(BNGLBuilder, self).__init__()
        with BngConsole(model=None) as con:
            con.load_bngl(filename)
            con.action('writeXML', evaluate_expressions=0)
            # import shutil
            # shutil.copy(con.base_filename+'.xml', '/Users/alex/tmp')
            self.x = xml.etree.ElementTree.parse('%s.xml' %
                             con.base_filename).getroot().find(_ns('{0}model'))
            self._model_env = {}
        self.parse_bng_xml()

    def _eval_in_model_env(self, expression):
        components = dict((c.name, c) for c in self.model.all_components())
        self._model_env.update(components)

        # Quick security check on the expression
        if not re.match(r'^[\w\s\(\)/\+\-\._\*]*$', expression):
            raise Exception('Security check on expression "%s" failed' %
                            expression)

        return eval(expression, {}, self._model_env)

    def _parse_species(self, species_xml):
        # Species may be empty for synthesis/degradation reaction patterns
        if species_xml is None:
            return []
        # Give the bonds unique numeric IDs
        bond_ids = {}
        next_bond_id = 1
        for bond in species_xml.iterfind(_ns('{0}ListOfBonds/{0}Bond')):
            for bond_attr in bond.attrib.keys():
                if bond_attr.startswith('site'):
                    bond_ids[bond.attrib[bond_attr]] = next_bond_id
            next_bond_id += 1

        # Create a list of monomer patterns
        mon_pats = []
        for mon in species_xml.iterfind(_ns('{0}ListOfMolecules/{0}Molecule')):
            mon_name = mon.get('name')
            mon_obj = self.model.monomers[mon_name]
            mon_states = {}
            for comp in mon.iterfind(_ns('{0}ListOfComponents/{0}Component')):
                state_nm = comp.get('name')
                bonds = comp.get('numberOfBonds')
                if bonds == "0":
                    mon_states[state_nm] = None
                elif bonds == "1":
                    mon_states[state_nm] = bond_ids[comp.get('id')]
                elif bonds == "?":
                    mon_states[state_nm] = WILD
                elif bonds == "+":
                    mon_states[state_nm] = ANY
                else:
                    raise Exception('Bond %s on monomer %s has '
                                    'unknown/unsupported number '
                                    'of bonds: %s ' % (state_nm,
                                                       mon_name,
                                                       bonds))
                state = comp.get('state')
                if state:
                    if mon_states[state_nm]:
                        # Site has a bond and a state
                        mon_states[state_nm] = (state, mon_states[state_nm])
                    else:
                        # Site only has a state, no bond
                        mon_states[state_nm] = state
            mon_cpt = self.model.compartments.get(mon.get('compartment'))
            mon_pats.append(MonomerPattern(mon_obj, mon_states, mon_cpt))
        return mon_pats

    def parse_monomers(self):
        for m in self.x.iterfind(_ns('{0}ListOfMoleculeTypes/'
                                     '{0}MoleculeType')):
            mon_name = m.get('id')
            sites = []
            states = {}
            for ctype in m.iterfind(_ns('{0}ListOfComponentTypes/'
                                        '{0}ComponentType')):
                c_name = ctype.get('id')
                sites.append(c_name)
                states_list = ctype.find(_ns('{}ListOfAllowedStates'))
                if states_list is not None:
                    states[c_name] = [s.get('id') for s in
                                      states_list.iterfind(_ns(
                                                           '{}AllowedState'))]
            try:
                self.monomer(mon_name, sites, states)
            except Exception as e:
                if e.message.startswith('Duplicate sites specified'):
                    raise NotImplementedError('Molecule %s has multiple '
                                              'sites with the same name. '
                                              'This is not supported in '
                                              'PySB.' % mon_name)
                else:
                    raise e

    def parse_parameters(self):
        for p in self.x.iterfind(_ns('{0}ListOfParameters/{0}Parameter')):
            p_name = p.get('id')
            if p.get('type') == 'Constant':
                p_value = p.get('value').replace('10^', '1e')
                self.parameter(name=p_name, value=p_value)
            elif p.get('type') == 'ConstantExpression':
                self.expression(name=p_name,
                                expr=self._eval_in_model_env(p.get('value')))
            else:
                raise Exception('Parameter %s has unknown type: %s' % (
                    p_name, p.get('type')))

    def parse_observables(self):
        # @TODO: Species-level compartment for observables?
        for o in self.x.iterfind(_ns('{0}ListOfObservables/{0}Observable')):
            o_name = o.get('name')
            cplx_pats = []
            for mp in o.iterfind(_ns('{0}ListOfPatterns/{0}Pattern')):
                match_once = mp.get('matchOnce')
                match_once = 1 if match_once is not None \
                    and match_once == "1" else 0
                cplx_pats.append(ComplexPattern(self._parse_species(mp),
                                                compartment=None,
                                                match_once=match_once))
            self.observable(o_name,
                            ReactionPattern(cplx_pats),
                            match=o.get('type').lower())

    def parse_initials(self):
        for i in self.x.iterfind(_ns('{0}ListOfSpecies/{0}Species')):
            value_param = i.get('concentration')
            try:
                value = float(value_param)
                # Need to create a new parameter for the initial conc. literal
                name = re.sub('[^\w]+', '_', i.get('name').replace(
                    ')', '').replace('(', '')) + '_0'
                try:
                    value_param = self.parameter(name, value)
                except ValueError as ve:
                    raise Exception(ve.message)
            except ValueError:
                # Retrieve existing parameter or (constant) expression
                try:
                    value_param = self.model.parameters[i.get('concentration')]
                except KeyError:
                    value_param = self.model.expressions[i.get(
                        'concentration')]
            mon_pats = self._parse_species(i)
            species_cpt = self.model.compartments.get(i.get('compartment'))
            self.initial(ComplexPattern(mon_pats, species_cpt),
                             value_param)

    def parse_compartments(self):
        for c in self.x.iterfind(_ns('{0}ListOfCompartments/{0}compartment')):
            cpt_size = None
            if c.get('size'):
                cpt_size = self.parameter('%s_size' % c.get('id'),
                                          c.get('size'))
            cpt_parent = None
            if c.get('outside'):
                cpt_parent = self.model.compartments[c.get('outside')]
            self.compartment(name=c.get('id'),
                             parent=cpt_parent,
                             dimension=int(c.get('spatialDimensions')),
                             size=cpt_size)

    def _parse_rate_law(self, rl):
        if rl.get('type') == 'Ele':
            rate_law_name = rl.find(_ns('{0}ListOfRateConstants/'
                                        '{0}RateConstant')).get('value')
            try:
                return self.model.parameters[rate_law_name]
            except KeyError:
                return self.model.expressions[rate_law_name]
        elif rl.get('type') == 'Function':
            return self.model.expressions[rl.get('name')]
        else:
            raise Exception('Rate law %s has unknown type %s' %
                            (rl.get('id'), rl.get('type')))

    def parse_rules(self):
        # Store reversible rates for post-processing (we don't know if we'll
        # encounter fwd or rev rule first)
        rev_rates = {}
        for r in self.x.iterfind(_ns('{0}ListOfReactionRules/'
                                     '{0}ReactionRule')):
            r_name = r.get('name')
            r_rate_xml = r.find(_ns('{}RateLaw'))
            if r_rate_xml is None:
                raise Exception('Rate law missing for rule %s' % r_name)
            r_rate = self._parse_rate_law(r_rate_xml)
            if r_name.startswith('_reverse_'):
                r_name = r_name[9:]
                rev_rates[r_name] = r_rate
                continue
            reactant_pats = []
            for rp in r.iterfind(_ns('{0}ListOfReactantPatterns/'
                                     '{0}ReactantPattern')):
                reactant_pats.append(ComplexPattern(self._parse_species(rp),
                                                    rp.get('compartment')))
            product_pats = []
            for pp in r.iterfind(_ns('{0}ListOfProductPatterns/'
                                 '{0}ProductPattern')):
                product_pats.append(ComplexPattern(self._parse_species(pp),
                                                   pp.get('compartment')))
            rule_exp = RuleExpression(ReactionPattern(reactant_pats),
                                      ReactionPattern(product_pats),
                                      is_reversible=False)
            delete_molecules = False
            for del_operations in r.iterfind(_ns('{0}ListOfOperations/'
                                                 '{0}Delete')):
                if del_operations.get('DeleteMolecules') == "1":
                    delete_molecules = True
                    break
            self.rule(r_name, rule_exp, r_rate,
                      delete_molecules=delete_molecules)

        # Set the reverse rates
        for r_name, rev_rate in rev_rates.items():
            rule = self.model.rules[r_name]
            rule.rule_expression.is_reversible = True
            rule.is_reversible = True
            rule.rate_reverse = rev_rate

    def parse_expressions(self):
        for e in self.x.iterfind(_ns('{0}ListOfFunctions/{0}Function')):
            if e.find(_ns('{0}ListOfArguments/{0}Argument')) is not None:
                raise NotImplementedError('Function %s is local, which is not '
                                          'supported in PySB' % e.get('id'))
            self.expression(e.get('id'), parse_expr(e.find(_ns(
                '{0}Expression')).text.replace('^', '**')))

    def parse_bng_xml(self):
        self.model.name = self.x.get(_ns('id'))
        self.parse_monomers()
        self.parse_parameters()
        self.parse_compartments()
        self.parse_initials()
        self.parse_observables()
        self.parse_expressions()
        self.parse_rules()
        # for r in self.model.rules:
            # print(r)


def read_bngl(filename):
    bb = BNGLBuilder(filename)
    return bb.model
