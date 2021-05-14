from pysb.core import MonomerPattern, ComplexPattern, RuleExpression, \
    ReactionPattern, ANY, WILD, MultiState
from pysb.builder import Builder
from pysb.bng import BngFileInterface, parse_bngl_expr
import xml.etree.ElementTree
import re
import sympy
import warnings
import pysb.logging
import collections
import numbers
import os


def _ns(tag_string):
    """
    Shorthand to apply XML namespace
    """
    return tag_string.format('{http://www.sbml.org/sbml/level3}')


class BnglImportError(Exception):
    pass


class BnglBuilder(Builder):
    """
    Assemble a Model from a .bngl file.

    See :py:func:`model_from_bngl` for further details.
    """
    def __init__(self, filename, force=False, cleanup=True):
        super(BnglBuilder, self).__init__()
        filename = os.path.abspath(filename)
        with BngFileInterface(model=None, cleanup=cleanup) as con:
            con.action('readFile', file=filename, skip_actions=1)
            con.action('writeXML', evaluate_expressions=0)
            con.execute()
            self._x = xml.etree.ElementTree.parse('%s.xml' %
                                                  con.base_filename)\
                                           .getroot().find(_ns('{0}model'))

        self.model.name = os.path.splitext(os.path.basename(filename))[0]
        self._force = force
        self._model_env = {}
        self._renamed_states = collections.defaultdict(dict)
        self._renamed_observables = {}
        self._log = pysb.logging.get_logger(__name__)
        self._parse_bng_xml()

    def _warn_or_except(self, msg):
        """
        Raises a warning or Exception, depending of the value of self._force
        """
        if self._force:
            warnings.warn(msg)
        else:
            raise BnglImportError(msg)

    def _eval_in_model_env(self, expression):
        """
        Evaluates an expression string in the model environment
        """
        components = dict((c.name, c) for c in self.model.all_components())
        self._model_env.update(components)

        # Quick security check on the expression
        if re.match(r'^[\w\s()/+\-._*^]*$', expression):
            return parse_bngl_expr(expression, local_dict=self._model_env)
        else:
            self._warn_or_except('Security check on expression "%s" failed' %
                                 expression)
            return None

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
                    bond_ids.setdefault(bond.attrib[bond_attr], [])\
                                        .append(next_bond_id)
            next_bond_id += 1

        # Create a list of monomer patterns
        mon_pats = []
        for mon in species_xml.iterfind(_ns('{0}ListOfMolecules/{0}Molecule')):
            mon_name = mon.get('name')
            mon_obj = self.model.monomers[mon_name]
            mon_states = collections.defaultdict(list)
            for comp in mon.iterfind(_ns('{0}ListOfComponents/{0}Component')):
                # BioNetGen component (state) labels are not supported yet
                if 'label' in comp.attrib.keys():
                    self._warn_or_except('BioNetGen component/state labels '
                                         'are not yet supported in PySB')

                state_nm = comp.get('name')
                bonds = comp.get('numberOfBonds')
                if bonds == "0":
                    last_bond = None
                elif bonds == "?":
                    last_bond = WILD
                elif bonds == "+":
                    last_bond = ANY
                else:
                    bond_list = bond_ids[comp.get('id')]
                    assert int(bonds) == len(bond_list)
                    if len(bond_list) == 1:
                        last_bond = bond_list[0]
                    else:
                        last_bond = bond_list
                state = comp.get('state')
                if state and state != '?':
                    if state in ('PLUS', 'MINUS'):
                        self._warn_or_except(
                            'PLUS/MINUS state values are non-standard BNGL '
                            'used to increment or decrement a numeric state '
                            'value. They are not supported in PySB.'
                        )
                    # If we changed the state string, use the updated version
                    state = self._renamed_states.get(mon_name, {}).get(
                        state, state)
                    if last_bond:
                        # Site has a bond and a state
                        mon_states[state_nm].append((state, last_bond))
                    else:
                        # Site only has a state, no bond
                        mon_states[state_nm].append(state)
                else:
                    mon_states[state_nm].append(last_bond)

            mon_states = {k: MultiState(*v) if len(v) > 1 else v[0]
                          for k, v in mon_states.items()}

            mon_cpt = self.model.compartments.get(mon.get('compartment'))
            mon_pats.append(MonomerPattern(mon_obj, mon_states, mon_cpt))
            if 'label' in mon.attrib.keys():
                try:
                    tag = self.model.tags[mon.get('label')]
                except KeyError:
                    tag = self.tag(mon.get('label'))
                mon_pats[-1]._tag = tag
        return mon_pats

    def _parse_monomers(self):
        for m in self._x.iterfind(_ns('{0}ListOfMoleculeTypes/'
                                      '{0}MoleculeType')):
            mon_name = m.get('id')
            sites = []
            states = collections.defaultdict(dict)
            for ctype in m.iterfind(_ns('{0}ListOfComponentTypes/'
                                        '{0}ComponentType')):
                c_name = ctype.get('id')
                sites.append(c_name)
                states_list = ctype.find(_ns('{}ListOfAllowedStates'))
                if states_list is not None:
                    for s in states_list.iterfind(_ns('{}AllowedState')):
                        state = s.get('id')
                        if re.match('[0-9]*$', state):
                            if state not in states[c_name]:
                                new_state = '_' + state
                                while new_state in states[c_name].values():
                                    new_state = '_' + new_state
                                states[c_name][state] = new_state
                                self._renamed_states[mon_name][state] = \
                                    new_state
                        else:
                            states[c_name][state] = state
                    if self._renamed_states[mon_name]:
                        self._log.info('Monomer "{}" states were renamed as '
                                       'follows: {}'.format(
                            mon_name,
                            self._renamed_states[mon_name])
                        )
            try:
                self.monomer(mon_name, sites,
                             {c_name: list(statedict.values())
                              for c_name, statedict in states.items()})
            except Exception as e:
                if str(e).startswith('Duplicate sites specified'):
                    self._warn_or_except('Molecule %s has multiple '
                                         'sites with the same name. '
                                         'This is not supported in PySB.' %
                                         mon_name)
                else:
                    raise BnglImportError(str(e))

    def _parse_parameters(self):
        for p in self._x.iterfind(_ns('{0}ListOfParameters/{0}Parameter')):
            p_name = p.get('id')
            if p.get('type') == 'Constant':
                p_value = p.get('value').replace('10^', '1e')
                try:
                    self.parameter(name=p_name, value=p_value)
                except ValueError:
                    # Despite the "Constant" label, some constant expressions
                    # appear here e.g. ln(2)/120 in BNG's Repressilator model
                    self.expression(name=p_name,
                                    expr=self._eval_in_model_env(
                                        p.get('value')))
            elif p.get('type') == 'ConstantExpression':
                self.expression(name=p_name,
                                expr=self._eval_in_model_env(p.get('value')))
            else:
                self._warn_or_except('Parameter %s has unknown type: %s' %
                                     (p_name, p.get('type')))

    def _parse_observables(self):
        for o in self._x.iterfind(_ns('{0}ListOfObservables/{0}Observable')):
            o_name = o.get('name')
            match_mode = o.get('type').lower()
            # Some BNG observables have same name as a monomer, but in PySB
            # these must be unique
            if o_name in self.model.monomers.keys():
                o_name_old = o_name
                o_name = 'Obs_{}'.format(o_name)
                self._renamed_observables[o_name_old] = o_name
            cplx_pats = []
            for mp in o.iterfind(_ns('{0}ListOfPatterns/{0}Pattern')):
                cpt = self.model.compartments.get(mp.get('compartment'))
                match_once = mp.get('matchOnce')
                match_once = True if match_once == "1" and \
                    match_mode != 'species' else False
                cplx_pats.append(ComplexPattern(self._parse_species(mp),
                                                compartment=cpt,
                                                match_once=match_once))
            self.observable(o_name,
                            ReactionPattern(cplx_pats),
                            match=match_mode)

    def _parse_initials(self):
        for i in self._x.iterfind(_ns('{0}ListOfSpecies/{0}Species')):
            value_param = i.get('concentration')
            try:
                value = float(value_param)
                # Need to create a new parameter for the initial conc. literal
                name = re.sub('[^\w]+', '_', i.get('name').replace(
                    ')', '').replace('(', '')) + '_0'
                try:
                    value_param = self.parameter(name, value)
                except ValueError as ve:
                    raise BnglImportError(str(ve))
            except ValueError:
                # Retrieve existing parameter or (constant) expression
                try:
                    value_param = self.model.parameters[i.get('concentration')]
                except KeyError:
                    value_param = self.model.expressions[i.get(
                        'concentration')]
            mon_pats = self._parse_species(i)
            species_cpt = self.model.compartments.get(i.get('compartment'))
            cp = ComplexPattern(mon_pats, species_cpt)
            fixed = i.get('Fixed') == "1"
            self.initial(cp, value_param, fixed)

    def _parse_compartments(self):
        for c in self._x.iterfind(_ns('{0}ListOfCompartments/{0}compartment')):
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
            try:
                return self.model.expressions[rl.get('name')]
            except KeyError:
                return self.model.parameters[rl.get('name')]
        else:
            self._warn_or_except('Rate law %s has unknown type %s' %
                                 (rl.get('id'), rl.get('type')))
            return None

    def _parse_rules(self):
        # Store reversible rates for post-processing (we don't know if we'll
        # encounter fwd or rev rule first)
        rev_rates = {}
        for r in self._x.iterfind(_ns('{0}ListOfReactionRules/'
                                      '{0}ReactionRule')):
            r_name = r.get('name')

            r_rate_xml = r.find(_ns('{}RateLaw'))
            if r_rate_xml is None:
                raise BnglImportError('Rate law missing for rule %s' % r_name)
            r_rate = self._parse_rate_law(r_rate_xml)

            # Store reverse rates for post-processing
            if r_name.startswith('_reverse_'):
                r_name = r_name[9:]
                rev_rates[r_name] = r_rate
                continue

            # Compile reactant and product patterns
            reactant_pats = []
            for rp in r.iterfind(_ns('{0}ListOfReactantPatterns/'
                                     '{0}ReactantPattern')):
                cpt = self.model.compartments.get(rp.get('compartment'))
                reactant_pats.append(ComplexPattern(self._parse_species(rp),
                                                    cpt))
                if 'label' in rp.attrib:
                    reactant_pats[-1]._tag = self.model.tags[rp.get('label')]
            product_pats = []
            for pp in r.iterfind(_ns('{0}ListOfProductPatterns/'
                                     '{0}ProductPattern')):
                cpt = self.model.compartments.get(pp.get('compartment'))
                product_pats.append(ComplexPattern(self._parse_species(pp),
                                                   cpt))

                if 'label' in pp.attrib:
                    product_pats[-1]._tag = self.model.tags[pp.get('label')]
            rule_exp = RuleExpression(ReactionPattern(reactant_pats),
                                      ReactionPattern(product_pats),
                                      is_reversible=False)

            # Process any DeleteMolecules declaration
            delete_molecules = False
            for del_operations in r.iterfind(_ns('{0}ListOfOperations/'
                                                 '{0}Delete')):
                if del_operations.get('DeleteMolecules') == "1":
                    delete_molecules = True
                    break

            # Process any MoveConnected declaration
            move_connected = False
            for change_cpt_ops in r.iterfind(_ns('{0}ListOfOperations/'
                                                 '{0}ChangeCompartment')):
                if change_cpt_ops.get('moveConnected') == "1":
                    move_connected = True
                    break

            # Give warning/error if ListOfExcludeReactants or
            # ListOfExcludeProducts is present
            if r.find(_ns('{}ListOfExcludeReactants')) is not None or \
               r.find(_ns('{}ListOfExcludeProducts')) is not None:
                self._warn_or_except('ListOfExcludeReactants and/or '
                                     'ListOfExcludeProducts declarations '
                                     'are deprecated in BNG, and not supported '
                                     'in PySB.')

            # Give warning/error if ListOfIncludeReactants or
            # ListOfIncludeProducts is present
            if r.find(_ns('{}ListOfIncludeReactants')) is not None or \
               r.find(_ns('{}ListOfIncludeProducts')) is not None:
                self._warn_or_except('ListOfIncludeReactants and/or '
                                     'ListOfIncludeProducts declarations '
                                     'are deprecated in BNG, and not supported '
                                     'in PySB.')

            self.rule(r_name, rule_exp, r_rate,
                      delete_molecules=delete_molecules,
                      move_connected=move_connected)

        # Set the reverse rates
        for r_name, rev_rate in rev_rates.items():
            rule = self.model.rules[r_name]
            rule.rule_expression.is_reversible = True
            rule.is_reversible = True
            rule.rate_reverse = rev_rate

    def _parse_expressions(self):
        expr_namespace = (self.model.parameters | self.model.expressions)
        expr_symbols = {e.name: e for e in expr_namespace}

        for e in self._x.iterfind(_ns('{0}ListOfFunctions/{0}Function')):
            for arg in e.iterfind(_ns('{0}ListOfArguments/{0}Argument')):
                tag_name = arg.get('id')
                try:
                    self.model.tags[tag_name]
                except KeyError:
                    tag = self.tag(tag_name)
                    expr_symbols[tag_name] = tag
            expr_name = e.get('id')
            expr_text = e.find(_ns('{0}Expression')).text
            expr_val = 0
            try:
                expr_val = parse_bngl_expr(expr_text, local_dict=expr_symbols)
            except Exception as ex:
                self._warn_or_except('Could not parse expression %s: '
                                     '%s\n\nError: %s' % (expr_name,
                                                          expr_text,
                                                          str(ex)))
            # Replace observables now, so they get expanded by .expand_expr()
            # Doing this as part of expr_symbols breaks local functions!
            observables = {
                sympy.Symbol(o.name): o for o in self.model.observables
            }
            # Add renamed observables
            observables.update({
                sympy.Symbol(obs_old): self.model.observables[obs_new]
                for obs_old, obs_new in self._renamed_observables.items()
            })
            expr_val = expr_val.xreplace(observables)

            if isinstance(expr_val, numbers.Number):
                self.parameter(expr_name, expr_val)
            else:
                self.expression(expr_name, expr_val)

    def _parse_bng_xml(self):
        self._parse_monomers()
        self._parse_parameters()
        self._parse_compartments()
        self._parse_initials()
        self._parse_observables()
        self._parse_expressions()
        self._parse_rules()


def model_from_bngl(filename, force=False, cleanup=True):
    """
    Convert a BioNetGen .bngl model file into a PySB Model.

    Notes
    -----

    The following features are not supported in PySB and will cause an error
    if present in a .bngl file:

    * Fixed species (with a ``$`` prefix, like ``$Null``)
    * BNG excluded or included reaction patterns (deprecated in BNG)
    * BNG local functions
    * Molecules with identically named sites, such as ``M(l,l)``
    * BNG's custom rate law functions, such as ``MM`` and ``Sat``
      (deprecated in BNG)

    Parameters
    ----------
    filename : string
        A BioNetGen .bngl file
    force : bool, optional
        The default, False, will raise an Exception if there are any errors
        importing the model to PySB, e.g. due to unsupported features.
        Setting to True will attempt to ignore any import errors, which may
        lead to a model that only poorly represents the original. Use at own
        risk!
    cleanup : bool
        Delete temporary directory on completion if True. Set to False for
        debugging purposes.
    """
    bb = BnglBuilder(filename, force=force, cleanup=cleanup)
    return bb.model
