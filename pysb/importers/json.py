from json import JSONDecoder
from pysb.builder import Builder
from pysb.core import RuleExpression, ReactionPattern, ComplexPattern, \
    MonomerPattern, MultiState, ANY, WILD, Parameter, Expression
from pysb.annotation import Annotation
from pysb.pattern import SpeciesPatternMatcher
import sympy
import collections
from collections.abc import Mapping
import json
import re
import warnings
from sympy.parsing.sympy_parser import parse_expr


class PySBJSONDecodeError(ValueError):
    pass


class PySBJSONDecoder(JSONDecoder):
    """
    Decode a JSON-encoded PySB model

    See :py:mod:`pysb.export.json` for implementation details.
    """
    MAX_SUPPORTED_PROTOCOL = 1

    def _modelget(self, name):
        if name is None:
            return None
        if name == 'model':
            return self.b.model

        return self.b.model.components[name]

    def decode_state_value(self, sv):
        if isinstance(sv, Mapping) and '__object__' in sv:
            if sv['__object__'] == '__multistate__':
                return MultiState(*sv['sites'])
            if sv['__object__'] == 'ANY':
                return ANY
            if sv['__object__'] == 'WILD':
                return WILD
        try:
            if len(sv) == 2 and isinstance(sv[0], str) \
                    and isinstance(sv[1], (int, Mapping)):
                return sv[0], self.decode_state_value(sv[1])
        except TypeError:
            pass
        return sv

    def decode_monomer(self, mon):
        self.b.monomer(mon['name'], mon['sites'], mon['states'])

    def decode_parameter(self, par):
        self.b.parameter(par['name'], par['value'])

    def decode_derived_parameter(self, par):
        self.b.model._derived_parameters.add(
            Parameter(par['name'], par['value'], _export=False)
        )

    def _parse_expr(self, e, e_name):
        # Quick security check on the expression
        if not re.match(r'^[\w\s()/+\-._*]*$', e):
            raise PySBJSONDecodeError(
                'Security check on %s failed' % e_name
            )

        expr_symbols = {
            s.name: s for s in
            (self.b.model.parameters | self.b.model.expressions |
             self.b.model.tags | self.b.model._derived_parameters |
             self.b.model._derived_expressions)
        }

        expression = parse_expr(e, local_dict=expr_symbols)

        # Replace observables now, to enable expand_expr()
        # Replacing as part of expr_symbols breaks local functions!
        expression = expression.xreplace(
            {sympy.Symbol(s.name): s for s in self.b.model.observables})

        return expression

    def decode_expression(self, expr, derived=False):
        expression = self._parse_expr(expr['expr'],
                                      'expression "{}"'.format(expr['name']))

        if derived:
            self.b.model._derived_expressions.add(
                Expression(expr['name'], expression, _export=False)
            )
        else:
            self.b.expression(
                str(expr['name']),  # Need str() to avoid unicode errors on Py2
                expression
            )

    def decode_derived_expression(self, expr):
        return self.decode_expression(expr, derived=True)

    def decode_observable(self, obs):
        o = self.b.observable(
            obs['name'],
            self.decode_reaction_pattern(obs['reaction_pattern']),
            obs['match']
        )
        try:
            o.coefficients = obs['coefficients']
            o.species = obs['species']
        except KeyError:
            pass

    def decode_monomer_pattern(self, mp):
        mon = self._modelget(mp['monomer'])
        mp_obj = MonomerPattern(
            mon,
            {site: self.decode_state_value(sv)
             for site, sv in mp['site_conditions'].items()},
            self._modelget(mp['compartment'])
        )
        mp_obj._tag = self._modelget(mp['tag'])
        return mp_obj

    def decode_complex_pattern(self, cp):
        cp_obj = ComplexPattern(
            [self.decode_monomer_pattern(mp) for mp in cp['monomer_patterns']],
            self._modelget(cp['compartment']),
            cp['match_once']
        )
        cp_obj._tag = self._modelget(cp['tag'])
        return cp_obj

    def decode_species(self, sp):
        self.b.model.species.append(self.decode_complex_pattern(sp))

    def decode_reaction_pattern(self, rp):
        return ReactionPattern(
            [self.decode_complex_pattern(cp) for cp in rp['complex_patterns']]
        )

    def decode_rule_expression(self, rexp):
        return RuleExpression(
            self.decode_reaction_pattern(rexp['reactant_pattern']),
            self.decode_reaction_pattern(rexp['product_pattern']),
            rexp['reversible']
        )

    def decode_rule(self, r):
        self.b.rule(
            r['name'],
            self.decode_rule_expression(r['rule_expression']),
            self._modelget(r['rate_forward']),
            self._modelget(r['rate_reverse']),
            r['delete_molecules'],
            r['move_connected']
        )

    def decode_tag(self, tag):
        self.b.tag(tag['name'])

    def decode_compartment(self, cpt):
        self.b.compartment(
            cpt['name'],
            self._modelget(cpt['parent']),
            cpt['dimension'],
            self._modelget(cpt['size'])
        )

    def decode_initial(self, init):
        self.b.initial(
            self.decode_complex_pattern(init['pattern']),
            self._modelget(init['parameter_or_expression']),
            init['fixed']
        )

    def decode_annotation(self, ann):
        self.b.model.add_annotation(
            Annotation(
                self._modelget(ann['subject']),
                ann['object'],
                ann['predicate'],
                _export=False
            )
        )

    def decode_reaction(self, rxn):
        rxn['rate'] = self._parse_expr(rxn['rate'], 'reaction rate')
        self.b.model.reactions.append(rxn)

    def decode(self, s):
        res = super(PySBJSONDecoder, self).decode(s)

        if not isinstance(res, dict):
            raise PySBJSONDecodeError('Decode error (not dictionary)')
        if 'protocol' not in res:
            raise PySBJSONDecodeError(
                'No "protocol" entry found - is this a PySB model?')
        if not isinstance(res['protocol'], int):
            raise PySBJSONDecodeError('"protocol" attribute is not an integer')

        if res['protocol'] < 1:
            raise PySBJSONDecodeError('"Invalid "protocol" value: {}'.format(
                res['protocol']))

        if res['protocol'] > self.MAX_SUPPORTED_PROTOCOL:
            raise PySBJSONDecodeError(
                'Protocol {} is not supported (max: {})'.format(
                    res['protocol'], self.MAX_SUPPORTED_PROTOCOL
                )
            )

        self.b = Builder()
        self.b.model.name = res['name']

        decoders = collections.OrderedDict((
            ('monomers', self.decode_monomer),
            ('parameters', self.decode_parameter),
            ('_derived_parameters', self.decode_derived_parameter),
            ('compartments', self.decode_compartment),
            ('observables', self.decode_observable),
            ('tags', self.decode_tag),
            ('expressions', self.decode_expression),
            ('_derived_expressions', self.decode_derived_expression),
            ('rules', self.decode_rule),
            ('initials', self.decode_initial),
            ('annotations', self.decode_annotation),
            ('reactions', self.decode_reaction),
            ('reactions_bidirectional', self.decode_reaction),
            ('species', self.decode_species)
        ))

        for component_type, decoder in decoders.items():
            for component in res.get(component_type, []):
                decoder(component)

        if self.b.model.reactions and self.b.model.observables \
                and 'species' not in res['observables'][0]:

            # We have network, need to regenerate Observable species and coeffs
            warnings.warn(
                'This SimulationResult file is missing Observable species and '
                'coefficients data. These will be generated now - we recommend '
                'you re-save your SimulationResult file to avoid this warning.'
            )

            for obs in self.b.model.observables:
                if obs.match in ('molecules', 'species'):
                    obs_matches = SpeciesPatternMatcher(self.b.model).match(
                        obs.reaction_pattern, index=True, counts=True)
                    sp, vals = zip(*sorted(obs_matches.items()))
                    obs.species = list(sp)
                    if obs.match == 'molecules':
                        obs.coefficients = list(vals)
                    else:
                        obs.coefficients = [1] * len(obs_matches.values())
                else:
                    raise ValueError(f'Unknown obs.match value: {obs.match}')

        return self.b.model


def model_from_json(json_str):
    return json.loads(json_str, cls=PySBJSONDecoder)
