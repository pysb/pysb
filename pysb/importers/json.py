from __future__ import absolute_import
from json import JSONDecoder
from pysb.builder import Builder
from pysb.core import RuleExpression, ReactionPattern, ComplexPattern, \
    MonomerPattern, MultiState, ANY, WILD
from pysb.annotation import Annotation
import sympy
import collections
import json
import re
try:
    basestring
except NameError:
    # Python 3 compatibility.
    basestring = str


class PySBJSONDecodeError(ValueError):
    pass


class PySBJSONDecoder(JSONDecoder):
    MAX_SUPPORTED_FORMAT = 1

    def _modelget(self, name):
        if name is None:
            return None
        if name == 'model':
            return self.b.model

        return self.b.model.components[name]

    def decode_state_value(self, sv):
        if isinstance(sv, collections.Mapping) and '__object__' in sv:
            if sv['__object__'] == '__multistate__':
                return MultiState(*sv['sites'])
            if sv['__object__'] == 'ANY':
                return ANY
            if sv['__object__'] == 'WILD':
                return WILD
        try:
            if len(sv) == 2 and isinstance(sv[0], basestring) \
                    and isinstance(sv[1], (int, collections.Mapping)):
                return sv[0], self.decode_state_value(sv[1])
        except TypeError:
            pass
        return sv

    def decode_monomer(self, mon):
        self.b.monomer(mon['name'], mon['sites'], mon['states'])

    def decode_parameter(self, par):
        self.b.parameter(par['name'], par['value'])

    def decode_expression(self, expr):
        e = expr['expr']
        # Quick security check on the expression
        if not re.match(r'^[\w\s()/+\-._*]*$', e):
            raise PySBJSONDecodeError(
                'Security check on expression "%s" failed' % expr['name']
            )

        self.b.expression(
            expr['name'],
            sympy.sympify(e, locals={
                c.name: c for c in self.b.model.components
            }, evaluate=False)
        )

    def decode_observable(self, obs):
        self.b.observable(
            obs['name'],
            self.decode_reaction_pattern(obs['reaction_pattern']),
            obs['match']
        )

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

    def decode(self, s):
        res = super(PySBJSONDecoder, self).decode(s)

        if not isinstance(res, dict):
            raise PySBJSONDecodeError('Decode error (not dictionary)')
        if 'format' not in res:
            raise PySBJSONDecodeError(
                'No "format" entry found - is this a PySB model?')
        if not isinstance(res['format'], int):
            raise PySBJSONDecodeError('"format" attribute is not an integer')

        if res['format'] < 1:
            raise PySBJSONDecodeError('"Invalid format value: {}'.format(
                res['format']))

        if res['format'] > self.MAX_SUPPORTED_FORMAT:
            raise PySBJSONDecodeError(
                'Format {} is not supported (max: {})'.format(
                    res['format'], self.MAX_SUPPORTED_FORMAT
                )
            )

        self.b = Builder()
        self.b.model.name = res['name']

        decoders = collections.OrderedDict((
            ('monomers', self.decode_monomer),
            ('parameters', self.decode_parameter),
            ('expressions', self.decode_expression),
            ('compartments', self.decode_compartment),
            ('tags', self.decode_tag),
            ('rules', self.decode_rule),
            ('observables', self.decode_observable),
            ('initials', self.decode_initial),
            ('annotations', self.decode_annotation),
        ))

        for component_type, decoder in decoders.items():
            for component in res[component_type]:
                decoder(component)

        return self.b.model


def model_from_json(json_str):
    return json.loads(json_str, cls=PySBJSONDecoder)
