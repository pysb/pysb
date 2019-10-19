"""
Module containing a class for exporting a PySB model to JSON

For information on how to use the model exporters, see the documentation
for :py:mod:`pysb.export`.
"""

from __future__ import absolute_import
from pysb.export import Exporter
import json
from pysb.core import Model, MultiState, KeywordMeta


class JsonExporter(Exporter):
    """A class for returning the JSON for a given PySB model.

    Inherits from :py:class:`pysb.export.Exporter`, which implements
    basic functionality for all exporters.
    """

    def export(self):
        """Generate the corresponding JSON for the PySB model associated
        with the exporter.

        Returns
        -------
        string
            The JSON output for the model.
        """
        return json.dumps(self.model, cls=PySBJSONEncoder)


class PySBJSONEncoder(json.JSONEncoder):
    FORMAT = 1

    @classmethod
    def encode_keyword(cls, keyword):
        return {'__object__': str(keyword)}

    @classmethod
    def encode_multistate(cls, stateval):
        return {
            '__object__': '__multistate__',
            'sites': stateval.sites
        }

    @classmethod
    def encode_monomer(cls, mon):
        return {
            'name': mon.name,
            'sites': mon.sites,
            'states': mon.site_states
        }

    @classmethod
    def encode_compartment(cls, cpt):
        return {
            'name': cpt.name,
            'parent': cpt.parent.name if cpt.parent else None,
            'dimension': cpt.dimension,
            'size': cpt.size.name if cpt.size else None
        }

    @classmethod
    def encode_parameter(cls, par):
        return {
            'name': par.name,
            'value': par.value
        }

    @classmethod
    def encode_expression(cls, expr):
        return {
            'name': expr.name,
            'expr': str(expr.expr)
        }

    @classmethod
    def encode_monomer_pattern(cls, mp):
        return {
            'monomer': mp.monomer.name,
            'site_conditions': mp.site_conditions,
            'compartment': mp.compartment.name if mp.compartment else None,
            'tag': mp._tag.name if mp._tag else None
        }

    @classmethod
    def encode_complex_pattern(cls, cp):
        return {
            'monomer_patterns': [cls.encode_monomer_pattern(mp)
                                 for mp in cp.monomer_patterns],
            'compartment': cp.name if cp.compartment else None,
            'match_once': cp.match_once,
            'tag': cp._tag.name if cp._tag else None
        }

    @classmethod
    def encode_reaction_pattern(cls, rp):
        return {
            'complex_patterns': [cls.encode_complex_pattern(cp)
                                 for cp in rp.complex_patterns]
        }

    @classmethod
    def encode_rule_expression(cls, rexp):
        return {
            'reactant_pattern': cls.encode_reaction_pattern(
                rexp.reactant_pattern),
            'product_pattern': cls.encode_reaction_pattern(
                rexp.product_pattern),
            'reversible': rexp.is_reversible
        }

    @classmethod
    def encode_rule(cls, r):
        return {
            'name': r.name,
            'rule_expression': cls.encode_rule_expression(r.rule_expression),
            'rate_forward': r.rate_forward.name,
            'rate_reverse': r.rate_reverse.name if r.rate_reverse else None,
            'delete_molecules': r.delete_molecules,
            'move_connected': r.move_connected
        }

    @classmethod
    def encode_observable(cls, obs):
        return {
            'name': obs.name,
            'reaction_pattern': cls.encode_reaction_pattern(
                obs.reaction_pattern),
            'match': obs.match
        }

    @classmethod
    def encode_initial(cls, init):
        return {
            'pattern': cls.encode_complex_pattern(init.pattern),
            'parameter_or_expression': init.value.name,
            'fixed': init.fixed
        }

    @classmethod
    def encode_annotation(cls, ann):
        return {
            'subject': 'model' if isinstance(ann.subject, Model)
            else ann.subject.name,
            'object': str(ann.object),
            'predicate': str(ann.predicate)
        }

    @classmethod
    def encode_tag(cls, tag):
        return {
            'name': tag.name
        }

    @classmethod
    def encode_model(cls, model):
        d = dict(format=cls.FORMAT, name=model.name)

        encoders = {
            'monomers': cls.encode_monomer,
            'compartments': cls.encode_compartment,
            'tags': cls.encode_tag,
            'parameters': cls.encode_parameter,
            'expressions': cls.encode_expression,
            'rules': cls.encode_rule,
            'observables': cls.encode_observable,
            'initials': cls.encode_initial,
            'annotations': cls.encode_annotation
        }

        for component_type, encoder in encoders.items():
            d[component_type] = [encoder(component)
                                 for component in
                                 getattr(model, component_type)]

        return d

    def default(self, o):
        if isinstance(o, Model):
            return self.encode_model(o)
        elif isinstance(o, MultiState):
            return self.encode_multistate(o)
        elif isinstance(o, KeywordMeta):
            return self.encode_keyword(o)

        return super(PySBJSONEncoder, self).default(o)
