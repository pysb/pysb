import networkx as nx
import numpy as np
import sympy as sp

import itertools
import math

from .core import (
    ReactionPattern, Monomer, NO_BOND, Compartment, Expression, Tag, autoinc
)
from .pattern import match_complex_pattern, SpeciesPatternMatcher
from networkx.algorithms.isomorphism.vf2userfunc import GraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match
from collections import ChainMap, Counter


class ReactionGenerator:
    """
    Provides methods to generate a reaction from a set of reactants based on
    the :class:`Rule` it was initialized with.

    Parameters
    ----------
    rule: pysb.Rule
        Rule for which this instance should generate reactions
    reverse: bool
        Indicates whether forward or reverse reactions should be generated
    model: pysb.Model
        Model containing the rule
    """
    def __init__(self, rule, reverse, model):
        self.name = rule.name
        self.reverse = reverse
        self.reactant_pattern = rule.product_pattern if reverse else \
            rule.reactant_pattern
        self.product_pattern = rule.reactant_pattern if reverse else \
            rule.product_pattern
        self.rate = rule.rate_reverse if reverse else rule.rate_forward

        self.is_pure_synthesis_rule = \
            len(rule.reactant_pattern.complex_patterns) == 0

        self.delete_molecules = rule.delete_molecules

        self.spm = SpeciesPatternMatcher(model)

        # precompute generic variables
        self.graph_diff = GraphDiffGenerator(self, rule.move_connected)

        self._needs_rate_localization = isinstance(self.rate, Expression) \
            and isinstance(self.rate.expr.func,
                           sp.function.UndefinedFunction) \
            and any(isinstance(a, Tag) for a in self.rate.expr.atoms())

        if self._needs_rate_localization:
            rate = self.rate
            tags = tuple(a for a in rate.expr.atoms() if isinstance(a, Tag))
            self._tag_rp_ids = tuple(
                next(
                    icp for icp, cp in enumerate(
                        self.reactant_pattern.complex_patterns
                    )
                    if cp._tag is not None and cp._tag.name == tag.name
                )
                for tag in tags
            )
            # creates a function that evaluates observable functions of tags
            # according to the matching count of the respectively provided
            # ComplexPattern
            self._localfunc = sp.lambdify(
                tags,
                model.expressions[rate.expr.func.name].expr,
                modules=[
                    {par.name: par for par in model.parameters},
                    {expr.name: expr for expr in model.expressions},
                    {
                        obs.name: lambda x: sum(match_complex_pattern(
                            cp, x, count=True
                        ) for cp in obs.reaction_pattern.complex_patterns)
                        for obs in model.observables
                    }
                ]
            )
        else:
            self._tags = tuple()
            self._tag_rp_ids = tuple()
            self._localfunc = None

    def generate_reactions(self, reactant_idx, model):
        """
        Generates reactions for a set of reactant indices. These indices may be
        unordered and reactions will be computed for all possible matchings.

        Parameters
        ----------
        reactant_idx: Sequence[int]
            Species indices of the reactants
        model:
            Model containing the ordered species
        """
        matches = get_matching_patterns(self.reactant_pattern,
                                        [model.species[ix]
                                         for ix in reactant_idx])

        # we only generate one reaction for every unique set of educts. the
        # remaining symmetry is accounted for by the statfactor in the reaction
        # rate
        educt_matches = [
            np.where(matches[irp, :])[0]
            for irp in range(matches.shape[0])
        ]
        cp_indices = {
            tuple(np.unique(indices)): indices
            for indices in itertools.product(*educt_matches)
            if len(np.unique(indices)) == len(reactant_idx)
        }.values()

        return [
            rxn
            for cp_index in cp_indices
            for rxn in self._generate_reactions_mapped_cps(
                tuple(reactant_idx[cp_idx] for cp_idx in cp_index), model
            )
        ]

    def is_species_transport(self):
        """
        Check whether this generator implements a species transport.
        """
        return are_unequal_compartments(*self.graph_diff.compartment_transport)

    def _generate_reactions_mapped_cps(self, reactant_idx, model):
        """
        Generates reactions for a sequence of reactant indices. These indices 
        must be ordered and reactions will only be computed for the 
        restricted matchings.

        Parameters
        ----------
        reactant_idx: Sequence[int]
            Species indices of the reactants
        model:
            Model containing the ordered species
        """
        reactants = [model.species[ix] for ix in reactant_idx]
        reactant_mappings, mp_alignment_cp = \
            self._compute_reactant_mappings(reactants)

        reactant_graph = ReactionPattern(reactants)._as_graph(
            mp_alignment_cp
        )

        graph_diffs = set(
            self.graph_diff.generate(mapping)
            for mapping in reactant_mappings
        )

        reactions = [
            self._generate_reaction(reactant_idx, graph_diff,
                                    reactant_graph, model)
            for graph_diff in graph_diffs
        ]
        # summarize reactions with duplicate products by adding rates
        rxns = dict()
        for rxn in reactions:
            normalized_products = tuple(sorted(rxn['products']))
            if normalized_products in rxns:
                rxns[normalized_products]['rate'] += rxn['rate']
            else:
                rxns[normalized_products] = rxn

        return rxns.values()

    def _generate_reaction(self, reactant_idx, graph_diff,
                           reactant_graph, model):
        """
        Generates a single reaction for a sequence of reactant indices. These
        indices must be ordered and reactions will only be computed for the
        restricted matchings.

        Parameters
        ----------
        reactant_idx: Sequence[int]
            Species indices of the reactants
        model:
            Model containing the ordered species
        """
        reactants = [model.species[ix] for ix in reactant_idx]

        product_graph = graph_diff.apply(reactant_graph, self.delete_molecules)

        products_pattern = ReactionPattern._from_graph(product_graph)
        if products_pattern is not None:
            products = products_pattern.complex_patterns
        else:
            products = [None]

        self.update_compartments(products, model)

        # accounts for molecule/species transport
        vfactor = _compute_volume_factor(reactants, products)

        rate = self._localize_rate(reactants)

        product_indices = tuple(self.spm.match(product, index=True,
                                               exact=True)[0]
                                for product in products
                                if product is not None)

        # accounts for symmetries in educts
        sfactor = _compute_stat_factor(reactant_idx)

        reaction = {
            'rule': self.name,
            'rate': rate * vfactor * sfactor * np.prod([
                sp.Symbol(f'__s{ix}') for ix in reactant_idx
            ]),
            'reactants': reactant_idx,
            'products': product_indices,
            'base_rate': rate,
            'volume_factor': vfactor,
            'stat_factor': sfactor,
            'product_patterns': products,
            'reactant_patterns': reactants,
        }
        return reaction

    def update_compartments(self, products, model):
        """
        Updates compartments to fix child/parent compartments for species
        transports that are not accounted for by the graph difference.

        Parameters
        ----------
        products: Sequence[pysb.ComplexPattern]
            Product species
        model:
            Model containing compartment definitions
        """
        if not self.is_species_transport():
            return

        reactant_cpt, product_cpt = self.graph_diff.compartment_transport

        cpt_updates = {
            reactant_cpt.name: product_cpt
        }
        if reactant_cpt.dimension == 2 and product_cpt.dimension == 2:
            # when moving from surface compartment to surface compartment
            # inside/outside are flipped
            inside_cpt = next((
                cpt for cpt in model.compartments
                if are_equal_compartments(cpt.parent, product_cpt)
            ), None)
            outside_cpt = reactant_cpt.parent
            if are_unequal_compartments(inside_cpt, outside_cpt):
                cpt_updates[outside_cpt.name] = inside_cpt

            inside_cpt = next((
                cpt for cpt in model.compartments
                if are_equal_compartments(cpt.parent, reactant_cpt)
            ), None)
            outside_cpt = product_cpt.parent
            if are_unequal_compartments(inside_cpt, outside_cpt):
                cpt_updates[inside_cpt.name] = outside_cpt

        def update_cpt(pattern):
            if pattern.compartment is not None:
                pattern.compartment = cpt_updates.get(pattern.compartment.name,
                                                      pattern.compartment)

        for cp in products:
            update_cpt(cp)
            for mp in cp.monomer_patterns:
                update_cpt(mp)

    def _compute_reactant_mappings(self, reactants):
        """
        Computes all mappings between reactants and reactant patterns
        according to subgraph isomorphisms restricted to equal ordering of
        ComplexPatterns in reactants and the reactant pattern.

        Parameters
        ----------
        reactants: Sequence[pysb.ComplexPattern]
            Reactant species
        """
        node_matcher = categorical_node_match('id', default=None)

        # alignment of mps in cps of pattern allows merging of mappings through
        # ChainMap, also enables us to apply the graph diff to the graph of the
        # reactant pattern of all cps in pattern in the end
        mp_count_pattern = autoinc()
        mp_alignment_cp = [
            [next(mp_count_pattern) for _ in cp.monomer_patterns]
            for cp in reactants
        ]

        gms = [
            GraphMatcher(
                cp._as_graph(mp_alignment_cp[icp]),
                rp._as_graph(self.graph_diff.mp_alignment_rp[icp],
                             prefix='rp'),
                node_match=node_matcher
            )
            for icp, (rp, cp)
            in enumerate(zip(self.reactant_pattern.complex_patterns,
                             reactants))
        ]

        # compute all isomorphisms for each reactant
        matches = [
            [mapping for mapping in gm.subgraph_isomorphisms_iter()]
            if not cp.match_once and not self.is_species_transport()
            else [next(gm.subgraph_isomorphisms_iter())]
            for cp, gm in zip(self.reactant_pattern.complex_patterns, gms)
        ]

        # invert and merge mappings for the product of isomorphisms for all
        # reactants
        return [
            dict(ChainMap(*[
               dict((y, x) for x, y in mapping.items())
               for mapping in mappings
            ]))
            for mappings in itertools.product(*matches)
        ], mp_alignment_cp

    def _localize_rate(self, reactants):
        """
        Computes rates according to the tagged species. Only has an effect if
        the rate is defined as local function.

        Parameters
        ----------
        reactants: Sequence[pysb.ComplexPattern]
            Reactant species
        """
        rate = self.rate
        if self._needs_rate_localization:
            rate = self._localfunc(*[reactants[i] for i in self._tag_rp_ids])

        return rate


class GraphDiff:
    """
    Encodes the specific action of a Rule on the graph of a species.

    Parameters
    ----------
    removed_nodes: Tuple[str]
        nodes to be removed
    added_nodes: Tuple[str, Dict[str, Union[str,pysb.Component]]]
        node to be added including attributes
    added_edges: Tuple[Tuple[str,str]]
        edges to be added
    removed_edges: Tuple[Tuple[str,str]]
        edges to be removed
    changed_node_ids: Dict[str, Union[str,pysb.Component]]
        nodes for which the id is to be changed
    """
    def __init__(self, removed_nodes, added_nodes, added_edges,
                 removed_edges, changed_node_ids):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.added_edges = added_edges
        self.removed_edges = removed_edges
        self.changed_node_ids = changed_node_ids

    def __hash__(self):
        return hash((
            # use frozenset such that ordering doesnt matter
            # convert values to strings such that Components become hashable
            frozenset((k, str(v)) for k, v in self.changed_node_ids.items()),
            frozenset((name, tuple((k, str(v)) for k, v in data.items()))
                      for name, data in self.added_nodes),
            frozenset(self.removed_nodes),
            # edges must be ordered as they are undirected
            *[frozenset(tuple(sorted(edge)) for edge in getattr(self, attr))
              for attr in ['added_edges', 'removed_edges']]
        ))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def apply(self, ingraph, delete_molecules):
        """
        Apply the encoded graph difference to the graph of the species

        Parameters
        ----------
        ingraph: nx.graph
            graph of the species
        delete_molecules: bool
            indicates whether delete_molecules flag was set for the rule

        Returns
        -------
        outgraph:
            graph of the new species with the rule applied
        """
        outgraph = ingraph.copy()
        dangling_bonds = []
        if delete_molecules:
            for node in self.removed_nodes:
                if isinstance(outgraph.nodes[node]['id'], Monomer):
                    neighborhood = nx.ego_graph(outgraph, node, 2)
                    mp_id = outgraph.nodes[node]['mp_id']
                    for n in neighborhood.nodes:
                        if n in self.removed_nodes:
                            continue  # skip removal here
                        if outgraph.nodes[n]['mp_id'] == mp_id:
                            outgraph.remove_node(n)  # remove nodes from
                            # same monomer
                        else:
                            # dont fix dangling bonds here as we might mess
                            # this up again when adding/removing nodes in
                            # the next steps
                            dangling_bonds.append(n)
        outgraph.remove_nodes_from(self.removed_nodes)
        outgraph.add_nodes_from(self.added_nodes)
        outgraph.add_edges_from(self.added_edges)
        outgraph.remove_edges_from(self.removed_edges)
        if self.changed_node_ids:
            nx.set_node_attributes(outgraph,
                                   self.changed_node_ids, 'id')
        # fix dangling bonds:
        if delete_molecules:
            for node in list(dangling_bonds):
                mp_id = outgraph.nodes[node]['mp_id']
                unbound_node = f'{mp_id}_unbound'
                if unbound_node not in outgraph.nodes():
                    outgraph.add_node(unbound_node, id=NO_BOND,
                                      mp_id=mp_id)
                outgraph.add_edge(node, unbound_node)
        return outgraph


class GraphDiffGenerator:
    """
    Encodes the generic action of a Rule on any graph of a species.

    Parameters
    ----------
    rg: ReactionGenerator
        ReactionGenerator corresponding to the rule that is to be encoded
    move_connected:
        indicates whether move_connected flag was set for this rule

    Attributes
    ----------
    removed_nodes: Tuple[str]
        nodes to be removed
    added_nodes: Tuple[str, Dict[str, Union[str,pysb.Component]]]
        node to be added including attributes
    added_edges: Tuple[Tuple[str,str]]
        edges to be added
    removed_edges: Tuple[Tuple[str,str]]
        edges to be removed
    changed_node_ids: Dict[str, Union[str,pysb.Component]]
        nodes for which the id are to be changed
    compartment_transport: Tuple(Union[None,Compartment],...)
        tuple of source and target compartent for species transport rules,
        otherwise tuple of `None`
    """
    def __init__(self, rg, move_connected):
        # to compute the graph diff we cannot match reactant pattern and
        # product pattern using isomorphism matching. Instead, we follow
        # what bng does and implement a left to right matching based on
        # monomer names of monomer patterns, taking into account tagged
        # patterns
        self.mp_alignment_rp, self.mp_alignment_pp = align_monomer_indices(
            rg.reactant_pattern,
            rg.product_pattern
        )
        rp_graph = rg.reactant_pattern._as_graph(
            prefix='rp', mp_alignments=self.mp_alignment_rp
        )
        pp_graph = rg.product_pattern._as_graph(
            prefix='rp', mp_alignments=self.mp_alignment_pp
        )

        # check whether there is a change in species (not molecule!)
        # compartment
        if not move_connected:
            rp_scompartmentents = {
                idx: cp.compartment
                for cp, mp_alignment in zip(rg.reactant_pattern.complex_patterns,
                                            self.mp_alignment_rp)
                for idx, _ in zip(mp_alignment, cp.monomer_patterns)
            }
            pp_scompartments = {
                idx: cp.compartment
                for cp, mp_alignment in zip(rg.product_pattern.complex_patterns,
                                            self.mp_alignment_pp)
                for idx, _ in zip(mp_alignment, cp.monomer_patterns)
            }
            common_mps = set(rp_scompartmentents.keys()).intersection(
                set(pp_scompartments.keys())
            )
            species_transport = next((
                True
                for mp in common_mps
                if are_unequal_compartments(rp_scompartmentents[mp],
                                            pp_scompartments[mp])
            ), False)
        else:
            species_transport = True

        self.removed_nodes = tuple(
            n for n, d in rp_graph.nodes(data=True)
            if n not in pp_graph
            or n in pp_graph and pp_graph.nodes[n]['id'] != d['id']
        )

        source_cpt_node, source_cpt = next((
            (node, rp_graph.nodes[node]['id'])
            for node in self.removed_nodes
            if node.startswith('compartment')
        ), (None, None))

        # if we remove a compartment node this is indicates that this rule
        # implements a transportation. For species transportation we are not
        # allowed to remove the compartment node as this will irreparably
        # remove connecting edges. Instead, we will change the id of the
        # compartment node. For molecule transportations this is not
        # necessary, and the source/target compartment are only stored for
        # the computation of the volume factor in the reaction rate.
        if source_cpt_node is not None:
            # dont remove cpt node
            if species_transport:
                self.removed_nodes = tuple(
                    node for node in self.removed_nodes
                    if node != source_cpt_node
                )
        rp_graph.remove_nodes_from(self.removed_nodes)

        self.added_nodes = tuple(
            (n, d) for n, d in pp_graph.nodes(data=True)
            if n not in rp_graph
        )
        target_cpt_node, target_cpt = next((
            (n, d['id'])
            for n, d in self.added_nodes
            if isinstance(d['id'], Compartment)
        ), (None, None))
        if target_cpt_node is not None:
            # dont remove cpt node
            if species_transport:
                self.added_nodes = tuple(
                    (n, d) for n, d in self.added_nodes
                    if n != target_cpt_node
                )
        rp_graph.add_nodes_from(self.added_nodes)

        # nx.difference requires two graphs with the same nodes. for species
        # transport rules we want to ignore all removed/added edges for
        # compartment as we implement those changes independently
        if species_transport:
            pp_graph = pp_graph.subgraph(n
                                         for n in pp_graph.nodes()
                                         if n != target_cpt_node)

            rp_graph = rp_graph.subgraph(n
                                         for n in rp_graph.nodes()
                                         if n != source_cpt_node)

        self.removed_edges = tuple(
            nx.difference(rp_graph, pp_graph).edges()
        )
        self.added_edges = tuple(
            nx.difference(pp_graph, rp_graph).edges()
        )

        if source_cpt_node and target_cpt and species_transport:
            self.changed_node_ids = {source_cpt_node: target_cpt}
        else:
            self.changed_node_ids = {}

        self.compartment_transport = (source_cpt, target_cpt)

    def generate(self, mapping):
        """
        Compute the specific action on a given species according to the
        provided mapping

        Parameters
        ----------
        mapping: Dict[str,str]
            maps nodes in the reactant pattern to nodes in the species

        Return
        ------
        graph_diff: GraphDiff
            GraphDiff encoding the specific action
        """
        mapped_mp_ids = {
            key.split('_')[0]: val.split('_')[0]
            for key, val in mapping.items()
            if key.split('_')[1] == 'monomer'
        }

        graph_diff = dict()

        for attr in ['removed_edges', 'added_edges']:
            graph_diff[attr] = tuple(
                (mapping.get(e[0], e[0]),
                 mapping.get(e[1], e[1]))
                for e in self.__getattribute__(attr)
            )

        graph_diff['removed_nodes'] = tuple(
            mapping[n]
            for n in self.removed_nodes
        )
        # for newly synthesized monomers there exists no mapping in extended
        # mapping so we need to fall back to the reaction pattern mp_ids in
        # that case
        graph_diff['added_nodes'] = tuple(
            (mapping.get(n, n),
             dict(mp_id=mapped_mp_ids.get(d['mp_id'], d['mp_id']), id=d['id']))
            for n, d in self.added_nodes
        )
        if self.changed_node_ids:
            graph_diff['changed_node_ids'] = {
                mapping.get(node, node): node_id
                for node, node_id in self.changed_node_ids.items()
            }
        else:
            graph_diff['changed_node_ids'] = dict()

        return GraphDiff(**graph_diff)


def align_monomer_indices(rp, pp):
    """
    Align MonomerPatterns in reactant and product ReactionPattern. This
    implements left to right matching respecting tags.

    Parameters
    ----------
    rp: ReactionPattern
        ReactantPattern
    pp: ReactionPattern
        ProductPattern

    Return
    ------
    rp_alignment, pp_alignment: Sequence[Sequence[int]]
        Sequence of sequences encoding the order of MonomerPatterns
    """
    mp_count = autoinc()
    rp_alignment = [
        [next(mp_count) for _ in cp.monomer_patterns]
        for cp in rp.complex_patterns
    ]

    def tag_label(cp, mp):
        label = ''
        # use a reserved character here to avoid matches to actual monomer
        # names
        if cp._tag is not None:
            label += '#cp' + cp._tag.name
        if mp._tag is not None:
            label += '#mp' + mp._tag.name

        return label

    pp_monos, rp_monos = [{
        (icp, imp):
            mp.monomer.name + tag_label(cp, mp)
        for icp, cp in enumerate(reaction_pattern.complex_patterns)
        for imp, mp in enumerate(cp.monomer_patterns)
    } for reaction_pattern in [pp, rp]]

    pp_alignment = [
        [np.NaN] * len(cp.monomer_patterns)
        for cp in pp.complex_patterns
    ]

    for imono, ((_, _), rp_mono) in enumerate(rp_monos.items()):
        # find first MonomerPattern in productpattern with same monomer name
        index = next((
            (icp, imp) for (icp, imp), pp_mono in pp_monos.items()
            if pp_mono == rp_mono
        ), None)
        # if we find a match, set alignment index and delete to prevent
        # rematch, else continue
        if index is not None:
            pp_alignment[index[0]][index[1]] = imono
            del pp_monos[index]

    # add alignment for all unmatched (synthesized) MonomerPatterns
    for new_count, index in enumerate(pp_monos.keys()):
        pp_alignment[index[0]][index[1]] = -(new_count+1)

    return rp_alignment, pp_alignment


def get_matching_patterns(reactant_pattern, species):
    """
    Computes a matrix of matches between the ComplexPatterns of a
    ReactionPattern and the provided species

    Parameters
    ----------
    reactantpattern: ReactionPattern
        ReactantPattern
    productpattern: ReactionPattern
        ProductPattern

    Return
    ------
    matches: np.ndarray
        matrix of boolean values indicating whether there is a match
    """
    return np.asarray([
            [
                match_complex_pattern(cp, s)
                if s is not None and cp is not None
                else False
                for s in species
            ]
            for cp in reactant_pattern.complex_patterns
    ])


def _compute_volume_factor(reactants, products):
    # converts intrinsic rates to extrinsic rates, for reference see
    # bionetgen/bng2/Perl2/Rxn.pm::get_intensive_to_extensive_units_conversion
    volume_factor = 1
    rcpt = [r.species_compartment() for r in reactants
            if r is not None]
    rcpt = [cpt for cpt in rcpt if cpt is not None]
    if rcpt:
        surfaces = [cpt for cpt in rcpt if cpt.dimension == 2]
        volumes = [cpt for cpt in rcpt if cpt.dimension == 3]
        if surfaces:
            surfaces.pop(0)
        else:
            volumes.pop(0)

        for cpt in surfaces + volumes:
            volume_factor /= cpt.size.value
    else:
        for cpt in [p.species_compartment() for p in products
                    if p is not None]:
            if cpt is not None:
                volume_factor *= cpt.size.value
                break

    return volume_factor


def _compute_stat_factor(reactant_indices):
    stat_factor = 1
    for count in Counter(reactant_indices).values():
        stat_factor /= math.factorial(count)
    return stat_factor


def are_equal_compartments(cpt1, cpt2):
    """Checks whether two compartments are equal, with None leading to
        inequality"""
    if cpt1 is None:
        return False
    if cpt2 is None:
        return False
    return cpt1.name == cpt2.name


def are_unequal_compartments(cpt1, cpt2):
    """Checks whether two compartments are unequal, with None leading to
        equality"""
    if cpt1 is None:
        return False
    if cpt2 is None:
        return False
    return cpt1.name != cpt2.name