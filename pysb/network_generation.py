import networkx as nx
import numpy as np
import sympy as sp

import itertools
import math

from .core import ReactionPattern, Monomer, NO_BOND, Compartment
from .pattern import match_complex_pattern
from networkx.algorithms.isomorphism.vf2userfunc import GraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match
from collections import ChainMap, Counter


class ReactionGenerator:
    def __init__(self, rule, reverse, spm):
        self.name = rule.name
        self.reverse = reverse,
        self.reactant_pattern = rule.product_pattern if reverse else \
            rule.reactant_pattern
        self.product_pattern = rule.reactant_pattern if reverse else \
            rule.product_pattern
        self.rate = rule.rate_reverse if reverse else rule.rate_forward

        self.is_pure_synthesis_rule = \
            len(rule.reactant_pattern.complex_patterns) == 0

        self.delete_molecules = rule.delete_molecules

        self.graph_diff = GraphDiff(self, rule.move_connected)
        self.spm = spm

    def generate_reactions(self, reactant_idx, model):
        # order reactants such that they match
        matches = get_matching_patterns(
            self.reactant_pattern, [model.species[ix]
                                    for ix in reactant_idx]
        )

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

    def _generate_reactions_mapped_cps(self, reactant_indices, model):
        reactants = [model.species[ix] for ix in reactant_indices]
        reactant_mappings, mp_alignment_cp = \
            self._compute_reactant_mapping(reactants)

        reactant_graph = ReactionPattern(reactants)._as_graph(
            mp_alignment_cp
        )

        reactions = [
            self._generate_reaction(reactant_indices, reactant_mapping,
                                    reactant_graph, model)
            for reactant_mapping in reactant_mappings
        ]
        # filter reactions with duplicate products, is there a smarter way?
        return {
            rxn['products']: rxn
            for rxn in reactions
        }.values()

    def _generate_reaction(self, reactant_indices, reactant_mapping,
                           reactant_graph, model):

        sfactor = _compute_stat_factor(reactant_indices)
        reactants = [model.species[ix] for ix in reactant_indices]

        product_graph = self.graph_diff.apply(
            reactant_mapping, reactant_graph, self.delete_molecules
        )

        products = ReactionPattern._from_graph(product_graph).complex_patterns

        self.fix_compartments(products, model)

        vfactor = _compute_volume_factor(
            *self.graph_diff.compartment_transport
        )

        reaction = {
            'rule': self.name,
            'rate': self.rate * vfactor * sfactor * np.prod([
                sp.Symbol(f'__s{ix}') for ix in reactant_indices
            ]),
            'product_patterns': products,
            'reactant_patterns': reactants,
            'reactants': reactant_indices,
            'products': tuple(self.spm.match(product, index=True,
                                             exact=True)[0]
                              for product in products)
        }
        return reaction

    def fix_compartments(self, products,  model):
        reactant_cpt, product_cpt = self.graph_diff.compartment_transport
        if not are_unequal_compartments(reactant_cpt, product_cpt):
            return

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

    def _compute_reactant_mapping(self, reactants):
        node_matcher = categorical_node_match('id', default=None)

        def autoinc():
            i = 0
            while True:
                yield i
                i += 1

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
            for gm in gms
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


class GraphDiff:
    def __init__(self, rg, move_connected):
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

        self._mapped_removed_edges = ()
        self._mapped_added_edges = ()
        self._mapped_removed_nodes = ()
        self._mapped_added_nodes = ()
        self._mapped_changed_node_ids = {}

        self.compartment_transport = (source_cpt, target_cpt)

    def _apply_mapping(self, mapping):
        mapped_mp_ids = {
            key.split('_')[0]: val.split('_')[0]
            for key, val in mapping.items()
            if key.split('_')[1] == 'monomer'
        }
        extended_mapping = {
            **mapping,
            **{
                f'{mp_id}_unbound': f'{mapped_mp_id}_unbound'
                for mp_id, mapped_mp_id in mapping.items()
            }
        }

        for attr in ['removed_edges', 'added_edges']:
            self.__setattr__(
                f'_mapped_{attr}',
                tuple(
                    (extended_mapping.get(e[0], e[0]),
                     extended_mapping.get(e[1], e[1]))
                    for e in self.__getattribute__(attr)
                )
            )

        self._mapped_removed_nodes = tuple(
            extended_mapping[n]
            for n in self.removed_nodes
        )
        # for newly synthesized monomers there exists no mapping in extended
        # mapping so we need to fall back to the reaction pattern mp_ids in
        # that case
        self._mapped_added_nodes = tuple(
            (extended_mapping.get(n, n),
             dict(mp_id=mapped_mp_ids.get(d['mp_id'], d['mp_id']), id=d['id']))
            for n, d in self.added_nodes
        )
        if self.changed_node_ids:
            self._mapped_changed_node_ids = {
                extended_mapping.get(node, node): id
                for node, id in self.changed_node_ids.items()
            }

    def apply(self, mapping, ingraph, delete_molecules):
        self._apply_mapping(mapping)
        outgraph = ingraph.copy()
        dangling_bonds = []
        if delete_molecules:
            for node in self._mapped_removed_nodes:
                if isinstance(outgraph.nodes[node]['id'], Monomer):
                    neighborhood = nx.ego_graph(outgraph, node, 2)
                    mp_id = outgraph.nodes[node]['mp_id']
                    for n in neighborhood.nodes:
                        if n in self._mapped_removed_nodes:
                            continue  # skip removal here
                        if outgraph.nodes[n]['mp_id'] == mp_id:
                            outgraph.remove_node(n)  # remove nodes from
                            # same monomer
                        else:
                            # dont fix dangling bonds here as we might mess
                            # this up again when adding/removing nodes in
                            # the next steps
                            dangling_bonds.append(n)
        outgraph.remove_nodes_from(self._mapped_removed_nodes)
        outgraph.add_nodes_from(self._mapped_added_nodes)
        outgraph.add_edges_from(self._mapped_added_edges)
        outgraph.remove_edges_from(self._mapped_removed_edges)
        if self._mapped_changed_node_ids:
            nx.set_node_attributes(outgraph,
                                   self._mapped_changed_node_ids, 'id')
        # fix dangling bonds:
        if delete_molecules:
            for node in list(dangling_bonds):
                mp_id = outgraph.nodes[node]['mp_id']
                if f'{mp_id}_unbound' not in outgraph.nodes():
                    outgraph.add_node(f'{mp_id}_unbound', id=NO_BOND,
                                      mp_id=mp_id)
                outgraph.add_edge(node, f'{mp_id}_unbound')
        return outgraph


def align_monomer_indices(reactantpattern, productpattern):
    def autoinc():
        i = 0
        while True:
            yield i
            i += 1

    mp_count = autoinc()
    rp_alignment = [
        [next(mp_count) for _ in cp.monomer_patterns]
        for cp in reactantpattern.complex_patterns
    ]

    rp_monos = [
        mp.monomer.name
        for cp in reactantpattern.complex_patterns
        for mp in cp.monomer_patterns
    ]

    pp_monos = {
        (icp, imp): mp.monomer.name
        for icp, cp in enumerate(productpattern.complex_patterns)
        for imp, mp in enumerate(cp.monomer_patterns)
    }

    pp_alignment = [
        [np.NaN] * len(cp.monomer_patterns)
        for cp in productpattern.complex_patterns
    ]

    for imono, rp_mono in enumerate(rp_monos):
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

    # add alignment for all unmatched MonomerPatterns
    for new_count, index in enumerate(pp_monos.keys()):
        pp_alignment[index[0]][index[1]] = -(new_count+1)

    return rp_alignment, pp_alignment


def get_matching_patterns(reactant_pattern, species):
    return np.asarray([
            [
                match_complex_pattern(cp, s)
                if s is not None and cp is not None
                else False
                for s in species
            ]
            for cp in reactant_pattern.complex_patterns
    ])


def _compute_volume_factor(compartment_reactants, compartment_products):
    volume_factor = 1
    if compartment_reactants is not None and compartment_products is not None \
            and compartment_products.name != compartment_reactants.name:
        volume_factor = 1 * compartment_products.size.value
    return volume_factor


def _compute_stat_factor(reactant_indices):
    stat_factor = 1
    for count in Counter(reactant_indices).values():
        if count == 1:
            continue  # avoid conversion to float to keep consistent with bng
        stat_factor *= 1 / math.factorial(count)
    return stat_factor


def are_equal_compartments(cpt1, cpt2):
    if cpt1 is None:
        return False
    if cpt2 is None:
        return False
    return cpt1.name == cpt2.name


def are_unequal_compartments(cpt1, cpt2):
    if cpt1 is None:
        return False
    if cpt2 is None:
        return False
    return cpt1.name != cpt2.name