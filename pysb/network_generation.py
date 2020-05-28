import networkx as nx
import numpy as np

from .core import ReactionPattern, Monomer, NO_BOND
from networkx.algorithms.isomorphism.vf2userfunc import GraphMatcher
from networkx.algorithms.isomorphism import categorical_node_match
from collections import ChainMap


class ReactionGenerator:
    def __init__(self, rule, reverse):
        self.name = f'{rule.name}{"__reverse" if reverse else ""}'
        self.rule = rule.name,
        self.reverse = reverse,
        self.reactant_pattern = rule.product_pattern if reverse else \
            rule.reactant_pattern
        self.product_pattern = rule.reactant_pattern if reverse else \
            rule.product_pattern

        self.is_pure_synthesis_rule = \
            len(rule.reactant_pattern.complex_patterns) == 0

        self.delete_molecules = rule.delete_molecules

        self.graph_diff = GraphDiff(self)

    def generate_reaction(self, reactants):
        reactant_mapping, mp_alignment_cp = \
            self.compute_reactant_mapping(reactants)

        reactant_graph = ReactionPattern(reactants)._as_graph(
            mp_alignment_cp
        )

        product_graph = self.graph_diff.apply(
            reactant_mapping, reactant_graph, self.delete_molecules
        )

        products = ReactionPattern._from_graph(product_graph).complex_patterns

        reaction = {
            'rule': self.name,
            'product_patterns': products,
            'reactant_patterns': reactants,
        }
        return reaction

    def compute_reactant_mapping(self, reactants):
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

        matches = [
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

        for rpmatch in matches:
            assert (rpmatch.subgraph_is_isomorphic())

        # invert and merge mapping
        return dict(ChainMap(*[
            dict(zip(match.mapping.values(), match.mapping.keys()))
            for match in matches
        ])), mp_alignment_cp


class GraphDiff:
    def __init__(self, reaction_generator):
        self.mp_alignment_rp, self.mp_alignment_pp = align_monomer_indices(
            reaction_generator.reactant_pattern,
            reaction_generator.product_pattern
        )
        rp_graph = reaction_generator.reactant_pattern._as_graph(
            prefix='rp', mp_alignments=self.mp_alignment_rp
        )
        pp_graph = reaction_generator.product_pattern._as_graph(
            prefix='rp', mp_alignments=self.mp_alignment_pp
        )

        self.removed_nodes = tuple(
            n for n, d in rp_graph.nodes(data=True)
            if n not in pp_graph
            or n in pp_graph and pp_graph.nodes[n]['id'] != d['id']
        )
        rp_graph.remove_nodes_from(self.removed_nodes)
        self.added_nodes = tuple(
            (n, d) for n, d in pp_graph.nodes(data=True)
            if n not in rp_graph
        )
        rp_graph.add_nodes_from(self.added_nodes)
        self.removed_edges = tuple(
            nx.difference(rp_graph, pp_graph).edges()
        )
        self.added_edges = tuple(
            nx.difference(pp_graph, rp_graph).edges()
        )
        self._mapped_removed_edges = ()
        self._mapped_added_edges = ()
        self._mapped_removed_nodes = ()
        self._mapped_added_nodes = ()

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