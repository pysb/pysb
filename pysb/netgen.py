"""Native Python reaction network generation for PySB models.

This module provides a pure-Python implementation of reaction network
generation, replacing the external BioNetGen (BNG) dependency for expanding
Rules into lists of Species and Reactions.

Algorithm
---------
Network generation proceeds by iterative rule application (the "BNG algorithm"):

1. **Seed species**: the set of distinct molecular species present in the
   model's initial conditions is used as the starting point.
2. **Synthesis rules**: rules that produce species from nothing (no reactant
   pattern) are fired once unconditionally, adding any new species they create.
3. **Iterative expansion**: in each iteration, :class:`SpeciesPatternMatcher`
   finds all (rule, reactant-combo) pairs where the rule's reactant pattern
   matches the current species list. Each matching combination is passed to
   :func:`_apply_rule_to_species` (or :func:`_apply_delete_molecules` for
   ``delete_molecules`` rules), which constructs the product species. New
   species are added to the list and new reactions are recorded. The loop
   continues until no new species are found (fixed-point convergence) or
   ``max_iterations`` is reached.
4. **Result**: ``NetworkGenerator.species`` contains all reachable species;
   ``NetworkGenerator.reactions`` contains all elementary reactions.
   Call :meth:`NetworkGenerator.populate_model` to write these results back
   into the model's ``species``, ``reactions``, ``reactions_bidirectional``,
   and observable ``species``/``coefficients`` fields in the same format as
   :func:`pysb.bng.generate_equations`.

Call Hierarchy
--------------
The following maps the algorithm steps to the functions that implement them::

    NetworkGenerator.generate_network()        ← main entry point
    ├── _fire_rule_combo()                      ← synthesis rules (once, unconditionally)
    └── main loop (until fixed point):
        ├── _update_match_cache()               ← check new species against rule patterns
        │   ├── _sp_could_match_cp()            ← fast pre-filter (monomer/bond/state counts)
        │   └── _sp_contains_mp() / match_complex_pattern()  ← full isomorphism check
        ├── _fire_rule_combo()                  ← apply each matching reactant combo
        │   └── _apply_rule_to_species()        ← generate products from one combo
        │       ├── _apply_delete_molecules()   ← delete_molecules rules only
        │       ├── _build_all_species_monomer_maps()  ← enumerate rule→species assignments
        │       └── _apply_rule_with_monomer_map()  ← build products for one assignment
        │           ├── _build_src_bond_map()   ← correlate rule and species bond numbers
        │           ├── _find_all_multistate_permutations()  ← resolve symmetric MS slots
        │           └── _get_extra_monomers()   ← carry over bystander monomers
        └── _lookup_or_add_species()            ← O(1) species identity via canonical key
            └── _species_canonical_key()        ← Morgan-like refinement + branch-and-bound

    Species identity helpers:
        _species_canonical_key() → _mp_base_label(), _canonical_key_from_order()

    Rule monomer mapping (product→reactant correspondence):
        _build_rule_monomer_mapping()           ← forward direction
        _build_reverse_rule_mapping()           ← reverse direction

    Rule pattern matching (called by _update_match_cache):
        _sp_could_match_cp()                    ← necessary-condition pre-filter
        _sp_contains_mp()                       ← fast path for single-monomer CPs
        _mp_site_matches_vf2()                  ← per-site check matching VF2 semantics
        _species_matches_rule_site()            ← general per-site condition checker
        _multistate_slots_match()               ← symmetric MultiState slot matching

Design Decisions
----------------
* **Pure Python, no subprocess**: unlike :mod:`pysb.bng`, this module
  requires no external BioNetGen installation. All graph operations are
  performed directly on PySB :class:`~pysb.core.ComplexPattern` objects.
* **Bond-conflict resolution**: when two species are brought together as
  reactants, their bond numbers may collide. :func:`fix_bond_conflicts`
  renumbers bonds to ensure uniqueness before pattern matching.
* **Graph isomorphism via sorting**: species identity is determined by
  :func:`~pysb.pattern.match_complex_pattern` with ``exact=True``, which
  compares species up to monomer-ordering symmetry.
* **MatchOnce**: rules with ``match_once=True`` produce at most one reaction
  per distinct set of reactant species (rather than one per graph-automorphism
  orbit). This is handled in :mod:`pysb.pattern`.
* **DeleteMolecules**: rules with ``delete_molecules=True`` remove
  monomers that appear in the reactant pattern but not the product pattern,
  and break any bonds to those monomers. Connected-component BFS re-fragments
  the remaining monomers into product species.
* **Compartment inference for synthesised monomers**: when a synthesis rule
  produces a monomer that has no explicit compartment annotation on the product
  pattern (e.g. ``Source() -> Source() + Product()``), the compartment is
  inferred from the matched reactant species if all matched monomers share
  exactly one compartment. This replicates BNG behaviour for rules such as
  ``Source()@EC -> Product()``, which BNG resolves to ``Product()@EC``.

Performance and Implementation Notes (for developers)
------------------------------------------------------
The original implementation used NetworkX VF2 subgraph isomorphism for every
species identity check. On ``fceri_ji`` (354 species, 3680 reactions) this
cost ~136 s vs BNG's 2.5 s (~54× slower). The notes below describe how the
current implementation achieves a relative average speedup over BNG (~0.73s
vs 2.3s on fceri_ji; Macbook Air M4 16GB) and is much faster than the original
implementation.

**Canonical species hashing** (primary optimisation, ~30× gain)

Species identity is determined by a canonical hashable key rather than pairwise
graph isomorphism. ``_species_canonical_key`` implements a Morgan-like label
refinement:

1. Each monomer gets an initial label from its name and site conditions
   (``_mp_base_label``), encoding state values and bond/free status symbolically
   (``'BOND'``, ``'STATE'``, ``'BOND_STATE'``, etc.).
2. Labels are iteratively refined by combining each monomer's current label with
   the sorted labels of its bonded neighbours, then normalising to contiguous
   integers. This propagates topological context outward until labels stabilise
   (typically 2–4 rounds for species up to 7 monomers).
3. Monomers that still share a label after convergence are topologically
   symmetric (interchangeable). All permutations of each symmetric group are
   tried; the lexicographically smallest encoding is selected as the canonical
   key. Branch-and-bound pruning aborts a permutation as soon as its partial key
   exceeds the current best, reducing k! to ~k comparisons per group in practice.

The resulting tuple is used as a dict key in ``_species_by_key``
(canonical key → species index), making species lookup O(1).

**Incremental rule firing** (~2–4× additional gain)

Rather than re-matching every rule against every species on every iteration, the
generator maintains a per-``(rule, direction, cp_idx)`` match cache
(``_match_cache``) mapping to the set of species indices that satisfy that
reactant complex-pattern. Each iteration only newly added species are checked
against existing CPs, and existing species against CPs that may fire with new
reactants (``_update_match_cache``). The product of matching index sets is
enumerated via ``itertools.product`` to produce reactant combos.

**Pre-filters before VF2** (``_sp_could_match_cp``)

Three cheap necessary-condition checks are applied before the NetworkX VF2 call
inside ``_update_match_cache``:

1. *Monomer count*: the species must contain at least as many copies of each
   required monomer type as the CP specifies.
2. *Bond count*: the species must have at least as many bonds as the CP
   requires. Bond counts are cached per ``id(cp)``/``id(sp)`` and handle
   ``MultiState`` bivalent sites correctly.
3. *Site-state*: for every ``(monomer, site, state_str)`` triple required by
   the CP, the species must contain at least one monomer with that state. This
   is particularly effective for phosphorylation-state models where most species
   lack a given phospho-state early in generation.

**Unimolecular fast path** (``_sp_contains_mp``, ~30% gain on fceri_ji)

For CPs with exactly one ``MonomerPattern`` and no complex-level compartment
(~67% of CPs in fceri_ji), VF2 is bypassed entirely. ``_sp_contains_mp``
iterates over the species' monomers directly, calling ``_mp_site_matches_vf2``
for each site condition.

Importantly, ``_mp_site_matches_vf2`` differs from ``_species_matches_rule_site``
in one critical way: a bare string state (e.g. ``s='pY'``) is treated as state
*and unbound*. VF2 adds a ``NO_BOND`` graph node for string states, so matching
a bonded monomer in the same state would be a false positive. Using the plain
``_species_matches_rule_site`` here produces extra reactions not generated by BNG.

**Reactant index threading**

The generate-network loop constructs reactant combos as tuples of species indices
(``combo_idxs``). Rather than recomputing canonical keys inside
``_fire_rule_combo`` to recover those same indices, they are threaded through as
the optional ``reactant_indices`` parameter. This is particularly beneficial for
``MultiState`` models where the permutation search in ``_species_canonical_key``
is expensive.

**Instance-scoped caches**

Several caches are keyed by ``id(monomer)`` or ``id(cp)``
(``_sorted_sites_cache``, ``_cp_bond_count_cache``, ``_sp_bond_count_cache``,
``_cp_state_cache``, ``_sp_state_cache``). These are model specific, and so
live on the ``NetworkGenerator`` instance.
"""

import collections
import itertools
import logging
import math
import sympy
import time
import warnings

from pysb.core import (
    ANY,
    WILD,
    ComplexPattern,
    MonomerPattern,
    MultiState,
)
from pysb.pattern import (
    SpeciesPatternMatcher,
    get_bonds_in_pattern,
    match_complex_pattern,
)


# ---------------------------------------------------------------------------
# Utility functions (used throughout this module)
# ---------------------------------------------------------------------------


def _extract_bond(val):
    """Extract a bond number from a site condition value.

    Returns the integer bond number, or None if the value carries no bond.
    Does not recurse into MultiState; use :func:`_extract_bonds_from_val`
    when all bond numbers in a value (including those inside MultiState slots)
    are needed.
    """
    if isinstance(val, int):
        return val
    elif isinstance(val, tuple) and len(val) == 2:
        bond = val[1]
        if isinstance(bond, int):
            return bond
    return None


def _extract_bonds_from_val(val):
    """Return all integer bond numbers contained in a site condition value.

    Unlike :func:`_extract_bond`, this recurses into MultiState slots so that
    bond numbers stored inside multi-valent sites are not missed.

    Parameters
    ----------
    val : site condition value
        An entry from ``MonomerPattern.site_conditions``.

    Returns
    -------
    list of int
        All bond numbers found; empty list if none.
    """
    if isinstance(val, MultiState):
        bonds = []
        for slot in val:
            b = _extract_bond(slot)
            if b is not None:
                bonds.append(b)
        return bonds
    b = _extract_bond(val)
    return [b] if b is not None else []


def _extract_state_from_val(val):
    """Extract the explicit state string from a site condition value, or None.

    Only handles plain (non-MultiState) conditions. MultiState slots are
    skipped to keep the function conservative; callers that need states from
    multi-valent sites must inspect the MultiState object themselves.

    Returns
    -------
    str or None
        The state string if the value is a state or a (state, bond) tuple;
        None otherwise (bond-only, unbound, ANY, WILD, MultiState, …).
    """
    if isinstance(val, str):
        return val
    if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], str):
        return val[0]
    return None


# ---------------------------------------------------------------------------
# Species normalisation
# ---------------------------------------------------------------------------


def _normalise_cp(cp):
    """Expand a complex-level compartment (``@C:`` prefix notation) to monomers.

    In BNGL, ``@C:A(x!1).B(y!1)`` is equivalent to ``A(x!1)@C.B(y!1)@C``
    when all monomers belong to the same compartment. PySB preserves the
    complex-level compartment on :attr:`ComplexPattern.compartment`, but
    species generated by BioNetGen always store the compartment on each
    individual :class:`~pysb.core.MonomerPattern`. This function returns a
    normalised copy of *cp* with ``cp.compartment = None`` and the
    compartment pushed down to every :class:`~pysb.core.MonomerPattern` that
    currently lacks one.

    If *cp* already has ``compartment = None``, the original object is
    returned unchanged.

    Parameters
    ----------
    cp : ComplexPattern
        The pattern to normalise (may or may not carry a complex compartment).

    Returns
    -------
    ComplexPattern
        A normalised copy (or the original if no change was needed).
    """
    if cp.compartment is None:
        return cp
    fallback = cp.compartment
    new_mps = []
    for mp in cp.monomer_patterns:
        if mp.compartment is None:
            new_mp = MonomerPattern(mp.monomer, mp.site_conditions.copy(), fallback)
            new_mp._tag = mp._tag
            new_mps.append(new_mp)
        else:
            new_mps.append(mp)
    return ComplexPattern(new_mps, compartment=None)


# ---------------------------------------------------------------------------
# Canonical species hashing
# ---------------------------------------------------------------------------


def _mp_base_label(mp):
    """Compute the base label for a MonomerPattern, ignoring bond numbers.

    The label captures the monomer type and the *state* (non-bond) part of
    every site condition, enough to distinguish monomers of different types
    or different phosphorylation states. Bond numbers are replaced by
    symbolic tags so the label is bond-number-independent.
    """
    parts = []
    for site in sorted(mp.monomer.sites):
        val = mp.site_conditions.get(site)
        if val is None:
            parts.append((site, "FREE"))
        elif isinstance(val, str):
            parts.append((site, "STATE", val))
        elif isinstance(val, int):
            parts.append((site, "BOND"))
        elif isinstance(val, tuple) and len(val) == 2:
            state, bond = val
            if isinstance(bond, int):
                parts.append((site, "BOND_STATE", state))
            elif bond is None:
                parts.append((site, "STATE", state))
            else:
                # ANY or WILD: not expected in concrete species
                parts.append((site, "STATE_WILD", state))
        elif isinstance(val, MultiState):
            slot_labels = []
            for slot in val:
                if slot is None:
                    slot_labels.append(("FREE",))
                elif isinstance(slot, str):
                    slot_labels.append(("STATE", slot))
                elif isinstance(slot, int):
                    slot_labels.append(("BOND",))
                elif isinstance(slot, tuple) and len(slot) == 2:
                    st, b = slot
                    if isinstance(b, int):
                        slot_labels.append(("BOND_STATE", st))
                    elif b is None:
                        slot_labels.append(("STATE", st))
                    else:
                        slot_labels.append(("STATE_WILD", st))
                else:
                    slot_labels.append(("OTHER", str(slot)))
            # Sort slot labels so symmetric slots contribute equally
            parts.append((site, "MS", tuple(sorted(slot_labels))))
        else:
            parts.append((site, "OTHER", str(val)))
    compartment_name = mp.compartment.name if mp.compartment is not None else None
    return (mp.monomer.name, compartment_name, tuple(parts))


def _species_canonical_key(sp, sorted_sites_cache=None):
    """Return a canonical hashable key for a concrete species.

    The key is invariant under:

    * Permutation of monomer order within the ComplexPattern.
    * Bond renumbering (bond integers are arbitrary labels).

    Uses a Morgan-like iterative refinement to assign each monomer a
    topological label, then enumerates all permutations of same-label
    monomers (symmetric monomers) to find the lexicographically smallest
    canonical encoding.

    For the largest species in fceri_ji (7 monomers), the number of
    symmetric-group permutations tried is at most 4 (two pairs of identical
    monomers each give 2! options).

    Parameters
    ----------
    sp : ComplexPattern
        A concrete species (all sites fully specified).
    sorted_sites_cache : dict or None
        Optional dict mapping ``id(monomer)`` → ``sorted(monomer.sites)``.
        Pass the ``NetworkGenerator._sorted_sites_cache`` instance dict to
        avoid both recomputing and stale-id hits across model instances.

    Returns
    -------
    tuple
        A hashable canonical key, unique per distinct species.
    """
    mps = sp.monomer_patterns
    n = len(mps)

    # Fast path for single-monomer species
    if n == 1:
        return _canonical_key_from_order(mps, [0], {}, sorted_sites_cache)

    # Build bond-endpoint map: bond_num → [(mp_idx, site, slot_idx, state), ...]
    bond_eps = {}  # bond_num → list of (mp_idx, site_name, slot_idx_or_None)
    for i, mp in enumerate(mps):
        for site, val in mp.site_conditions.items():
            if isinstance(val, MultiState):
                for si, slot in enumerate(val):
                    b = _extract_bond(slot)
                    if b is not None:
                        bond_eps.setdefault(b, []).append((i, site, si))
            else:
                b = _extract_bond(val)
                if b is not None:
                    bond_eps.setdefault(b, []).append((i, site, None))

    # Build per-monomer neighbour list: adj[i] = sorted list of neighbour indices
    adj = [[] for _ in range(n)]
    for b, eps in bond_eps.items():
        if len(eps) == 2:
            (i, *_), (j, *_) = eps
            adj[i].append(j)
            adj[j].append(i)

    # Morgan-like label refinement: normalise labels to contiguous integers
    # at every step so that labels are cheap to hash/compare/sort, and so
    # that the canonical key is invariant to monomer ordering.
    labels = [_mp_base_label(mps[i]) for i in range(n)]
    for _ in range(n + 2):
        _lbl_to_int = {lbl: i for i, lbl in enumerate(sorted(set(labels)))}
        int_labels = [_lbl_to_int[lbl] for lbl in labels]
        new_labels = [
            (int_labels[i], tuple(sorted(int_labels[j] for j in adj[i])))
            for i in range(n)
        ]
        if new_labels == labels:
            break
        labels = new_labels

    # Group monomers by (integer) Morgan label.
    label_groups: dict = {}
    for i in range(n):
        label_groups.setdefault(labels[i], []).append(i)
    sorted_groups = [label_groups[lbl] for lbl in sorted(label_groups)]

    # Enumerate orderings (permutations within same-label groups) and return
    # the lexicographically smallest canonical key.
    #
    # Branch-and-bound: bond numbers are assigned in first-encounter order, so
    # a partial prefix of the canonical key is valid. If partial > best_key
    # prefix, no completion can improve, so we prune early.
    best_key = None

    def _explore_canonical_ordering(gi, cur):
        """Branch-and-bound DFS over label-group permutations.

        Tries all orderings within each symmetric group (monomers that share
        the same Morgan label and are therefore interchangeable). Prunes any
        branch whose partial canonical key already exceeds the current best,
        so in practice only a handful of candidates are evaluated.
        """
        nonlocal best_key
        if gi == len(sorted_groups):
            k = _canonical_key_from_order(mps, cur, bond_eps, sorted_sites_cache)
            if best_key is None or k < best_key:
                best_key = k
            return
        grp = sorted_groups[gi]
        if len(grp) == 1:
            if best_key is not None and gi > 0:
                # Pruning: check partial key before recursing into later groups.
                new_cur = cur + grp
                partial = _canonical_key_from_order(
                    mps, new_cur, bond_eps, sorted_sites_cache
                )
                if partial > best_key[: len(new_cur)]:
                    return
                _explore_canonical_ordering(gi + 1, new_cur)
            else:
                _explore_canonical_ordering(gi + 1, cur + grp)
        else:
            for perm in itertools.permutations(grp):
                new_cur = cur + list(perm)
                if best_key is not None:
                    partial = _canonical_key_from_order(
                        mps, new_cur, bond_eps, sorted_sites_cache
                    )
                    if partial > best_key[: len(new_cur)]:
                        continue  # prune: prefix already worse than best
                _explore_canonical_ordering(gi + 1, new_cur)

    _explore_canonical_ordering(0, [])
    return best_key


def _canonical_key_from_order(mps, order, bond_eps, sorted_sites_cache=None):
    """Build a canonical species key given a specific monomer visit order.

    Bond numbers are assigned sequentially in the order they are first
    encountered. For MultiState sites, slots are visited in ascending
    order of their bonded partner's position in ``order`` (earlier-visited
    partners get lower bond numbers), making the key invariant under
    permutations of MultiState slot order in the original species.

    Parameters
    ----------
    mps : list of MonomerPattern
        All monomer patterns in the species.
    order : list of int
        Indices into ``mps`` specifying the traversal order.
    bond_eps : dict
        bond_num → [(mp_idx, site, slot_idx_or_None), ...] from
        :func:`_species_canonical_key`.
    sorted_sites_cache : dict or None
        Optional dict mapping ``id(monomer)`` → ``sorted(monomer.sites)``.

    Returns
    -------
    tuple
        The canonical representation as a nested tuple.
    """
    # Position of each monomer in the visit order (for MS slot sorting)
    mp_to_pos = {idx: pos for pos, idx in enumerate(order)}
    n = len(order)

    # For each (bond_num, mp_idx) pair, what is the visit-order position
    # of the other endpoint?
    bond_partner_pos = {}
    for b, eps in bond_eps.items():
        if len(eps) == 2:
            (i, *_), (j, *_) = eps
            pi, pj = mp_to_pos.get(i, n), mp_to_pos.get(j, n)
            bond_partner_pos[(b, i)] = pj
            bond_partner_pos[(b, j)] = pi

    bond_remap = {}
    next_b = [1]

    def _get_canonical_bond(bond_num):
        """Return the canonical bond number for bond_num, assigning one if new."""
        if bond_num not in bond_remap:
            bond_remap[bond_num] = next_b[0]
            next_b[0] += 1
        return bond_remap[bond_num]

    result = []
    for idx in order:
        mp = mps[idx]
        mon = mp.monomer
        mon_id = id(mon)
        if sorted_sites_cache is not None:
            ss = sorted_sites_cache.get(mon_id)
        else:
            ss = None
        if ss is None:
            ss = sorted(mon.sites)
            if sorted_sites_cache is not None:
                sorted_sites_cache[mon_id] = ss
        sites = []
        for site in ss:
            val = mp.site_conditions.get(site)
            if val is None:
                sites.append((site, None))
            elif isinstance(val, str):
                sites.append((site, val))
            elif isinstance(val, int):
                sites.append((site, _get_canonical_bond(val)))
            elif isinstance(val, tuple) and len(val) == 2:
                state, bond = val
                if isinstance(bond, int):
                    sites.append((site, state, _get_canonical_bond(bond)))
                else:
                    sites.append((site, state, bond))
            elif isinstance(val, MultiState):
                # Sort slots: bonded slots first (by partner visit-order
                # position, then original slot index), then unbound slots
                # (sorted by their value for local canonicalisation).
                bonded_slots = []
                unbound_slots = []
                for si, slot in enumerate(val):
                    b = _extract_bond(slot)
                    if b is not None:
                        ppos = bond_partner_pos.get((b, idx), n)
                        bonded_slots.append((ppos, si, slot, b))
                    else:
                        unbound_slots.append(slot)
                bonded_slots.sort(key=lambda x: (x[0], x[1]))

                slot_reprs = []
                for ppos, si, slot, b in bonded_slots:
                    rb = _get_canonical_bond(b)
                    if isinstance(slot, tuple):
                        slot_reprs.append((slot[0], rb))
                    else:
                        slot_reprs.append(rb)
                # Unbound slots: sort by canonical string representation
                unbound_slots.sort(
                    key=lambda x: (x is None, str(x) if x is not None else "")
                )
                for slot in unbound_slots:
                    slot_reprs.append(slot)
                sites.append((site, "MS", tuple(slot_reprs)))
            else:
                sites.append((site, str(val)))
        compartment_name = mp.compartment.name if mp.compartment is not None else None
        result.append((mp.monomer.name, compartment_name, tuple(sites)))
    return tuple(result)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-filter helpers (fast necessary-condition checks before VF2)
# ---------------------------------------------------------------------------


def _sp_could_match_cp(
    sp,
    cp,
    sp_bond_count_cache=None,
    cp_bond_count_cache=None,
    sp_state_cache=None,
    cp_state_cache=None,
):
    """Return False early if species cannot possibly match pattern cp.

    Applies three fast necessary-condition checks before the expensive subgraph
    isomorphism test:

    1. **Monomer-count filter**: the species must contain at least as many
       copies of each monomer type as the pattern requires.
    2. **Bond-count filter**: the species must contain at least as many bonds
       as the pattern specifies internally. A pattern with k explicit bond
       integers requires a species with ≥k bonds.
    3. **Site-state filter**: every (monomer_name, site, state_str) tuple
       required by the CP must be present in the species. Only plain state
       conditions (``str`` or ``(str, bond)`` values) are checked; MultiState
       slots are skipped to keep the filter conservative.

    Parameters
    ----------
    sp : ComplexPattern
        A candidate species.
    cp : ComplexPattern
        A rule reactant ComplexPattern (may contain wildcards).
    sp_bond_count_cache : dict or None
        Optional dict mapping ``id(sp)`` → bond count for species.
    cp_bond_count_cache : dict or None
        Optional dict mapping ``id(cp)`` → bond count for rule patterns.
    sp_state_cache : dict or None
        Optional dict mapping ``id(sp)`` → frozenset of
        ``(monomer_name, site, state_str)`` tuples for species.
    cp_state_cache : dict or None
        Optional dict mapping ``id(cp)`` → frozenset of
        ``(monomer_name, site, state_str)`` tuples for rule patterns.

    All caches should be the corresponding ``NetworkGenerator`` instance dicts,
    so that counts are reused within a single network-generation run without
    persisting across different model instances.

    Returns
    -------
    bool
        False if ``sp`` definitely cannot match ``cp``; True if it *might*.
    """
    # --- Monomer-count filter ---
    required = collections.Counter(mp.monomer for mp in cp.monomer_patterns)
    available = collections.Counter(mp.monomer for mp in sp.monomer_patterns)
    if not all(available[mon] >= cnt for mon, cnt in required.items()):
        return False

    # --- Bond-count filter ---
    # Number of distinct bond integers in the CP (cached per CP object).
    cp_id = id(cp)
    if cp_bond_count_cache is not None:
        n_cp_bonds = cp_bond_count_cache.get(cp_id)
    else:
        n_cp_bonds = None
    if n_cp_bonds is None:
        bonds: set = set()
        for mp in cp.monomer_patterns:
            for val in mp.site_conditions.values():
                for b in _extract_bonds_from_val(val):
                    bonds.add(b)
        n_cp_bonds = len(bonds)
        if cp_bond_count_cache is not None:
            cp_bond_count_cache[cp_id] = n_cp_bonds

    if n_cp_bonds > 0:
        sp_id = id(sp)
        if sp_bond_count_cache is not None:
            n_sp_bonds = sp_bond_count_cache.get(sp_id)
        else:
            n_sp_bonds = None
        if n_sp_bonds is None:
            bonds = set()
            for mp in sp.monomer_patterns:
                for val in mp.site_conditions.values():
                    for b in _extract_bonds_from_val(val):
                        bonds.add(b)
            n_sp_bonds = len(bonds)
            if sp_bond_count_cache is not None:
                sp_bond_count_cache[sp_id] = n_sp_bonds
        if n_sp_bonds < n_cp_bonds:
            return False

    if cp_state_cache is not None:
        cp_req_states = cp_state_cache.get(cp_id)
    else:
        cp_req_states = None
    if cp_req_states is None:
        states: set = set()
        for mp in cp.monomer_patterns:
            mon_name = mp.monomer.name
            for site, val in mp.site_conditions.items():
                s = _extract_state_from_val(val)
                if s is not None:
                    states.add((mon_name, site, s))
        cp_req_states = frozenset(states)
        if cp_state_cache is not None:
            cp_state_cache[cp_id] = cp_req_states

    if cp_req_states:
        sp_id = id(sp)
        if sp_state_cache is not None:
            sp_avail_states = sp_state_cache.get(sp_id)
        else:
            sp_avail_states = None
        if sp_avail_states is None:
            states = set()
            for mp in sp.monomer_patterns:
                mon_name = mp.monomer.name
                for site, val in mp.site_conditions.items():
                    s = _extract_state_from_val(val)
                    if s is not None:
                        states.add((mon_name, site, s))
            sp_avail_states = frozenset(states)
            if sp_state_cache is not None:
                sp_state_cache[sp_id] = sp_avail_states
        if not cp_req_states <= sp_avail_states:
            return False

    return True


def _exceeds_max_stoich(species, max_stoich):
    """Check whether a species exceeds a max-stoichiometry constraint.

    Parameters
    ----------
    species : ComplexPattern
        A concrete species.
    max_stoich : dict
        Mapping of monomer name (str) to maximum allowed count (int).

    Returns
    -------
    bool
        True if any monomer exceeds its allowed count.
    """
    counts = {}
    for mp in species.monomer_patterns:
        name = mp.monomer.name
        counts[name] = counts.get(name, 0) + 1
    for name, limit in max_stoich.items():
        if counts.get(name, 0) > limit:
            return True
    return False


# ---------------------------------------------------------------------------
# Bond renumbering and rule monomer mapping
# ---------------------------------------------------------------------------


def _renumber_bond_in_pattern(cplx_pat, bond_from, bond_to):
    """Renumber a specific bond in a ComplexPattern.

    Parameters
    ----------
    cplx_pat : ComplexPattern
        The pattern whose bond should be renumbered.
    bond_from : int
        The bond number to replace.
    bond_to : int
        The new bond number.

    Returns
    -------
    ComplexPattern
        A copy of the pattern with the bond renumbered.
    """
    cplx_pat = cplx_pat.copy()
    for mp in cplx_pat.monomer_patterns:
        for sc_name, sc_val in mp.site_conditions.items():
            if isinstance(sc_val, int) and sc_val == bond_from:
                mp.site_conditions[sc_name] = bond_to
            elif (
                isinstance(sc_val, tuple)
                and len(sc_val) == 2
                and sc_val[1] == bond_from
            ):
                mp.site_conditions[sc_name] = (sc_val[0], bond_to)
            elif isinstance(sc_val, MultiState):
                new_slots = []
                changed = False
                for slot in sc_val:
                    if isinstance(slot, int) and slot == bond_from:
                        new_slots.append(bond_to)
                        changed = True
                    elif (
                        isinstance(slot, tuple)
                        and len(slot) == 2
                        and slot[1] == bond_from
                    ):
                        new_slots.append((slot[0], bond_to))
                        changed = True
                    else:
                        new_slots.append(slot)
                if changed:
                    mp.site_conditions[sc_name] = MultiState(*new_slots)
    return cplx_pat


def fix_bond_conflicts(list_of_cps):
    """Ensure bond numbers are unique across a list of ComplexPatterns.

    When combining ComplexPatterns from different sources (e.g. reactant
    species), the same bond number may appear in multiple patterns.
    This function renumbers bonds so that each bond number is only used
    within a single ComplexPattern.

    Parameters
    ----------
    list_of_cps : list of ComplexPattern
        The patterns to fix. Modified in place and returned.

    Returns
    -------
    list of ComplexPattern
        The input list with bonds renumbered to avoid conflicts.
    """
    bonds_used = collections.defaultdict(set)
    for i, cp in enumerate(list_of_cps):
        for b in get_bonds_in_pattern(cp):
            bonds_used[b].add(i)

    next_avail_bond = max(bonds_used.keys()) + 1 if bonds_used.keys() else 1
    for bond, pat_idxs in bonds_used.items():
        if len(pat_idxs) < 2:
            continue
        for pat_idx in sorted(pat_idxs)[1:]:
            list_of_cps[pat_idx] = _renumber_bond_in_pattern(
                list_of_cps[pat_idx], bond, next_avail_bond
            )
            next_avail_bond += 1

    return list_of_cps


def _build_rule_monomer_mapping(rule):
    """Build a mapping from reactant-side to product-side monomers.

    For each monomer on the product side of a rule, determine which
    monomer on the reactant side it corresponds to. This is done by
    matching monomer types positionally within each ComplexPattern, then
    across ComplexPatterns.

    Parameters
    ----------
    rule : Rule
        The rule to analyse.

    Returns
    -------
    dict
        Mapping of ``(product_cp_idx, product_mp_idx)`` to
        ``(reactant_cp_idx, reactant_mp_idx)``.
    """
    rp_cps = rule.reactant_pattern.complex_patterns
    pp_cps = rule.product_pattern.complex_patterns

    # Track which reactant monomers have been claimed
    claimed = set()
    mapping = {}

    for pp_cp_idx, pp_cp in enumerate(pp_cps):
        for pp_mp_idx, pp_mp in enumerate(pp_cp.monomer_patterns):
            best_match = None
            best_specificity = -1

            for rp_cp_idx, rp_cp in enumerate(rp_cps):
                for rp_mp_idx, rp_mp in enumerate(rp_cp.monomer_patterns):
                    if (rp_cp_idx, rp_mp_idx) in claimed:
                        continue
                    if rp_mp.monomer != pp_mp.monomer:
                        continue

                    # Count matching site conditions as a specificity
                    # score - prefer the reactant monomer whose
                    # specified sites best match the product monomer
                    specificity = 0
                    for site, pp_val in pp_mp.site_conditions.items():
                        if site not in rp_mp.site_conditions:
                            continue
                        rp_val = rp_mp.site_conditions[site]
                        if rp_val == pp_val:
                            specificity += 2
                        else:
                            # Compatible (same monomer type) but different
                            # condition (e.g. state change, bond creation).
                            specificity += 1

                    if specificity > best_specificity:
                        best_specificity = specificity
                        best_match = (rp_cp_idx, rp_mp_idx)

            if best_match is not None:
                mapping[(pp_cp_idx, pp_mp_idx)] = best_match
                claimed.add(best_match)

    return mapping


def _build_reverse_rule_mapping(rule):
    """Build monomer mapping for a rule applied in reverse.

    When applying a rule in reverse, the product pattern becomes the
    source (what we match against) and the reactant pattern becomes the
    target (what we generate). Mirror of :func:`_build_rule_monomer_mapping`.

    Parameters
    ----------
    rule : Rule
        A reversible rule.

    Returns
    -------
    dict
        Mapping of ``(tgt_cp_idx, tgt_mp_idx)`` to
        ``(src_cp_idx, src_mp_idx)``, where *src* is the product
        pattern and *tgt* is the reactant pattern.
    """
    # In reverse: we match against the product pattern (src) and
    # generate the reactant pattern (tgt)
    src_cps = rule.product_pattern.complex_patterns
    tgt_cps = rule.reactant_pattern.complex_patterns

    claimed = set()
    mapping = {}

    for tgt_cp_idx, tgt_cp in enumerate(tgt_cps):
        for tgt_mp_idx, tgt_mp in enumerate(tgt_cp.monomer_patterns):
            best_match = None
            best_specificity = -1

            for src_cp_idx, src_cp in enumerate(src_cps):
                for src_mp_idx, src_mp in enumerate(src_cp.monomer_patterns):
                    if (src_cp_idx, src_mp_idx) in claimed:
                        continue
                    if src_mp.monomer != tgt_mp.monomer:
                        continue

                    specificity = 0
                    for site, tgt_val in tgt_mp.site_conditions.items():
                        if site not in src_mp.site_conditions:
                            continue
                        src_val = src_mp.site_conditions[site]
                        if src_val == tgt_val:
                            specificity += 2
                        else:
                            specificity += 1

                    if specificity > best_specificity:
                        best_specificity = specificity
                        best_match = (src_cp_idx, src_mp_idx)

            if best_match is not None:
                mapping[(tgt_cp_idx, tgt_mp_idx)] = best_match
                claimed.add(best_match)

    return mapping


# ---------------------------------------------------------------------------
# Rule application core
# ---------------------------------------------------------------------------


def _apply_rule_to_species(rule, reactant_combo, rev_dir=False):
    """Apply a rule to a combination of reactant species.

    This is the core species generation algorithm. Given a rule and a
    tuple of reactant species that match the rule's reactant pattern,
    produce **all** product species by enumerating every valid monomer
    mapping from the rule's source pattern to the species.

    When a species contains multiple monomers of the same type that all
    match a rule monomer, each distinct mapping may produce different
    products (separate reactions in BNG semantics).

    Parameters
    ----------
    rule : Rule
        The rule being applied.
    reactant_combo : tuple of ComplexPattern
        The reactant species (one per ComplexPattern in the rule's
        reactant pattern).
    rev_dir : bool, optional
        If True, apply the rule in reverse (swap reactant and product
        patterns).

    Returns
    -------
    list of list of ComplexPattern
        A list of product-species-lists; one inner list per distinct
        monomer mapping.
    """
    if rev_dir:
        src_cps = rule.product_pattern.complex_patterns
        tgt_cps = rule.reactant_pattern.complex_patterns
    else:
        src_cps = rule.reactant_pattern.complex_patterns
        tgt_cps = rule.product_pattern.complex_patterns

    if not tgt_cps and not rule.delete_molecules:
        return [[]]

    # For delete_molecules with no explicit products, we still need to
    # process the remaining (non-deleted) monomers in the species
    if rule.delete_molecules:
        return _apply_delete_molecules(rule, src_cps, tgt_cps, reactant_combo, rev_dir)

    if all(cp.is_concrete() for cp in tgt_cps):
        return [[cp.copy() for cp in tgt_cps]]

    if rev_dir:
        rule_mapping = _build_reverse_rule_mapping(rule)
    else:
        rule_mapping = _build_rule_monomer_mapping(rule)

    reactant_combo_copies = [cp.copy() for cp in reactant_combo]
    reactant_combo_copies = fix_bond_conflicts(reactant_combo_copies)

    all_sp_maps = _build_all_species_monomer_maps(src_cps, reactant_combo_copies)

    all_results = []
    for species_monomer_map in all_sp_maps:
        products_list = _apply_rule_with_monomer_map(
            rule,
            src_cps,
            tgt_cps,
            rule_mapping,
            species_monomer_map,
            reactant_combo_copies,
        )
        all_results.extend(products_list)

    return all_results


def _apply_rule_with_monomer_map(
    rule,
    src_cps,
    tgt_cps,
    rule_mapping,
    species_monomer_map,
    reactant_combo_copies,
):
    """Apply a rule using a specific species monomer mapping.

    This is the inner product-building logic, factored out so that
    :func:`_apply_rule_to_species` can call it once per distinct mapping.

    When a non-MultiState rule site matches a MultiState species site,
    multiple slots may be valid match targets. Each slot choice produces
    a distinct product (separate reaction), so this function returns
    **all** valid product-species-lists.

    Parameters
    ----------
    rule : Rule
        The rule being applied.
    src_cps : list of ComplexPattern
        Rule's source (reactant-side) ComplexPatterns.
    tgt_cps : list of ComplexPattern
        Rule's target (product-side) ComplexPatterns.
    rule_mapping : dict
        Product-to-reactant monomer mapping from
        :func:`_build_rule_monomer_mapping`.
    species_monomer_map : dict
        Mapping from ``(rule_cp_idx, rule_mp_idx)`` to the actual
        species :class:`~pysb.core.MonomerPattern`.
    reactant_combo_copies : list of ComplexPattern
        Bond-conflict-resolved copies of the reactant species.

    Returns
    -------
    list of list of ComplexPattern
        Each inner list is one valid set of product species. Empty list
        if all slot combinations produce invalid (e.g. dangling-bond)
        products.
    """
    # Build correspondence from rule source bond numbers to species bond
    # numbers.
    src_bond_map = _build_src_bond_map(src_cps, species_monomer_map)

    # Determine the unique compartment of all matched species monomers.
    # When a synthesised product monomer carries no explicit compartment,
    # we fall back to this value so that rules of the form
    # ``Source()@EC -> Product()`` (with no compartment on the product
    # pattern) correctly produce ``Product()@EC``. If the matched
    # monomers span more than one compartment (multi-compartment rule),
    # we cannot infer a single fallback, so leave it as None.
    _sp_compartments = {
        sp_mp.compartment
        for sp_mp in species_monomer_map.values()
        if sp_mp.compartment is not None
    }
    _species_context_compartment = (
        next(iter(_sp_compartments)) if len(_sp_compartments) == 1 else None
    )

    # Find the next available bond number across reactant species and
    # target (product) pattern, so new bonds don't collide with either.
    bond_set = set()
    for cp in reactant_combo_copies:
        bond_set.update(get_bonds_in_pattern(cp))
    for cp in tgt_cps:
        bond_set.update(get_bonds_in_pattern(cp))
    next_bond = max(bond_set) + 1 if bond_set else 1

    # Build a mapping from product-side bond numbers to concrete bond
    # numbers. For bonds that are preserved from the source side (same
    # bond number on both LHS and RHS of the rule), map to the
    # corresponding species bond number so that extra monomers' bonds
    # stay consistent. For genuinely new bonds (product-only), allocate
    # fresh sequential numbers.
    src_rule_bonds = set()
    for src_cp in src_cps:
        src_rule_bonds.update(get_bonds_in_pattern(src_cp))

    product_bond_map = {}
    for tgt_cp in tgt_cps:
        for mp in tgt_cp.monomer_patterns:
            for site, val in mp.site_conditions.items():
                for bond in _extract_bonds_from_val(val):
                    if bond not in product_bond_map:
                        if bond in src_rule_bonds and bond in src_bond_map:
                            # Preserved bond: reuse species bond number
                            product_bond_map[bond] = src_bond_map[bond]
                        else:
                            # New bond: allocate fresh number
                            product_bond_map[bond] = next_bond
                            next_bond += 1

    # Track which product-side ComplexPatterns share monomers from the
    # same source species (for carrying over extra monomers)
    product_to_source_species = collections.defaultdict(set)

    # Build a mapping from (src_cp_idx, species_mp_idx) -> target CP
    # index, so we know which product CP each claimed species monomer
    # belongs to.
    species_idx_to_tgt_cp = {}

    # --- Phase 1: Build per-monomer site-condition templates and ---
    # --- collect MultiState slot choice points.                  ---
    # mp_template: (tgt_cp_idx, tgt_mp_idx) -> (MonomerPattern_from_tgt, dict_of_site_conditions)
    mp_template = {}
    slot_choice_points = []

    for tgt_cp_idx, tgt_cp in enumerate(tgt_cps):
        for tgt_mp_idx, tgt_mp in enumerate(tgt_cp.monomer_patterns):
            new_mp_sc = {}
            # Compartment for this product monomer: prefer an explicit
            # annotation on the product pattern, then the matched source
            # species monomer, then fall back to the species context
            # compartment (the unique compartment of all matched species
            # monomers) for synthesised monomers with no explicit
            # compartment (e.g. ``Source()@EC -> Product()``).
            mp_compartment = tgt_mp.compartment

            for site in tgt_mp.monomer.sites:
                new_mp_sc[site] = None

            rule_key = (tgt_cp_idx, tgt_mp_idx)
            if rule_key in rule_mapping:
                rp_cp_idx, rp_mp_idx = rule_mapping[rule_key]
                sp_key = (rp_cp_idx, rp_mp_idx)

                if sp_key in species_monomer_map:
                    src_sp_mp = species_monomer_map[sp_key]
                    if mp_compartment is None:
                        mp_compartment = src_sp_mp.compartment
                    product_to_source_species[tgt_cp_idx].add(rp_cp_idx)

                    sp_cp = reactant_combo_copies[rp_cp_idx]
                    for sp_mp_idx, mp in enumerate(sp_cp.monomer_patterns):
                        if mp is src_sp_mp:
                            species_idx_to_tgt_cp[(rp_cp_idx, sp_mp_idx)] = tgt_cp_idx
                            break

                    for site, val in src_sp_mp.site_conditions.items():
                        new_mp_sc[site] = val
            else:
                # Synthesised monomer: inherit species context compartment
                # when the product pattern gives no explicit compartment.
                if mp_compartment is None:
                    mp_compartment = _species_context_compartment

            # ANY and WILD are conditions, not concrete values.
            for site, val in tgt_mp.site_conditions.items():
                existing = new_mp_sc.get(site)
                is_species_multistate = isinstance(existing, MultiState)

                if val is ANY or val is WILD:
                    continue
                elif isinstance(val, MultiState):
                    existing_slots = list(existing) if is_species_multistate else []
                    product_slots = list(val)
                    all_perms = []
                    if rule_key in rule_mapping:
                        rp_ci, rp_mi = rule_mapping[rule_key]
                        rp_mp = src_cps[rp_ci].monomer_patterns[rp_mi]
                        reactant_ms = rp_mp.site_conditions.get(site)
                        if isinstance(reactant_ms, MultiState) and existing_slots:
                            all_perms = _find_all_multistate_permutations(
                                existing_slots,
                                list(reactant_ms),
                                bond_map=src_bond_map,
                            )
                    if not all_perms:
                        all_perms = [None]  # identity permutation

                    def _apply_multistate_perm(perm, existing_slots, product_slots):
                        """Apply product-side MultiState slots using slot permutation perm.

                        perm[rule_idx] = species_slot_idx; None means identity order.
                        Returns a new MultiState with product modifications applied.
                        """
                        new_slots = (
                            list(existing_slots)
                            if existing_slots
                            else [None] * len(product_slots)
                        )
                        for rule_idx, slot in enumerate(product_slots):
                            sp_idx = perm[rule_idx] if perm else rule_idx
                            ex = (
                                existing_slots[sp_idx]
                                if sp_idx < len(existing_slots)
                                else None
                            )
                            resolved = _resolve_multistate_slot(
                                slot, ex, product_bond_map
                            )
                            new_slots[sp_idx] = resolved
                        return MultiState(*new_slots)

                    if len(all_perms) > 1:
                        # Multiple permutations produce different products
                        alternatives = [
                            _apply_multistate_perm(p, existing_slots, product_slots)
                            for p in all_perms
                        ]
                        # Deduplicate (some permutations may produce the
                        # same result due to symmetry)
                        seen = set()
                        unique_alts = []
                        for alt in alternatives:
                            key = tuple(alt)
                            if key not in seen:
                                seen.add(key)
                                unique_alts.append(alt)
                        if len(unique_alts) > 1:
                            slot_choice_points.append(
                                (
                                    tgt_cp_idx,
                                    tgt_mp_idx,
                                    site,
                                    unique_alts,
                                )
                            )
                        new_mp_sc[site] = unique_alts[0]
                    else:
                        new_mp_sc[site] = _apply_multistate_perm(
                            all_perms[0], existing_slots, product_slots
                        )
                elif is_species_multistate:
                    # Non-MultiState rule site matching a MultiState species
                    # site. Enumerate ALL valid slot matches.
                    existing_slots = list(existing)
                    reactant_val = None
                    if rule_key in rule_mapping:
                        rp_ci, rp_mi = rule_mapping[rule_key]
                        rp_mp = src_cps[rp_ci].monomer_patterns[rp_mi]
                        reactant_val = rp_mp.site_conditions.get(site)

                    matched_indices = _find_all_matched_multistate_slots(
                        existing_slots, reactant_val, src_bond_map
                    )
                    # Resolve the product value
                    resolved = _resolve_plain_to_slot(val, product_bond_map)

                    if len(matched_indices) > 1:
                        # Multiple valid slots: record as a choice point
                        alternatives = []
                        for idx in matched_indices:
                            new_slots = list(existing_slots)
                            new_slots[idx] = resolved
                            alternatives.append(MultiState(*new_slots))
                        # Deduplicate
                        seen = set()
                        unique_alts = []
                        for alt in alternatives:
                            key = tuple(alt)
                            if key not in seen:
                                seen.add(key)
                                unique_alts.append(alt)
                        if len(unique_alts) > 1:
                            slot_choice_points.append(
                                (
                                    tgt_cp_idx,
                                    tgt_mp_idx,
                                    site,
                                    unique_alts,
                                )
                            )

                    # Use first match as the template default
                    new_slots = list(existing_slots)
                    new_slots[matched_indices[0]] = resolved
                    new_mp_sc[site] = MultiState(*new_slots)
                elif isinstance(val, tuple) and len(val) == 2:
                    state, bond = val
                    if bond is ANY or bond is WILD:
                        if isinstance(existing, tuple) and len(existing) == 2:
                            new_mp_sc[site] = (state, existing[1])
                        elif isinstance(existing, int):
                            new_mp_sc[site] = (state, existing)
                        else:
                            new_mp_sc[site] = state
                    elif isinstance(bond, int):
                        new_mp_sc[site] = (state, product_bond_map[bond])
                    else:
                        new_mp_sc[site] = val
                elif isinstance(val, int):
                    new_bond = product_bond_map[val]
                    existing = new_mp_sc.get(site)
                    if isinstance(existing, str):
                        # Source species has a state on this site; the rule
                        # adds a bond while leaving the state unchanged.
                        # Produce (state, bond) rather than discarding state.
                        new_mp_sc[site] = (existing, new_bond)
                    else:
                        new_mp_sc[site] = new_bond
                elif val is None:
                    existing = new_mp_sc.get(site)
                    if isinstance(existing, tuple) and len(existing) == 2:
                        # Source species has (state, bond); product says free.
                        # Preserve the state, strip the bond.
                        new_mp_sc[site] = existing[0]
                    else:
                        new_mp_sc[site] = val
                else:
                    new_mp_sc[site] = val

            mp_template[(tgt_cp_idx, tgt_mp_idx)] = (tgt_mp, new_mp_sc, mp_compartment)

    # --- Phase 2: Enumerate all slot-choice combinations and build ---
    # --- concrete product species for each.                        ---
    # Each choice point is (tgt_cp_idx, tgt_mp_idx, site, [MultiState_alt, ...]).
    # We take the Cartesian product over all choice points.

    if slot_choice_points:
        choice_alternatives = []
        for cp_idx, mp_idx, site, alternatives in slot_choice_points:
            alts = [(cp_idx, mp_idx, site, alt) for alt in alternatives]
            choice_alternatives.append(alts)
        slot_combos = list(itertools.product(*choice_alternatives))
    else:
        slot_combos = [()]  # Single combo with no overrides

    all_valid_products = []
    for combo in slot_combos:
        # Build override dict: (tgt_cp_idx, tgt_mp_idx, site) -> MultiState value
        overrides = {}
        for cp_idx, mp_idx, site, ms_val in combo:
            overrides[(cp_idx, mp_idx, site)] = ms_val

        gen_species_list = []
        valid = True
        for tgt_cp_idx, tgt_cp in enumerate(tgt_cps):
            new_mps = []
            for tgt_mp_idx, tgt_mp in enumerate(tgt_cp.monomer_patterns):
                tgt_mp_ref, base_sc, mp_compartment = mp_template[
                    (tgt_cp_idx, tgt_mp_idx)
                ]
                # Copy the template and apply any slot-choice overrides
                new_mp_sc = dict(base_sc)
                for (oc_ci, oc_mi, oc_site), oc_val in overrides.items():
                    if oc_ci == tgt_cp_idx and oc_mi == tgt_mp_idx:
                        new_mp_sc[oc_site] = oc_val
                new_mp = MonomerPattern(tgt_mp_ref.monomer, new_mp_sc, mp_compartment)
                new_mps.append(new_mp)

            # Carry over any extra monomers from source species that were
            # not mentioned in the rule's reactant pattern
            extra_mps = _get_extra_monomers(
                reactant_combo_copies,
                species_monomer_map,
                tgt_cp_idx,
                product_to_source_species,
                species_idx_to_tgt_cp,
            )
            new_mps.extend(extra_mps)

            # Validate bond consistency: every bond number must appear exactly twice
            _bond_counts = collections.Counter()
            for _mp in new_mps:
                for _site, _val in _mp.site_conditions.items():
                    for _b in _extract_bonds_from_val(_val):
                        _bond_counts[_b] += 1
            for _b, _c in _bond_counts.items():
                if _c != 2:
                    logger.debug(
                        "Dangling bond %d (count=%d) in monomers %s "
                        "(rule: %s, reactants: %s, mapping: %s)",
                        _b,
                        _c,
                        new_mps,
                        rule.name,
                        reactant_combo_copies,
                        species_monomer_map,
                    )
                    valid = False
                    break
            if not valid:
                break

            # Split into connected components; a rule that breaks a
            # bond within a single product CP can disconnect monomers.
            components = _split_connected_components(new_mps)
            for comp_mps in components:
                comp_sp = ComplexPattern(comp_mps, compartment=None)
                if not comp_sp.is_concrete():
                    raise ValueError(
                        f"Generated species {comp_sp} is not concrete "
                        f"(rule: {rule.name}, reactants: {reactant_combo_copies})"
                    )
                gen_species_list.append(comp_sp)

        if valid:
            all_valid_products.append(gen_species_list)

    return all_valid_products


def _split_connected_components(mps):
    """Split a list of MonomerPatterns into connected components by bonds.

    Two monomers are connected if they share a bond number. Returns a
    list of lists, each inner list being a connected component. When all
    monomers are connected, returns ``[mps]`` unchanged.

    Parameters
    ----------
    mps : list of MonomerPattern
        The monomers to split.

    Returns
    -------
    list of list of MonomerPattern
        One list per connected component, preserving order within each.
    """
    if len(mps) <= 1:
        return [mps]

    # Build bond graph: bond number → set of monomer indices
    bond_to_mps = collections.defaultdict(set)
    for i, mp in enumerate(mps):
        for site, val in mp.site_conditions.items():
            for bond in _extract_bonds_from_val(val):
                bond_to_mps[bond].add(i)

    # Build adjacency
    adjacency = collections.defaultdict(set)
    for bond_num, mp_idxs in bond_to_mps.items():
        mp_list = list(mp_idxs)
        for i in range(len(mp_list)):
            for j in range(i + 1, len(mp_list)):
                adjacency[mp_list[i]].add(mp_list[j])
                adjacency[mp_list[j]].add(mp_list[i])

    # DFS to find connected components.
    visited = set()
    components = []
    for start in range(len(mps)):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbour in adjacency.get(node, set()):
                if neighbour not in visited:
                    queue.append(neighbour)
        components.append(sorted(component))

    if len(components) == 1:
        return [mps]

    return [[mps[i] for i in comp] for comp in components]


def _apply_delete_molecules(rule, src_cps, tgt_cps, reactant_combo, rev_dir):
    """Apply a rule with delete_molecules=True.

    Monomers that appear on the reactant side but not the product side
    are deleted. Bonds to deleted monomers are broken (sites set to
    None). The remaining monomers, plus any explicit product monomers,
    are grouped into connected components to form product species.

    Parameters
    ----------
    rule : Rule
        The rule (must have delete_molecules=True).
    src_cps : list of ComplexPattern
        Rule's source (reactant) pattern ComplexPatterns.
    tgt_cps : list of ComplexPattern
        Rule's target (product) pattern ComplexPatterns.
    reactant_combo : tuple of ComplexPattern
        The actual reactant species.
    rev_dir : bool
        Whether this is the reverse direction.

    Returns
    -------
    list of list of ComplexPattern
        Each inner list is one set of product species (from one monomer
        mapping).
    """
    if rev_dir:
        rule_mapping = _build_reverse_rule_mapping(rule)
    else:
        rule_mapping = _build_rule_monomer_mapping(rule)

    rc_copies = [cp.copy() for cp in reactant_combo]
    rc_copies = fix_bond_conflicts(rc_copies)
    species_monomer_map = _build_species_monomer_map(src_cps, rc_copies)

    src_bond_map = _build_src_bond_map(src_cps, species_monomer_map)

    # Determine the unique compartment across all matched species monomers
    # for use as a fallback when synthesised product monomers carry no
    # explicit compartment (mirrors the logic in _apply_rule_with_monomer_map).
    _dm_sp_compartments = {
        sp_mp.compartment
        for sp_mp in species_monomer_map.values()
        if sp_mp.compartment is not None
    }
    _dm_species_context_compartment = (
        next(iter(_dm_sp_compartments)) if len(_dm_sp_compartments) == 1 else None
    )

    # Identify which species monomers are claimed by the rule's reactant pattern
    claimed_sp_mps = set()  # set of (cp_idx, mp_idx) in species
    for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
        for cp_idx, sp_cp in enumerate(rc_copies):
            if cp_idx != rp_ci:
                continue
            for mp_idx, mp in enumerate(sp_cp.monomer_patterns):
                if mp is sp_mp:
                    claimed_sp_mps.add((cp_idx, mp_idx))
                    break

    # Identify which claimed monomers survive (mapped to a product monomer) vs deleted
    surviving_rule_keys = set()  # (src_cp_idx, src_mp_idx) in rule space
    for (tgt_ci, tgt_mi), (src_ci, src_mi) in rule_mapping.items():
        surviving_rule_keys.add((src_ci, src_mi))

    deleted_sp_mps = set()  # (cp_idx, mp_idx) in species
    surviving_sp_mps = set()  # (cp_idx, mp_idx) in species
    for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
        for cp_idx, sp_cp in enumerate(rc_copies):
            if cp_idx != rp_ci:
                continue
            for mp_idx, mp in enumerate(sp_cp.monomer_patterns):
                if mp is sp_mp:
                    if (rp_ci, rp_mi) in surviving_rule_keys:
                        surviving_sp_mps.add((cp_idx, mp_idx))
                    else:
                        deleted_sp_mps.add((cp_idx, mp_idx))
                    break

    # Collect all monomers from reactant species into a flat list
    all_mps = []
    for cp_idx, sp_cp in enumerate(rc_copies):
        for mp_idx, mp in enumerate(sp_cp.monomer_patterns):
            all_mps.append((cp_idx, mp_idx, mp))

    # Remove deleted monomers
    remaining_mps = [
        (ci, mi, mp) for ci, mi, mp in all_mps if (ci, mi) not in deleted_sp_mps
    ]

    if not remaining_mps and not tgt_cps:
        return [[]]

    # Collect bonds held by deleted monomers, then clear those bonds from
    # remaining monomers.
    deleted_bonds = set()
    for ci, mi, mp in all_mps:
        if (ci, mi) in deleted_sp_mps:
            for site, val in mp.site_conditions.items():
                deleted_bonds.update(_extract_bonds_from_val(val))

    # For remaining monomers, clear any sites that reference deleted
    # bonds. Build new MonomerPatterns with cleaned site conditions.
    cleaned_remaining = []
    for ci, mi, mp in remaining_mps:
        new_sc = {}
        for site, val in mp.site_conditions.items():
            if isinstance(val, MultiState):
                new_slots = []
                for slot in val:
                    slot_bond = _extract_bond(slot)
                    if slot_bond is not None and slot_bond in deleted_bonds:
                        if isinstance(slot, tuple) and len(slot) == 2:
                            new_slots.append(slot[0])
                        else:
                            new_slots.append(None)
                    else:
                        new_slots.append(slot)
                new_sc[site] = MultiState(*new_slots)
            else:
                bond = _extract_bond(val)
                if bond is not None and bond in deleted_bonds:
                    # This bond goes to a deleted monomer: clear it
                    if isinstance(val, tuple) and len(val) == 2:
                        # (state, bond) → keep state, clear bond
                        new_sc[site] = val[0]
                    else:
                        new_sc[site] = None
                else:
                    new_sc[site] = val
        new_mp = MonomerPattern(mp.monomer, new_sc, mp.compartment)
        cleaned_remaining.append((ci, mi, new_mp))

    # Apply product-side modifications to surviving monomers.
    sp_to_rule = {}
    for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
        for cp_idx, sp_cp in enumerate(rc_copies):
            if cp_idx != rp_ci:
                continue
            for mp_idx, mp in enumerate(sp_cp.monomer_patterns):
                if mp is sp_mp:
                    sp_to_rule[(cp_idx, mp_idx)] = (rp_ci, rp_mi)
                    break

    rule_src_to_tgt = {}
    for (tgt_ci, tgt_mi), (src_ci, src_mi) in rule_mapping.items():
        rule_src_to_tgt[(src_ci, src_mi)] = (tgt_ci, tgt_mi)

    remaining_bonds = set()
    for ci, mi, mp in cleaned_remaining:
        for v in mp.site_conditions.values():
            remaining_bonds.update(_extract_bonds_from_val(v))
    if tgt_cps:
        for tgt_cp in tgt_cps:
            remaining_bonds.update(get_bonds_in_pattern(tgt_cp))
    next_bond = max(remaining_bonds) + 1 if remaining_bonds else 1

    product_bond_map = {}
    if tgt_cps:
        for tgt_cp in tgt_cps:
            for mp in tgt_cp.monomer_patterns:
                for site, val in mp.site_conditions.items():
                    for bond in _extract_bonds_from_val(val):
                        if bond not in product_bond_map:
                            product_bond_map[bond] = next_bond
                            next_bond += 1

    # Apply product-side overrides to surviving claimed monomers
    for idx, (ci, mi, mp) in enumerate(cleaned_remaining):
        if (ci, mi) not in surviving_sp_mps:
            continue
        rule_key = sp_to_rule.get((ci, mi))
        if rule_key is None:
            continue
        tgt_key = rule_src_to_tgt.get(rule_key)
        if tgt_key is None:
            continue
        tgt_ci, tgt_mi = tgt_key
        if tgt_ci >= len(tgt_cps):
            continue
        tgt_mp = tgt_cps[tgt_ci].monomer_patterns[tgt_mi]

        new_sc = dict(mp.site_conditions)
        for site, val in tgt_mp.site_conditions.items():
            existing = new_sc.get(site)
            is_species_multistate = isinstance(existing, MultiState)

            if val is ANY or val is WILD:
                continue
            elif isinstance(val, MultiState):
                existing_slots = list(existing) if is_species_multistate else []
                product_slots = list(val)
                # Find the reactant-side permutation
                perm = None
                if rule_key is not None:
                    rp_ci, rp_mi = rule_key
                    rp_mp = src_cps[rp_ci].monomer_patterns[rp_mi]
                    reactant_ms = rp_mp.site_conditions.get(site)
                    if isinstance(reactant_ms, MultiState) and existing_slots:
                        perm = _find_multistate_permutation(
                            existing_slots,
                            list(reactant_ms),
                            bond_map=src_bond_map,
                        )
                new_slots = (
                    list(existing_slots)
                    if existing_slots
                    else [None] * len(product_slots)
                )
                for rule_idx, slot in enumerate(product_slots):
                    sp_idx = perm[rule_idx] if perm else rule_idx
                    ex = (
                        existing_slots[sp_idx] if sp_idx < len(existing_slots) else None
                    )
                    resolved = _resolve_multistate_slot(slot, ex, product_bond_map)
                    new_slots[sp_idx] = resolved
                new_sc[site] = MultiState(*new_slots)
            elif is_species_multistate:
                # Non-MultiState rule site matching a MultiState species site
                existing_slots = list(existing)
                reactant_val = None
                if rule_key is not None:
                    rp_ci, rp_mi = rule_key
                    rp_mp = src_cps[rp_ci].monomer_patterns[rp_mi]
                    reactant_val = rp_mp.site_conditions.get(site)
                matched_indices = _find_all_matched_multistate_slots(
                    existing_slots, reactant_val, src_bond_map
                )
                resolved = _resolve_plain_to_slot(val, product_bond_map)
                new_slots = list(existing_slots)
                new_slots[matched_indices[0]] = resolved
                new_sc[site] = MultiState(*new_slots)
            elif isinstance(val, tuple) and len(val) == 2:
                state, bond = val
                if bond is ANY or bond is WILD:
                    if isinstance(existing, tuple) and len(existing) == 2:
                        new_sc[site] = (state, existing[1])
                    elif isinstance(existing, int):
                        new_sc[site] = (state, existing)
                    else:
                        new_sc[site] = state
                elif isinstance(bond, int):
                    new_sc[site] = (state, product_bond_map[bond])
                else:
                    new_sc[site] = val
            elif isinstance(val, int):
                new_bond = product_bond_map[val]
                if isinstance(existing, str):
                    new_sc[site] = (existing, new_bond)
                else:
                    new_sc[site] = new_bond
            elif val is None:
                if isinstance(existing, tuple) and len(existing) == 2:
                    new_sc[site] = existing[0]
                else:
                    new_sc[site] = val
            else:
                new_sc[site] = val
        cleaned_remaining[idx] = (
            ci,
            mi,
            MonomerPattern(mp.monomer, new_sc, mp.compartment),
        )

    # Also add any purely-synthesised product monomers (product-side
    # monomers that don't correspond to any reactant-side monomer)
    synth_mps = []
    for tgt_ci, tgt_cp in enumerate(tgt_cps):
        for tgt_mi, tgt_mp in enumerate(tgt_cp.monomer_patterns):
            if (tgt_ci, tgt_mi) not in rule_mapping:
                new_sc = {}
                for site in tgt_mp.monomer.sites:
                    new_sc[site] = None
                for site, val in tgt_mp.site_conditions.items():
                    if isinstance(val, int):
                        new_sc[site] = product_bond_map.get(val, val)
                    elif isinstance(val, tuple) and len(val) == 2:
                        state, bond = val
                        if isinstance(bond, int):
                            new_sc[site] = (state, product_bond_map.get(bond, bond))
                        else:
                            new_sc[site] = val
                    elif isinstance(val, MultiState):
                        new_slots = []
                        for slot in val:
                            if isinstance(slot, int):
                                new_slots.append(product_bond_map.get(slot, slot))
                            elif isinstance(slot, tuple) and len(slot) == 2:
                                st, bond = slot
                                if isinstance(bond, int):
                                    new_slots.append(
                                        (st, product_bond_map.get(bond, bond))
                                    )
                                else:
                                    new_slots.append(slot)
                            else:
                                new_slots.append(slot)
                        new_sc[site] = MultiState(*new_slots)
                    else:
                        new_sc[site] = val
                mp_compartment = tgt_mp.compartment
                if mp_compartment is None:
                    mp_compartment = _dm_species_context_compartment
                synth_mps.append(MonomerPattern(tgt_mp.monomer, new_sc, mp_compartment))

    all_product_mps = [mp for _, _, mp in cleaned_remaining] + synth_mps

    if not all_product_mps:
        return [[]]

    bond_to_mps = collections.defaultdict(set)
    for i, mp in enumerate(all_product_mps):
        for site, val in mp.site_conditions.items():
            for bond in _extract_bonds_from_val(val):
                bond_to_mps[bond].add(i)

    adjacency = collections.defaultdict(set)
    for bond_num, mp_idxs in bond_to_mps.items():
        mp_list = list(mp_idxs)
        for i in range(len(mp_list)):
            for j in range(i + 1, len(mp_list)):
                adjacency[mp_list[i]].add(mp_list[j])
                adjacency[mp_list[j]].add(mp_list[i])

    # Find connected components via DFS.
    visited = set()
    components = []
    for start in range(len(all_product_mps)):
        if start in visited:
            continue
        component = []
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbour in adjacency.get(node, set()):
                if neighbour not in visited:
                    queue.append(neighbour)
        components.append(component)

    gen_species_list = []
    for component in components:
        mps = [all_product_mps[i] for i in sorted(component)]
        sp = ComplexPattern(mps, compartment=None)
        if not sp.is_concrete():
            raise ValueError(
                f"Generated species {sp} is not concrete "
                f"(rule: {rule.name}, reactants: {reactant_combo})"
            )
        gen_species_list.append(sp)

    return [gen_species_list]


# ---------------------------------------------------------------------------
# Site matching and species-to-rule monomer mapping
# ---------------------------------------------------------------------------


def _site_match_specificity(sp_val, rule_val):
    """Score how specifically a species site value matches a rule value.

    Returns a higher score for exact matches (same structure) than for
    compatible-but-loose matches. This is used by
    ``_build_species_monomer_map`` to prefer, e.g., a species ``b='Y'``
    over ``b=('Y', 2)`` when the rule says ``b='Y'``.

    Assumes ``_species_matches_rule_site(sp_val, rule_val)`` is True.

    Returns
    -------
    int
        2 for an exact (structural) match, 1 for a compatible match.
    """
    if rule_val is ANY or rule_val is WILD:
        # Wildcards always get low specificity
        return 1
    if rule_val is None:
        return 2 if sp_val is None else 1
    if isinstance(rule_val, str):
        # Rule says 'Y' (state, no bond restriction).
        # Exact: species also has just 'Y'.
        # Loose: species has ('Y', bond).
        if isinstance(sp_val, str) and sp_val == rule_val:
            return 2
        return 1
    if isinstance(rule_val, int):
        # Rule specifies a bond number: any bonded species value matches.
        if isinstance(sp_val, int):
            return 2
        return 1
    if isinstance(rule_val, tuple) and len(rule_val) == 2:
        state, bond = rule_val
        if isinstance(sp_val, tuple) and len(sp_val) == 2:
            state_match = sp_val[0] == state
            if bond is ANY or bond is WILD:
                return 2 if state_match else 1
            if isinstance(bond, int) and isinstance(sp_val[1], int):
                return 2 if state_match else 1
            return 2 if sp_val == rule_val else 1
        return 1
    if isinstance(rule_val, MultiState):
        return 2 if isinstance(sp_val, MultiState) else 1
    return 2 if sp_val == rule_val else 1


def _build_species_monomer_map(rule_src_cps, species_combo):
    """Build a mapping from rule reactant monomers to species monomers.

    Returns the first (best) valid mapping found. For enumerating all
    distinct mappings, use :func:`_build_all_species_monomer_maps`.

    Parameters
    ----------
    rule_src_cps : list of ComplexPattern
        The rule's source (reactant) pattern ComplexPatterns.
    species_combo : tuple or list of ComplexPattern
        The actual species matching each reactant ComplexPattern.

    Returns
    -------
    dict
        Mapping of ``(rule_cp_idx, rule_mp_idx)`` to the
        corresponding MonomerPattern from the species.
    """
    all_maps = _build_all_species_monomer_maps(rule_src_cps, species_combo)
    return all_maps[0] if all_maps else {}


def _build_all_species_monomer_maps(rule_src_cps, species_combo):
    """Build all valid monomer mappings from rule reactants to species.

    When a species contains multiple monomers of the same type that all
    match a rule monomer, there may be several *distinct* assignments
    (each leading to a potentially different product). This function
    enumerates all **perfect** (every rule monomer assigned) mappings
    that are bond-topology-consistent.

    Parameters
    ----------
    rule_src_cps : list of ComplexPattern
        The rule's source (reactant) pattern ComplexPatterns.
    species_combo : tuple or list of ComplexPattern
        The actual species matching each reactant ComplexPattern.

    Returns
    -------
    list of dict
        Each dict maps ``(rule_cp_idx, rule_mp_idx)`` to the
        corresponding MonomerPattern from the species.
    """
    # Build per-CP assignment lists, then take their Cartesian product.
    per_cp_assignments = []  # list of lists; one list per CP

    for cp_idx, (rule_cp, sp_cp) in enumerate(zip(rule_src_cps, species_combo)):
        # Build a compatibility matrix: for each rule monomer, which
        # species monomers it can match (with scores)
        rule_mps = list(enumerate(rule_cp.monomer_patterns))
        sp_mps = list(enumerate(sp_cp.monomer_patterns))

        candidates = {}  # rule_mp_idx -> list of (score, sp_mp_idx)
        for rule_mp_idx, rule_mp in rule_mps:
            cands = []
            for sp_mp_idx, sp_mp in sp_mps:
                if sp_mp.monomer != rule_mp.monomer:
                    continue
                score = 0
                compatible = True
                for site, rule_val in rule_mp.site_conditions.items():
                    sp_val = sp_mp.site_conditions.get(site)
                    if _species_matches_rule_site(sp_val, rule_val):
                        score += _site_match_specificity(sp_val, rule_val)
                    else:
                        compatible = False
                        break
                if compatible:
                    cands.append((score, sp_mp_idx))
            # Sort by score descending so backtracking tries best first
            cands.sort(key=lambda x: -x[0])
            candidates[rule_mp_idx] = cands

        # Enumerate all perfect, bond-topology-consistent assignments.
        n_rule = len(rule_mps)
        all_assignments = []
        assignment = [None] * n_rule
        used_sp = set()
        fwd_bond = {}
        rev_bond = {}

        def _get_bond_pairs(rule_mp_idx, sp_mp_idx):
            """Return (rule_bond, species_bond) pairs for non-MultiState bonded sites."""
            rule_mp = rule_mps_list[rule_mp_idx]
            sp_mp = sp_mps_list[sp_mp_idx]
            pairs = []
            for site, rule_val in rule_mp.site_conditions.items():
                sp_val = sp_mp.site_conditions.get(site)
                if isinstance(rule_val, MultiState) or isinstance(sp_val, MultiState):
                    continue
                r_bonds = _extract_bonds_from_val(rule_val)
                s_bonds = _extract_bonds_from_val(sp_val)
                if len(r_bonds) == 1 and len(s_bonds) == 1:
                    pairs.append((r_bonds[0], s_bonds[0]))
            return pairs

        rule_mps_list = [mp for _, mp in rule_mps]
        sp_mps_list = [mp for _, mp in sp_mps]

        def _try_complete(ri, n_assigned):
            """Backtracking search for a *perfect* (all-rule-monomers-assigned) mapping.

            Assigns rule monomers to species monomers one by one (index ri),
            enforcing bond-topology consistency at each step. Records every
            complete solution in all_assignments. Once a perfect solution is
            found, imperfect branches are skipped.
            """
            if ri == n_rule:
                if n_assigned == n_rule:
                    all_assignments.append(assignment[:])
                return
            rule_mp_idx = rule_mps[ri][0]
            cands = candidates.get(rule_mp_idx, [])
            for _, sp_mp_idx in cands:
                if sp_mp_idx in used_sp:
                    continue
                pairs = _get_bond_pairs(rule_mp_idx, sp_mp_idx)
                consistent = True
                for r_bond, s_bond in pairs:
                    if r_bond in fwd_bond:
                        if fwd_bond[r_bond] != s_bond:
                            consistent = False
                            break
                    if s_bond in rev_bond:
                        if rev_bond[s_bond] != r_bond:
                            consistent = False
                            break
                if not consistent:
                    continue
                added_fwd = []
                added_rev = []
                for r_bond, s_bond in pairs:
                    if r_bond not in fwd_bond:
                        fwd_bond[r_bond] = s_bond
                        added_fwd.append(r_bond)
                    if s_bond not in rev_bond:
                        rev_bond[s_bond] = r_bond
                        added_rev.append(s_bond)
                assignment[ri] = sp_mp_idx
                used_sp.add(sp_mp_idx)
                _try_complete(ri + 1, n_assigned + 1)
                used_sp.discard(sp_mp_idx)
                assignment[ri] = None
                for r_bond in added_fwd:
                    del fwd_bond[r_bond]
                for s_bond in added_rev:
                    del rev_bond[s_bond]
            # Also try leaving this rule monomer unassigned (only if we
            # haven't found any perfect assignment yet; once we know a
            # perfect assignment exists, skip imperfect branches).
            if not all_assignments:
                _try_complete(ri + 1, n_assigned)

        _try_complete(0, 0)

        if not all_assignments:
            # Fallback: find the best imperfect assignment (shouldn't
            # normally happen if species_matches_pattern is correct)
            best_assignment = [None] * n_rule
            best_count = [0]
            assignment2 = [None] * n_rule
            used_sp2 = set()
            fwd_bond2 = {}
            rev_bond2 = {}

            def _try_best_effort(ri, n_assigned):
                """Backtracking search for the highest-scoring partial assignment.

                Used as a fallback when no perfect assignment exists. Explores
                all partial assignments and keeps the one that matches the most
                rule monomers.
                """
                if ri == n_rule:
                    if n_assigned > best_count[0]:
                        best_count[0] = n_assigned
                        best_assignment[:] = assignment2[:]
                    return
                rule_mp_idx = rule_mps[ri][0]
                cands = candidates.get(rule_mp_idx, [])
                for _, sp_mp_idx in cands:
                    if sp_mp_idx in used_sp2:
                        continue
                    pairs = _get_bond_pairs(rule_mp_idx, sp_mp_idx)
                    consistent = True
                    for r_bond, s_bond in pairs:
                        if r_bond in fwd_bond2:
                            if fwd_bond2[r_bond] != s_bond:
                                consistent = False
                                break
                        if s_bond in rev_bond2:
                            if rev_bond2[s_bond] != r_bond:
                                consistent = False
                                break
                    if not consistent:
                        continue
                    added_fwd = []
                    added_rev = []
                    for r_bond, s_bond in pairs:
                        if r_bond not in fwd_bond2:
                            fwd_bond2[r_bond] = s_bond
                            added_fwd.append(r_bond)
                        if s_bond not in rev_bond2:
                            rev_bond2[s_bond] = r_bond
                            added_rev.append(s_bond)
                    assignment2[ri] = sp_mp_idx
                    used_sp2.add(sp_mp_idx)
                    _try_best_effort(ri + 1, n_assigned + 1)
                    used_sp2.discard(sp_mp_idx)
                    assignment2[ri] = None
                    for r_bond in added_fwd:
                        del fwd_bond2[r_bond]
                    for s_bond in added_rev:
                        del rev_bond2[s_bond]
                _try_best_effort(ri + 1, n_assigned)

            _try_best_effort(0, 0)
            all_assignments = [best_assignment]

        per_cp_assignments.append(
            [(cp_idx, rule_mps, sp_cp, assgn) for assgn in all_assignments]
        )

    # Cartesian product over CPs (for multi-reactant rules)
    result = []
    for combo in itertools.product(*per_cp_assignments):
        mapping = {}
        for cp_idx, rule_mps, sp_cp, assgn in combo:
            for ri, (rule_mp_idx, _) in enumerate(rule_mps):
                if assgn[ri] is not None:
                    mapping[(cp_idx, rule_mp_idx)] = sp_cp.monomer_patterns[assgn[ri]]
        result.append(mapping)

    return result


def _mp_site_matches_vf2(sp_val, rule_val):
    """Check if *sp_val* satisfies *rule_val* with VF2-equivalent semantics.

    This replicates the graph-level constraint that VF2 enforces but that
    :func:`_species_matches_rule_site` does not. Specifically, a bare string
    state (e.g. ``s='pY'``) implies the site is also **unbound** (VF2 adds a
    NO_BOND node for it), whereas :func:`_species_matches_rule_site` only
    checks the state string.

    Handles the common site-condition types found in concrete species:

    * ``None`` explicitly unbound
    * ``int`` bonded (any state)
    * ``str`` state only → also unbound (implicit NO_BOND)
    * ``(str, int)`` state + bonded
    * ``(str, None)`` state + explicitly unbound
    * ``(str, ANY)`` state + bonded to anything
    * ``ANY``, ``WILD`` wildcard conditions

    MultiState conditions are not handled here; if the rule or species value
    is a :class:`MultiState`, the function falls back to
    :func:`_species_matches_rule_site`.
    """
    if isinstance(rule_val, MultiState) or isinstance(sp_val, MultiState):
        return _species_matches_rule_site(sp_val, rule_val)
    if rule_val is WILD:
        return True
    if rule_val is ANY:
        return sp_val is not None
    if rule_val is None:
        # Rule requires no bond. Accept both None (free, no state) and a
        # bare string state (free with a state value - still no bond).
        return sp_val is None or isinstance(sp_val, str)
    if isinstance(rule_val, str):
        # Bare string state implies unbound (VF2 adds NO_BOND).
        if isinstance(sp_val, str):
            return sp_val == rule_val
        if isinstance(sp_val, tuple) and len(sp_val) == 2:
            state, bond = sp_val
            return state == rule_val and bond is None
        return False
    if isinstance(rule_val, int):
        # Integer bond → species site must be bonded (any state).
        if isinstance(sp_val, int):
            return True
        if isinstance(sp_val, tuple) and len(sp_val) == 2:
            return isinstance(sp_val[1], int)
        return False
    if isinstance(rule_val, tuple) and len(rule_val) == 2:
        state, bond = rule_val
        if isinstance(sp_val, tuple) and len(sp_val) == 2:
            sp_state, sp_bond = sp_val
            if sp_state != state:
                return False
            if bond is None:
                return sp_bond is None
            if bond is ANY:
                return sp_bond is not None
            if bond is WILD:
                return True
            return isinstance(sp_bond, int)  # integer bond → bonded
        if isinstance(sp_val, str):
            return sp_val == state and bond is None
        return False
    return sp_val == rule_val


def _sp_contains_mp(sp, rule_mp):
    """Return True if *sp* contains a monomer matching *rule_mp*.

    Fast-path equivalent of ``match_complex_pattern(cp, sp, exact=False)``
    when *cp* has exactly one :class:`MonomerPattern`. Bypasses NetworkX
    VF2 graph isomorphism entirely.

    Only valid when the CP has no complex-level compartment
    (``cp.compartment is None``). Monomer-level compartments are checked
    directly.

    Parameters
    ----------
    sp : ComplexPattern
        A candidate species (may have any number of monomers).
    rule_mp : MonomerPattern
        The single monomer pattern from a unimolecular rule CP.

    Returns
    -------
    bool
    """
    rule_mon = rule_mp.monomer
    rule_sc = rule_mp.site_conditions
    rule_cpt = rule_mp.compartment
    for mp in sp.monomer_patterns:
        if mp.monomer is not rule_mon:
            continue
        if rule_cpt is not None and mp.compartment is not rule_cpt:
            continue
        for site, rule_val in rule_sc.items():
            if not _mp_site_matches_vf2(mp.site_conditions.get(site), rule_val):
                break
        else:
            return True
    return False


def _species_matches_rule_site(sp_val, rule_val):
    """Check if a species site value satisfies a rule site condition.

    Handles all site-condition types (None, int, str, tuple, ANY, WILD,
    MultiState). For MultiState values, symmetry across slots is taken
    into account via :func:`_multistate_slots_match`.

    Parameters
    ----------
    sp_val : object
        The site value from a concrete species MonomerPattern.
    rule_val : object
        The site condition from a rule MonomerPattern.

    Returns
    -------
    bool
        True if *sp_val* satisfies *rule_val*.
    """
    if rule_val is ANY:
        if isinstance(sp_val, MultiState):
            return any(slot is not None for slot in sp_val)
        return sp_val is not None
    if rule_val is WILD:
        return True
    if rule_val is None:
        # Rule requires no bond (site is free). The species may have a
        # state on this site (str) without a bond - that is also "free".
        if isinstance(sp_val, MultiState):
            return any(slot is None or isinstance(slot, str) for slot in sp_val)
        return sp_val is None or isinstance(sp_val, str)
    elif isinstance(rule_val, MultiState):
        # Rule specifies a MultiState (duplicate site) - species must also
        # carry a MultiState with the same number of slots. Because
        # duplicate sites are symmetric, we try all permutations of the
        # species slots against the rule slots.
        if not isinstance(sp_val, MultiState):
            return False
        rule_slots = list(rule_val)
        sp_slots = list(sp_val)
        if len(rule_slots) != len(sp_slots):
            return False
        return _multistate_slots_match(sp_slots, rule_slots)
    elif isinstance(sp_val, MultiState):
        # Non-MultiState rule condition matching a MultiState species value.
        # Matches if at least one slot satisfies the rule condition.
        return any(_species_matches_rule_site(slot, rule_val) for slot in sp_val)
    elif isinstance(rule_val, str):
        # Rule requires a specific state (no bond)
        if isinstance(sp_val, str):
            return sp_val == rule_val
        elif isinstance(sp_val, tuple):
            return sp_val[0] == rule_val
        return False
    elif isinstance(rule_val, int):
        # Rule specifies a bond number - species must have a bond
        if isinstance(sp_val, int):
            return True
        elif isinstance(sp_val, tuple) and len(sp_val) == 2:
            return isinstance(sp_val[1], int)
        return False
    elif isinstance(rule_val, tuple) and len(rule_val) == 2:
        # Rule specifies (state, bond)
        state, bond = rule_val
        if isinstance(sp_val, tuple) and len(sp_val) == 2:
            if isinstance(bond, int):
                return sp_val[0] == state and isinstance(sp_val[1], int)
            if bond is ANY:
                return sp_val[0] == state and sp_val[1] is not None
            if bond is WILD:
                return sp_val[0] == state
            return sp_val[0] == state
        elif isinstance(sp_val, str):
            return sp_val == state and bond is None
        return False
    return sp_val == rule_val


def _multistate_slots_match(sp_slots, rule_slots):
    """Check if species MultiState slots match rule slots under any permutation.

    Duplicate sites in BNGL are symmetric, so we must try all assignments
    of species slots to rule slots. Uses simple recursive backtracking
    (slot counts are small - typically 2 or 3).

    Parameters
    ----------
    sp_slots : list
        Species MultiState slot values.
    rule_slots : list
        Rule MultiState slot values (same length as ``sp_slots``).

    Returns
    -------
    bool
        True if some permutation of *sp_slots* satisfies *rule_slots*.
    """
    n = len(sp_slots)
    used = [False] * n

    def _check_slot_assignment(ri):
        """Return True if rule slots ri..n-1 can all be assigned to unused species slots."""
        if ri == n:
            return True
        for si in range(n):
            if used[si]:
                continue
            if _species_matches_rule_site(sp_slots[si], rule_slots[ri]):
                used[si] = True
                if _check_slot_assignment(ri + 1):
                    return True
                used[si] = False
        return False

    return _check_slot_assignment(0)


def _build_src_bond_map(src_cps, species_monomer_map):
    """Build a mapping from rule source bond numbers to species bond numbers.

    Iterates over all rule-source monomers that have been matched to
    species monomers (via ``species_monomer_map``) and compares their
    non-MultiState bond-carrying site values to establish a correspondence
    between the bond numbering used in the rule's source pattern and the
    bond numbering in the actual species.

    MultiState sites are deliberately skipped here. They are the sites
    whose permutation we are trying to resolve, so using them to build
    the map would be circular.

    Parameters
    ----------
    src_cps : list of ComplexPattern
        The rule's source (reactant-side) ComplexPatterns.
    species_monomer_map : dict
        Mapping from ``(rule_cp_idx, rule_mp_idx)`` to the
        corresponding species MonomerPattern.

    Returns
    -------
    dict
        Mapping of rule bond number (int) to species bond number (int).
    """
    bond_map = {}
    for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
        rule_mp = src_cps[rp_ci].monomer_patterns[rp_mi]
        for site, rule_val in rule_mp.site_conditions.items():
            sp_val = sp_mp.site_conditions.get(site)
            # Skip MultiState - those are exactly the sites whose
            # permutation we need the bond map to resolve.
            if isinstance(rule_val, MultiState) or isinstance(sp_val, MultiState):
                continue
            rule_bonds = _extract_bonds_from_val(rule_val)
            sp_bonds = _extract_bonds_from_val(sp_val)
            if len(rule_bonds) == 1 and len(sp_bonds) == 1:
                r_bond = rule_bonds[0]
                s_bond = sp_bonds[0]
                if r_bond not in bond_map:
                    bond_map[r_bond] = s_bond
    return bond_map


def _find_multistate_permutation(sp_slots, rule_slots, bond_map=None):
    """Find the permutation mapping rule slot indices to species slot indices.

    Parameters
    ----------
    sp_slots : list
        Species MultiState slot values.
    rule_slots : list
        Rule MultiState slot values.
    bond_map : dict, optional
        Mapping from rule bond numbers to species bond numbers. When
        provided, a rule slot with bond ``r`` will only match a species
        slot whose bond equals ``bond_map[r]``. This prevents incorrect
        permutations when multiple slots carry different bonds.

    Returns
    -------
    list or None
        A list ``perm`` where ``perm[rule_idx] = species_idx``, or
        ``None`` if no valid assignment exists.
    """
    perms = _find_all_multistate_permutations(sp_slots, rule_slots, bond_map)
    return perms[0] if perms else None


def _find_all_multistate_permutations(sp_slots, rule_slots, bond_map=None):
    """Find all permutations mapping rule slot indices to species slot indices.

    Returns a list of permutation lists, where each ``perm[rule_idx] =
    species_idx``. Each valid permutation may produce a distinct product
    species when the rule is applied (e.g. unbinding different bonds in a
    symmetric MultiState site).

    Parameters
    ----------
    sp_slots : list
        Species MultiState slot values.
    rule_slots : list
        Rule MultiState slot values.
    bond_map : dict, optional
        Mapping from rule bond numbers to species bond numbers.
    """
    n = len(sp_slots)
    all_perms = []
    perm = [None] * n
    used = [False] * n

    def _slot_compatible(sp_val, rule_val):
        """Return True if sp_val satisfies rule_val, respecting bond_map constraints."""
        if not _species_matches_rule_site(sp_val, rule_val):
            return False
        if bond_map is None:
            return True
        rule_bond = _extract_bond(rule_val)
        if rule_bond is not None and rule_bond in bond_map:
            sp_bond = _extract_bond(sp_val)
            return sp_bond == bond_map[rule_bond]
        return True

    def _enumerate_slot_assignments(ri):
        """Enumerate all valid assignments of rule slots to species slots."""
        if ri == n:
            all_perms.append(perm[:])
            return
        for si in range(n):
            if used[si]:
                continue
            if _slot_compatible(sp_slots[si], rule_slots[ri]):
                perm[ri] = si
                used[si] = True
                _enumerate_slot_assignments(ri + 1)
                used[si] = False
                perm[ri] = None

    _enumerate_slot_assignments(0)
    return all_perms


def _resolve_multistate_slot(slot, existing, product_bond_map):
    """Resolve a single MultiState product slot against an existing species slot.

    Parameters
    ----------
    slot : site condition value
        The product pattern's slot value (may contain ANY/WILD/bond ints).
    existing : site condition value
        The species' current value for this slot.
    product_bond_map : dict
        Mapping from product-side bond numbers to new bond numbers.

    Returns
    -------
    The resolved concrete value for this slot.
    """
    if slot is ANY or slot is WILD:
        return existing
    elif isinstance(slot, tuple) and len(slot) == 2:
        state, bond = slot
        if bond is ANY or bond is WILD:
            if isinstance(existing, tuple) and len(existing) == 2:
                return (state, existing[1])
            elif isinstance(existing, int):
                return (state, existing)
            else:
                return state
        elif isinstance(bond, int):
            return (state, product_bond_map[bond])
        else:
            return slot
    elif isinstance(slot, int):
        return product_bond_map[slot]
    else:
        return slot


def _find_all_matched_multistate_slots(existing_slots, reactant_val, src_bond_map):
    """Find all indices of MultiState slots matched by a non-MultiState rule site.

    When a rule uses a non-MultiState site condition (e.g. ``None``, ``ANY``,
    or a bond int) to match a species whose corresponding site is a MultiState,
    this function determines which slots could have been matched. Each valid
    slot produces a distinct product (analogous to distinct monomer mappings).

    Parameters
    ----------
    existing_slots : list
        The species MultiState slot values.
    reactant_val : site condition value
        The rule reactant's site value (non-MultiState).
    src_bond_map : dict
        Rule bond → species bond mapping.

    Returns
    -------
    list of int
        The indices of all matching slots.
    """
    matched = []
    for i, sp_val in enumerate(existing_slots):
        if _species_matches_rule_site(sp_val, reactant_val):
            # If there's a bond constraint, check it
            rule_bond = _extract_bond(reactant_val)
            if rule_bond is not None and rule_bond in src_bond_map:
                sp_bond = _extract_bond(sp_val)
                if sp_bond != src_bond_map[rule_bond]:
                    continue
            matched.append(i)
    if not matched:
        # Fallback: return first slot (should not happen for valid matches)
        matched.append(0)
    return matched


def _resolve_plain_to_slot(val, product_bond_map):
    """Resolve a non-MultiState product site value for insertion into a MultiState slot.

    Parameters
    ----------
    val : site condition value
        The product pattern's site value (non-MultiState, non-ANY/WILD).
    product_bond_map : dict
        Mapping from product-side bond numbers to new bond numbers.

    Returns
    -------
    The resolved concrete value.
    """
    if isinstance(val, tuple) and len(val) == 2:
        state, bond = val
        if isinstance(bond, int):
            return (state, product_bond_map.get(bond, bond))
        return val
    elif isinstance(val, int):
        return product_bond_map.get(val, val)
    else:
        return val


def _get_extra_monomers(
    species_combo,
    species_monomer_map,
    tgt_cp_idx,
    product_to_source_species,
    species_idx_to_tgt_cp,
):
    """Get monomers from species not mentioned in the rule pattern.

    When a species has monomers bonded to it that the rule does not
    mention, those monomers need to be carried over to the product
    species - but only if they are bonded (directly or transitively)
    to a monomer that is being carried to this specific product CP.

    Parameters
    ----------
    species_combo : list of ComplexPattern
        The actual species (after bond conflict resolution).
    species_monomer_map : dict
        Mapping from ``(rule_cp_idx, rule_mp_idx)`` to species
        MonomerPattern.
    tgt_cp_idx : int
        Index of the product ComplexPattern we are building.
    product_to_source_species : dict
        Maps product CP index to set of source species CP indices.
    species_idx_to_tgt_cp : dict
        Maps ``(src_cp_idx, species_mp_idx)`` to the target CP index
        that monomer is assigned to.

    Returns
    -------
    list of MonomerPattern
        Extra monomers to append to the product species.
    """
    extra = []

    source_cp_idxs = product_to_source_species.get(tgt_cp_idx, set())

    for src_cp_idx in source_cp_idxs:
        if src_cp_idx >= len(species_combo):
            continue
        sp_cp = species_combo[src_cp_idx]

        # Build a bond graph for this species: map each bond number to
        # the set of monomer indices that share it
        bond_graph = collections.defaultdict(set)
        for mp_idx, mp in enumerate(sp_cp.monomer_patterns):
            for site, val in mp.site_conditions.items():
                for bond in _extract_bonds_from_val(val):
                    bond_graph[bond].add(mp_idx)

        # Build adjacency from the bond graph
        adjacency = collections.defaultdict(set)
        for bond_num, mp_idxs in bond_graph.items():
            mp_list = list(mp_idxs)
            for i in range(len(mp_list)):
                for j in range(i + 1, len(mp_list)):
                    adjacency[mp_list[i]].add(mp_list[j])
                    adjacency[mp_list[j]].add(mp_list[i])

        # Identify which species monomer indices are claimed by the rule
        claimed_indices = set()
        for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
            if rp_ci == src_cp_idx:
                for idx, mp in enumerate(sp_cp.monomer_patterns):
                    if mp is sp_mp and idx not in claimed_indices:
                        claimed_indices.add(idx)
                        break

        # Identify the "seed" indices for THIS target CP: claimed
        # monomers that map to this product CP
        seed_indices = set()
        for (rp_ci, rp_mi), sp_mp in species_monomer_map.items():
            if rp_ci != src_cp_idx:
                continue
            sp_mp_idx = None
            for idx, mp in enumerate(sp_cp.monomer_patterns):
                if mp is sp_mp:
                    sp_mp_idx = idx
                    break
            if sp_mp_idx is None:
                continue
            tgt = species_idx_to_tgt_cp.get((src_cp_idx, sp_mp_idx))
            if tgt == tgt_cp_idx:
                seed_indices.add(sp_mp_idx)

        # DFS from seed indices through unclaimed monomers.
        visited = set()
        queue = list(seed_indices)
        while queue:
            current = queue.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbour in adjacency.get(current, set()):
                if neighbour not in visited and neighbour not in claimed_indices:
                    queue.append(neighbour)

        for idx, mp in enumerate(sp_cp.monomer_patterns):
            if idx not in claimed_indices and idx in visited:
                extra.append(mp)

    return extra


class NetworkGenerator:
    """Reaction network generator for PySB models.

    Generates a reaction network by iteratively applying model rules to
    an expanding set of species, starting from the initial conditions.
    This replaces the need for external tools like BioNetGen for network
    generation.

    Parameters
    ----------
    model : Model
        A PySB model.

    Attributes
    ----------
    species : list of ComplexPattern
        The generated species list.
    reactions : OrderedDict
        Generated unidirectional reactions, keyed by
        ``(reactant_ids, product_ids)``.
    reactions_bidirectional : OrderedDict
        Generated bidirectional reactions.

    Examples
    --------
    >>> from pysb.examples import robertson
    >>> ng = NetworkGenerator(robertson.model)
    >>> ng.generate_network()
    >>> len(ng.species)
    3
    """

    def __init__(self, model):
        """Initialise the generator.

        Parameters
        ----------
        model : Model
            The PySB model whose rules will be expanded.
        """
        self.model = model
        self.reactions = collections.OrderedDict()
        self.reactions_bidirectional = collections.OrderedDict()
        self.species_pm = None
        self._max_stoich = None
        self._stoich_pruned = 0
        # Canonical-key → species index lookup; populated in generate_network.
        self._species_by_key = {}
        # Per-(rule_name, dir, cp_idx) → set of matching species indices.
        # Maintained incrementally: only new species are checked each iteration.
        self._match_cache = {}
        # id()-keyed caches; instance-scoped so that memory-address reuse across
        # different models in the same process cannot cause stale hits.
        self._sorted_sites_cache: dict = {}
        self._cp_bond_count_cache: dict = {}
        self._sp_bond_count_cache: dict = {}
        self._cp_state_cache: dict = {}
        self._sp_state_cache: dict = {}

    @property
    def species(self):
        """The current list of species."""
        return [] if self.species_pm is None else self.species_pm.species

    def _update_match_cache(self, rules, new_sp_ids, start_time=None, timeout=None):
        """Extend the match cache for a set of newly added species.

        **Side effect**: mutates ``self._match_cache`` in-place, adding
        newly matched species indices to each per-(rule, dir, cp_idx) set.

        For each (rule, direction, cp_idx) key, check each new species against
        the corresponding rule ComplexPattern. Uses :func:`_sp_could_match_cp`
        as a fast pre-filter before the full subgraph-isomorphism test.

        Parameters
        ----------
        rules : list of Rule
            Non-synthesis rules whose patterns are checked.
        new_sp_ids : iterable of int
            Indices of species that have just been added and have not yet been
            checked against the match cache.
        start_time : float or None
            ``time.monotonic()`` value from the start of network generation.
            Required when ``timeout`` is set.
        timeout : float or None
            If set, raises ``TimeoutError`` when elapsed > timeout.
        """
        for rule in rules:
            dirs = [("fwd", rule.reactant_pattern)]
            if rule.is_reversible:
                dirs.append(("rev", rule.product_pattern))
            for dir_name, pat in dirs:
                for cp_idx, cp in enumerate(pat.complex_patterns):
                    if timeout is not None:
                        elapsed = time.monotonic() - start_time
                        if elapsed > timeout:
                            raise TimeoutError(
                                f"Network generation timed out after "
                                f"{elapsed:.1f}s "
                                f"({len(self.species)} species, "
                                f"{len(self.reactions)} reactions)"
                            )
                    key = (rule.name, dir_name, cp_idx)
                    cache_set = self._match_cache.setdefault(key, set())
                    # Fast path for unimolecular CPs: bypass NetworkX VF2
                    # entirely and check monomer-level site conditions directly.
                    # Only valid when the CP has no complex-level compartment
                    # (monomer-level compartment is checked inside _sp_contains_mp).
                    uni_fast = len(cp.monomer_patterns) == 1 and cp.compartment is None
                    uni_mp = cp.monomer_patterns[0] if uni_fast else None
                    for check_count, sp_idx in enumerate(new_sp_ids):
                        # Check timeout every 16 species (0xF mask) to avoid
                        # calling time.monotonic() on every single lookup.
                        if timeout is not None and (check_count & 0xF) == 0:
                            elapsed = time.monotonic() - start_time
                            if elapsed > timeout:
                                raise TimeoutError(
                                    f"Network generation timed out after "
                                    f"{elapsed:.1f}s "
                                    f"({len(self.species)} species, "
                                    f"{len(self.reactions)} reactions)"
                                )
                        sp = self.species_pm.species[sp_idx]
                        if not _sp_could_match_cp(
                            sp,
                            cp,
                            self._sp_bond_count_cache,
                            self._cp_bond_count_cache,
                            self._sp_state_cache,
                            self._cp_state_cache,
                        ):
                            continue
                        if uni_fast:
                            if _sp_contains_mp(sp, uni_mp):
                                cache_set.add(sp_idx)
                        elif match_complex_pattern(cp, sp, exact=False):
                            cache_set.add(sp_idx)

    def generate_network(
        self, max_iterations=500, max_stoich=None, timeout=None, populate=True
    ):
        """Generate the reaction network.

        Iteratively applies model rules to the current species list,
        generating new species until a fixed point is reached or an
        early-stopping criterion is met.

        By default (``populate=True``) the model's ``species``,
        ``reactions``, ``reactions_bidirectional`` and observable fields
        are written back to the model once generation completes, matching
        the behaviour of :func:`pysb.bng.generate_equations`. Pass
        ``populate=False`` to skip this step and inspect
        :attr:`species`/:attr:`reactions` before committing.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of rule-application iterations (the BNG
            ``max_iter`` option). Default 500. A
            :class:`UserWarning` is emitted if this limit is reached
            before the network converges.
        max_stoich : dict, optional
            Maximum stoichiometry for each monomer type (the BNG
            ``max_stoich`` option). Keys are monomer names (str),
            values are maximum copy counts (int). Any candidate
            product species containing more than the specified number
            of copies of a monomer is discarded. A
            :class:`UserWarning` is emitted if any species are
            discarded. Default None (no stoichiometry limit).
        timeout : float, optional
            Maximum wall-clock time in seconds. If exceeded, a
            ``TimeoutError`` is raised. Default None (no limit).
        populate : bool, optional
            If ``True`` (default), call :meth:`populate_model` after
            generation to write results back to the model.

        Raises
        ------
        NotImplementedError
            If the model uses energy rules, energy patterns, or local
            functions (``Tag``/``@``-annotated rule patterns).
        TimeoutError
            If ``timeout`` is set and the generation exceeds it.

        Warns
        -----
        UserWarning
            If ``max_iterations`` is reached before convergence.
        UserWarning
            If ``max_stoich`` causes species to be discarded.
        """
        if self.model.uses_energy:
            raise NotImplementedError(
                "NetworkGenerator does not support energy rules or "
                "EnergyPattern components. Use pysb.bng.generate_equations() "
                "for energy-based models."
            )

        # Detect local-function (Tag @) usage: any rule whose reactant or
        # product pattern contains a tagged MonomerPattern or ComplexPattern.
        # Local functions require per-species rate expansion which is not yet
        # implemented. Silently ignoring tags would produce wrong reaction
        # rates, so raise immediately.
        for rule in self.model.rules:
            for pat in (rule.reactant_pattern, rule.product_pattern):
                if pat is None:
                    continue
                for cp in pat.complex_patterns:
                    if cp._tag is not None:
                        raise NotImplementedError(
                            f"NetworkGenerator does not support local functions "
                            f"(rule '{rule.name}' uses a Tag/@-annotated pattern). "
                            "Use pysb.bng.generate_equations() for models with "
                            "local functions."
                        )
                    for mp in cp.monomer_patterns:
                        if mp._tag is not None:
                            raise NotImplementedError(
                                f"NetworkGenerator does not support local functions "
                                f"(rule '{rule.name}' uses a Tag/@-annotated pattern). "
                                "Use pysb.bng.generate_equations() for models with "
                                "local functions."
                            )

        start_time = time.monotonic()

        self._max_stoich = max_stoich
        self._stoich_pruned = 0

        seed_species = [_normalise_cp(ic.pattern) for ic in self.model.initials]
        self.species_pm = SpeciesPatternMatcher(self.model, seed_species)

        # Build canonical-key → species-index dict for O(1) exact lookups.
        self._species_by_key = {
            _species_canonical_key(sp, self._sorted_sites_cache): idx
            for idx, sp in enumerate(self.species_pm.species)
        }

        logger.debug("Initial species: %s", [str(sp) for sp in self.species_pm.species])

        # Handle synthesis rules (fire once, unconditionally)
        synth_rules = [r for r in self.model.rules if r.is_synth()]
        non_synth_rules = [r for r in self.model.rules if not r.is_synth()]

        for rule in synth_rules:
            self._fire_rule_combo(rule, (), False)

        # Reversible synthesis rules (e.g. None <> A()) have a reverse
        # (degradation) direction whose reactants are the rule's products.
        # The forward direction will be skipped by the n_cps == 0 guard in
        # the main loop, but the reverse needs to enter the match cache so
        # that degradation reactions are generated when A() appears.
        rev_synth_rules = [
            r
            for r in synth_rules
            if r.is_reversible and len(r.product_pattern.complex_patterns) > 0
        ]
        non_synth_rules = non_synth_rules + rev_synth_rules

        # Incremental rule firing:
        # new_sp_ids holds species added since the last pass. In the first
        # pass all existing species (seeds + synthesis products) are "new".
        # Each pass fires only reactant combos that include ≥1 new species.
        # fired_combos prevents re-firing the same (rule, direction, reactants)
        # tuple across passes.
        new_sp_ids = set(range(len(self.species_pm.species)))
        self._match_cache = {}
        self._update_match_cache(
            non_synth_rules, new_sp_ids, start_time=start_time, timeout=timeout
        )
        fired_combos = set()

        iterations = 0

        while new_sp_ids and iterations < max_iterations:
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Network generation timed out after {elapsed:.1f}s "
                        f"({iterations} iterations, "
                        f"{len(self.species)} species, "
                        f"{len(self.reactions)} reactions)"
                    )

            iterations += 1
            prev_n = len(self.species_pm.species)

            logger.debug(
                "Iteration %d: %d new species to process",
                iterations,
                len(new_sp_ids),
            )

            for rule in non_synth_rules:
                dirs = [("fwd", rule.reactant_pattern, False)]
                if rule.is_reversible:
                    dirs.append(("rev", rule.product_pattern, True))

                for dir_name, pat, is_rev in dirs:
                    n_cps = len(pat.complex_patterns)
                    if n_cps == 0:
                        continue

                    all_lists = [
                        self._match_cache.get((rule.name, dir_name, i), set())
                        for i in range(n_cps)
                    ]
                    new_lists = [s & new_sp_ids for s in all_lists]

                    if not any(new_lists):
                        continue

                    sorted_lists = [sorted(s) for s in all_lists]
                    for combo_idxs in itertools.product(*sorted_lists):
                        # Only process combos that include at least one newly
                        # added species, while preserving the historical
                        # traversal order.
                        if not any(idx in new_sp_ids for idx in combo_idxs):
                            continue

                        if timeout is not None:
                            elapsed = time.monotonic() - start_time
                            if elapsed > timeout:
                                raise TimeoutError(
                                    f"Network generation timed out after "
                                    f"{elapsed:.1f}s "
                                    f"({iterations} iterations, "
                                    f"{len(self.species)} species, "
                                    f"{len(self.reactions)} reactions)"
                                )

                        combo_key = (rule.name, is_rev) + combo_idxs
                        if combo_key in fired_combos:
                            continue
                        fired_combos.add(combo_key)

                        reactant_combo = tuple(
                            self.species_pm.species[i] for i in combo_idxs
                        )
                        self._fire_rule_combo(
                            rule,
                            reactant_combo,
                            is_rev,
                            reactant_indices=combo_idxs,
                        )

            # Determine species added during this pass and update the cache
            new_sp_ids = set(range(prev_n, len(self.species_pm.species)))
            if new_sp_ids:
                self._update_match_cache(
                    non_synth_rules, new_sp_ids, start_time=start_time, timeout=timeout
                )

        if new_sp_ids and iterations >= max_iterations:
            warnings.warn(
                f"Network generation reached max_iterations={max_iterations} "
                f"without converging ({len(self.species)} species, "
                f"{len(self.reactions)} reactions). The network may be "
                f"incomplete. Increase max_iterations or use max_stoich to "
                f"bound the network.",
                stacklevel=2,
            )

        if self._stoich_pruned > 0:
            warnings.warn(
                f"max_stoich discarded {self._stoich_pruned} candidate "
                f"species during network generation "
                f"({len(self.species)} species, "
                f"{len(self.reactions)} reactions retained).",
                stacklevel=2,
            )

        elapsed = time.monotonic() - start_time
        logger.info(
            "Network generation complete: %d species, %d reactions "
            "in %d iterations (%.2fs)",
            len(self.species),
            len(self.reactions),
            iterations,
            elapsed,
        )

        if populate:
            self.populate_model()

    def populate_model(self):
        """Populate the model's equation fields from the generated network.

        Mirrors the effect of :func:`pysb.bng.generate_equations`: fills in
        ``model.species``, ``model.reactions``, ``model.reactions_bidirectional``
        and the ``species``/``coefficients`` attributes of each observable.

        Must be called after :meth:`generate_network`.

        .. warning::
            Calls :meth:`~pysb.core.Model.reset_equations` internally,
            which clears ``model.species``, ``model.reactions``,
            ``model.reactions_bidirectional``, and all observable
            ``species``/``coefficients`` fields before repopulating them.

        Raises
        ------
        RuntimeError
            If :meth:`generate_network` has not yet been called.

        Examples
        --------
        >>> from pysb.examples import robertson
        >>> from pysb.netgen import NetworkGenerator
        >>> ng = NetworkGenerator(robertson.model)
        >>> ng.generate_network()
        >>> ng.populate_model()
        >>> len(robertson.model.species)
        3
        """

        if self.species_pm is None:
            raise RuntimeError(
                "generate_network() must be called before populate_model()"
            )

        model = self.model

        # Reset equation fields
        model.reset_equations()
        model._netgen_source = "pysb"

        # Populate species
        model.species = list(self.species)

        # Build rate expressions and populate reactions / reactions_bidirectional
        reaction_cache = {}  # key (reactants, products) -> bidirectional entry

        for (reactant_ids, product_ids), rxn_info in self.reactions.items():
            rule_names = rxn_info["rule"]
            reverse_flags = rxn_info["reverse"]

            # Determine rate parameter/expression for this reaction from the
            # first matching rule. Supports multi-rule reactions by summing.
            combined_rate = sympy.Integer(0)
            # BNG divides by n! for each group of n identical reactant species
            # (statistical / symmetry factor).  Compute 1/∏(count!)  where
            # the product runs over unique species in reactant_ids.
            reactant_counts = collections.Counter(reactant_ids)
            sym_factor = sympy.Rational(
                1, math.prod(math.factorial(c) for c in reactant_counts.values())
            )
            for rule_name, is_rev in zip(rule_names, reverse_flags):
                rule = model.rules[rule_name]
                rate_obj = rule.rate_reverse if is_rev else rule.rate_forward
                # Build the propensity: symmetry_factor * rate_constant * product(s_i)
                # where s_i are the concentrations of each reactant species.
                species_syms = sympy.Mul(
                    *[sympy.Symbol("__s%d" % r) for r in reactant_ids]
                )
                combined_rate += sym_factor * sympy.Mul(sympy.S(rate_obj), species_syms)

            reaction = {
                "reactants": reactant_ids,
                "products": product_ids,
                "rate": combined_rate,
                "rule": tuple(rule_names),
                "reverse": tuple(reverse_flags),
            }
            model._reactions.append(reaction)

            # Build bidirectional entry.
            # Reactions are stored with sorted reactant/product tuples (see
            # _record_reaction), so (reactant_ids, product_ids) and
            # (product_ids, reactant_ids) are the only two orientations to check.
            key = (reactant_ids, product_ids)
            key_rev = (product_ids, reactant_ids)
            if key in reaction_cache:
                bd = reaction_cache[key]
                bd["rate"] += combined_rate
                for rn in rule_names:
                    if rn not in bd["rule"]:
                        bd["rule"] = bd["rule"] + (rn,)
            elif key_rev in reaction_cache:
                bd = reaction_cache[key_rev]
                bd["reversible"] = True
                bd["rate"] -= combined_rate
                for rn in rule_names:
                    if rn not in bd["rule"]:
                        bd["rule"] = bd["rule"] + (rn,)
            else:
                bd = dict(reaction)
                bd["reversible"] = False
                reaction_cache[key] = bd
                model._reactions_bidirectional.append(bd)

        if model._species:
            spm = SpeciesPatternMatcher(model, species=model._species)
            for obs in model.observables:
                for cp in obs.reaction_pattern.complex_patterns:
                    matches = spm.match(cp, index=True, exact=False, counts=True)
                    for sp_idx, count in matches.items():
                        obs.species.append(sp_idx)
                        obs.coefficients.append(count)

    def _lookup_or_add_species(self, sp):
        """Return the species index for ``sp``, adding it if new.

        Uses the canonical key for O(1) exact-identity lookup instead of
        iterating over all known species with graph isomorphism.

        Parameters
        ----------
        sp : ComplexPattern
            A concrete species to look up or register.

        Returns
        -------
        tuple of (int, bool)
            ``(species_index, is_new)`` where ``is_new`` is True if the
            species was not previously known.
        """
        # Normalise complex-level compartment (``@C:`` prefix form) so that
        # the species representation matches BNG's canonical form where each
        # MonomerPattern carries its own compartment.
        sp = _normalise_cp(sp)
        key = _species_canonical_key(sp, self._sorted_sites_cache)
        existing = self._species_by_key.get(key)
        if existing is not None:
            return existing, False
        new_id = len(self.species_pm.species)
        # Add without duplicate check (we know it's new via the key dict)
        self.species_pm.species.append(sp)
        self.species_pm._add_species(new_id, sp)
        self._species_by_key[key] = new_id
        return new_id, True

    def _fire_rule_combo(self, rule, reactant_combo, is_rev, reactant_indices=None):
        """Apply a rule to a reactant combination and record results.

        Parameters
        ----------
        rule : Rule
            The rule to apply.
        reactant_combo : tuple of ComplexPattern
            The reactant species.
        is_rev : bool
            Whether this is the reverse direction.
        reactant_indices : tuple of int or None
            Pre-computed species indices for each reactant. When provided,
            the canonical-key lookup is skipped.

        Returns
        -------
        bool
            True if new species were generated.
        """
        all_product_lists = _apply_rule_to_species(rule, reactant_combo, rev_dir=is_rev)

        if reactant_indices is not None:
            reactants = reactant_indices
        else:
            # Ensure _species_by_key is populated (handles callers that set
            # species_pm directly without going through generate_network).
            if self.species_pm is not None and not self._species_by_key:
                for i, sp in enumerate(self.species_pm.species):
                    k = _species_canonical_key(sp, self._sorted_sites_cache)
                    if k not in self._species_by_key:
                        self._species_by_key[k] = i

            # Determine species IDs for reactants via canonical-key lookup
            reactants = tuple(
                self._species_by_key[
                    _species_canonical_key(reac, self._sorted_sites_cache)
                ]
                for reac in reactant_combo
            )

        new_species_found = False

        for gen_species_list in all_product_lists:
            products = []
            stoich_violated = False
            for gen_species in gen_species_list:
                if self._max_stoich is not None:
                    if _exceeds_max_stoich(gen_species, self._max_stoich):
                        stoich_violated = True
                        break

                sp_id, is_new = self._lookup_or_add_species(gen_species)
                if is_new:
                    logger.debug("New species [%d]: %s", sp_id, gen_species)
                    new_species_found = True
                products.append(sp_id)

            if stoich_violated:
                self._stoich_pruned += 1
                continue

            products = tuple(sorted(products))

            rxn_key = (tuple(sorted(reactants)), products)
            if rxn_key not in self.reactions:
                self.reactions[rxn_key] = {
                    "rule": [rule.name],
                    "reactants": tuple(sorted(reactants)),
                    "products": products,
                    "reverse": [is_rev],
                }
            else:
                existing = self.reactions[rxn_key]
                if rule.name not in existing["rule"]:
                    existing["rule"].append(rule.name)
                    existing["reverse"].append(is_rev)

        return new_species_found

    def check_species_against_bng(self):
        """Compare generated species against BioNetGen output.

        Runs BioNetGen on the model and verifies that the species lists
        match (same species, possibly in different order).

        .. warning::
            Calls :func:`pysb.bng.generate_equations` internally, which
            overwrites ``model.species``, ``model.reactions``, and
            ``model.reactions_bidirectional``.

        Returns
        -------
        list of int
            Correspondence mapping: ``correspondence[i]`` is the BNG
            species index corresponding to netgen species ``i``.

        Raises
        ------
        ValueError
            If species counts differ or a species cannot be matched.
        """
        from pysb.bng import generate_equations

        self.model.reset_equations()
        generate_equations(self.model)

        bng_species_pm = SpeciesPatternMatcher(self.model)
        correspondence = []

        for i, sp in enumerate(self.species):
            matches = bng_species_pm.match(sp, index=True, exact=True)
            if not matches:
                raise ValueError(
                    f"Netgen species [{i}] {sp} not found in BNG "
                    f"species list:\n"
                    + "\n".join(
                        f"  [{j}] {s}" for j, s in enumerate(self.model.species)
                    )
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Netgen species [{i}] {sp} matches multiple BNG species: {matches}"
                )
            if matches[0] in correspondence:
                prev_idx = correspondence.index(matches[0])
                raise ValueError(
                    f"Netgen species [{i}] {sp} and [{prev_idx}] "
                    f"{self.species[prev_idx]} both match BNG species "
                    f"[{matches[0]}] {self.model.species[matches[0]]}"
                )
            correspondence.append(matches[0])

        if len(self.species) != len(self.model.species):
            missing = set(range(len(self.model.species))) - set(correspondence)
            raise ValueError(
                f"Species count mismatch: netgen={len(self.species)}, "
                f"BNG={len(self.model.species)}. "
                f"BNG species not in netgen: "
                + ", ".join(f"[{i}] {self.model.species[i]}" for i in sorted(missing))
            )

        return correspondence

    def check_reactions_against_bng(self, correspondence=None):
        """Compare generated reactions against BioNetGen output.

        Verifies that for every reaction generated by netgen, a
        corresponding reaction exists in BNG output with the same
        rule attribution.

        .. note::
            The check is one-directional: it verifies that every netgen
            reaction appears in BNG, but does **not** verify that every
            BNG reaction appears in netgen. Use
            :meth:`check_species_against_bng` (which checks both
            directions for species) as a complement.

        Parameters
        ----------
        correspondence : list of int, optional
            Species correspondence from ``check_species_against_bng``.
            If None, it will be computed.

        Raises
        ------
        ValueError
            If a reaction has no BNG counterpart or rule sets differ.
        """
        if correspondence is None:
            correspondence = self.check_species_against_bng()

        # Build BNG reaction lookup
        bng_rxns = {}
        for rxn in self.model.reactions:
            key = (tuple(sorted(rxn["reactants"])), tuple(sorted(rxn["products"])))
            if key not in bng_rxns:
                bng_rxns[key] = []
            bng_rxns[key].append(rxn)

        for (reactants, products), rxn_info in self.reactions.items():
            mapped_reactants = tuple(sorted(correspondence[r] for r in reactants))
            mapped_products = tuple(sorted(correspondence[p] for p in products))
            mapped_key = (mapped_reactants, mapped_products)

            if mapped_key not in bng_rxns:
                raise ValueError(
                    f"No BNG reaction for netgen reaction: "
                    f"reactants={reactants} products={products} "
                    f"rule={rxn_info['rule']} "
                    f"(mapped: {mapped_key})"
                )

            # Check rule attribution
            bng_rules = set()
            for bng_rxn in bng_rxns[mapped_key]:
                bng_rules.update(bng_rxn["rule"])
            netgen_rules = set(rxn_info["rule"])

            diff = netgen_rules.symmetric_difference(bng_rules)
            if diff:
                raise ValueError(
                    f"Rule mismatch for reaction "
                    f"reactants={reactants} products={products}: "
                    f"netgen={netgen_rules}, BNG={bng_rules}"
                )
