Native network generator (:py:mod:`pysb.netgen`)
================================================

:mod:`pysb.netgen` is a pure-Python replacement for the external
`BioNetGen <https://bionetgen.org>`_ (BNG) tool. It expands a PySB model's
reaction rules into an explicit list of chemical species and elementary
reactions without spawning a subprocess or requiring BioNetGen to be installed.

Calling :meth:`~pysb.netgen.NetworkGenerator.generate_network` populates the
model's ``species``, ``reactions``, ``reactions_bidirectional``, and observable
``species``/``coefficients`` fields in exactly the same format as
:func:`pysb.bng.generate_equations`.

Quickstart
----------

Basic usage — generate the network for the Robertson model:

.. code-block:: python

    from pysb.examples.robertson import model
    from pysb.netgen import NetworkGenerator

    ng = NetworkGenerator(model)
    ng.generate_network()  # also populates model.species / model.reactions

    for i, sp in enumerate(model.species):
        print(f"  s{i}: {sp}")

    for rxn in model.reactions:
        print(rxn)

Using ``max_stoich`` to cap combinatorial explosion:

.. code-block:: python

    from pysb.examples.bax_pore import model
    from pysb.netgen import NetworkGenerator

    ng = NetworkGenerator(model)
    ng.generate_network(max_stoich={"Bax": 4})

Setting a wall-clock timeout (partial networks are still usable):

.. code-block:: python

    from pysb.netgen import NetworkGenerator

    ng = NetworkGenerator(large_model)
    try:
        ng.generate_network(timeout=30.0)
    except TimeoutError as e:
        print("Generation timed out:", e)
    # ng.species / ng.reactions contain whatever was generated before the timeout

Comparison with :func:`pysb.bng.generate_equations`
----------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - :mod:`pysb.netgen`
     - :mod:`pysb.bng`
   * - External dependency
     - None
     - BioNetGen installation required
   * - Execution model
     - In-process Python
     - Subprocess (writes ``.bngl`` files)
   * - Output format
     - Same ``model.species`` / ``model.reactions`` layout
     - Same ``model.species`` / ``model.reactions`` layout
   * - Synthesis rules
     - Yes
     - Yes
   * - Degradation rules
     - Yes
     - Yes
   * - DeleteMolecules
     - Yes
     - Yes
   * - MatchOnce
     - Yes
     - Yes
   * - Compartments (assignment)
     - Yes
     - Yes
   * - Compartment volume scaling
     - Delegated to ODE/SSA builder (same as BNG)
     - Delegated to ODE/SSA builder
   * - ``max_stoich``
     - Yes
     - Yes
   * - ``max_iterations``
     - Yes (warns on early stop)
     - Yes
   * - ``Expression``/function-based rates
     - Yes (passed through symbolically)
     - Yes
   * - Energy rules
     - Not supported (raises :exc:`NotImplementedError`)
     - Yes
   * - Local functions (``Tag``/``@``)
     - Not supported (raises :exc:`NotImplementedError`)
     - Yes
   * - MultiState sites
     - Yes
     - Yes
   * - ``timeout`` parameter
     - Yes (raises ``TimeoutError``)
     - No

Performance
-----------

Because :mod:`pysb.netgen` runs in-process, it avoids the ~100 ms overhead of
launching a Perl subprocess and file I/O that every BNG call incurs
regardless of model size. Two speedup measures are reported:

**Speedup (wall)**
    End-to-end wall-clock time ratio ``BNG_wall / PySB``. Includes the Perl
    subprocess startup cost and file I/O, which particularly benefits
    smaller models.

**Speedup (CPU)**
    Algorithmic throughput ratio ``BNG_CPU / PySB``, where ``BNG_CPU`` is
    BNG's own self-reported CPU time. This is the more conservative and
    honest comparison for large models where generation itself dominates.

.. list-table:: Representative benchmarks (Python 3.14, BioNetGen 2.9.1, MacBook Air M4, 2025)
   :header-rows: 1
   :widths: 24 6 8 10 13 12 11 10

   * - Model
     - Sp
     - Rx
     - PySB (s)
     - BNG wall (s)
     - BNG CPU (s)
     - Wall ×
     - CPU ×
   * - robertson
     - 3
     - 3
     - <0.001
     - 0.077
     - <0.001
     - ~800×
     - N/A ¹
   * - kinase_cascade
     - 21
     - 30
     - 0.004
     - 0.113
     - 0.020
     - 27×
     - 5×
   * - earm_1_0
     - 58
     - 70
     - 0.011
     - 0.197
     - 0.060
     - 18×
     - 5.5×
   * - bngwiki_egfr_simple
     - 22
     - 86
     - 0.014
     - 0.184
     - 0.040
     - 13×
     - 3×
   * - fceri_ji (FcεRI receptor)
     - 354
     - 3680
     - 0.73
     - 2.35
     - 1.56
     - 3.3×
     - 2.2×

¹ Both runtimes are sub-millisecond; the CPU speedup ratio is not meaningful.

The wall speedup is dominated by subprocess overhead for small models (top
four rows). The CPU speedup ranges from about 2× to 5× across the tested set.

.. note::

    For highly combinatorial models the speedup narrows. In its current
    state, :mod:`pysb.netgen` has not been tuned for the same class of
    combinatorial models that BNG's Perl code handles best (e.g. bivalent
    ligand/receptor systems). For such models, check timings on your own
    hardware before choosing a generator.

To reproduce these numbers or benchmark your own model::

    python benchmarks/netgen_benchmark.py --models robertson kinase_cascade earm_1_0 bngwiki_egfr_simple fceri_ji

Limitations
-----------

Energy rules
    Models that use energy-rule annotations (``Rule(..., energy=True)``) or
    :class:`~pysb.core.EnergyPattern` components are not supported. A
    :exc:`NotImplementedError` is raised immediately. Use
    :func:`pysb.bng.generate_equations` for energy models.

Local functions (``Tag`` / ``@``-annotated patterns)
    BNGL local functions annotate rule patterns with ``@``-tags to give
    individual monomers rate contributions that depend on per-species context
    (e.g. ``A(b!+)@local_obs``). Correctly implementing local functions
    requires expanding each rule into one reaction per species in which the
    tagged pattern appears. This expansion is not yet implemented; attempting
    to generate a network for a model that uses local functions raises
    :exc:`NotImplementedError` immediately. Use
    :func:`pysb.bng.generate_equations` for such models.

Compartment volume scaling
    Rate expressions are built with the raw rate parameter; compartment volume
    factors are not multiplied in. This matches :func:`pysb.bng.generate_equations`,
    which also delegates volume scaling to the downstream ODE/SSA builder.

Automatic (lazy) network generation
------------------------------------

PySB models support *automatic* network generation via the ``auto_netgen`` flag
on :class:`~pysb.core.Model`. When set, the network is generated lazily on
the first access of ``model.species``, ``model.reactions``, or
``model.reactions_bidirectional``, and regenerated automatically whenever rules
or initials are added.

.. code-block:: python

    from pysb import *
    from pysb.examples.robertson import model

    model.reset_equations()
    model.auto_netgen = True

    # Network is generated here, on first access:
    print(len(model.species))    # 3
    print(len(model.reactions))  # 3

    # Adding a rule marks the model dirty:
    # the next access regenerates the network
    k_new = Parameter('k_new', 1.0)
    A, B, _ = model.monomers
    Rule('trivial', A() + B() >> A() + B(), k_new)
    print(len(model.reactions))  # 4 - regenerates transparently

To opt in at model construction time:

.. code-block:: python

    m = Model(auto_netgen=True)
    # … define monomers, rules, initials …
    print(m.species)  # generated on demand

.. note::

    Assigning a new list or component-set to :attr:`~pysb.core.Model.initials`
    or :attr:`~pysb.core.Model.rules` is detected automatically and marks the
    model dirty. However, deep mutations of *existing* :class:`~pysb.core.Rule`
    or :class:`~pysb.core.Initial` objects (e.g.
    ``model.rules['r'].rate_forward = new_param``) are **not** automatically
    detected. Call :meth:`~pysb.core.Model.reset_equations` explicitly to
    force regeneration on the next access in those cases.

Validation against BioNetGen
-----------------------------

Two methods cross-check the generated network against BNG output, which is
useful when debugging a new model or exploring edge cases:

.. code-block:: python

    from pysb.examples.earm_1_0 import model
    from pysb.netgen import NetworkGenerator

    ng = NetworkGenerator(model)
    ng.generate_network()

    correspondence = ng.check_species_against_bng()
    ng.check_reactions_against_bng(correspondence)

:meth:`~pysb.netgen.NetworkGenerator.check_species_against_bng` verifies that
both species lists contain the same set of species (in any order).
:meth:`~pysb.netgen.NetworkGenerator.check_reactions_against_bng` verifies
that every netgen reaction has a BNG counterpart with the same rule attribution.

.. warning::

    Both methods require a working BioNetGen installation. They overwrite
    ``model.species``, ``model.reactions``, and ``model.reactions_bidirectional``
    with BNG output as a side-effect.

Module reference
----------------

.. automodule:: pysb.netgen
    :members:
