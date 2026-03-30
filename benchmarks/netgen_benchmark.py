"""Benchmark PySB native network generator vs BioNetGen.

Measures wall-clock time for network generation of all PySB example models
and BNG Validate models (the subset known to import cleanly via pysb.importers).

Usage
-----
Run directly::

    python benchmarks/netgen_benchmark.py

or with ``--models`` to restrict to specific models::

    python benchmarks/netgen_benchmark.py --models robertson michment

Output
------
Prints a Markdown table with columns:

* Model              – model name
* N_species          – number of generated species
* N_reactions        – number of generated reactions
* PySB (s)           – wall-clock time for NetworkGenerator.generate_network()
* BNG wall (s)       – wall-clock time for pysb.bng.generate_equations() (subprocess)
* BNG CPU (s)        – BNG's own reported CPU time parsed from its stdout
* Speedup (wall)     – BNG_wall / PySB  (includes Perl/subprocess startup overhead)
* Speedup (CPU)      – BNG_CPU / PySB   (pure algorithmic throughput, no startup cost)

The two speedup columns tell different stories:

* **Speedup (wall)** captures the end-to-end benefit for a single generate-network
  call inside a Python process.  Even for tiny models it is large (≫10×) because
  BNG requires launching a Perl subprocess (~80 ms overhead on typical hardware).

* **Speedup (CPU)** isolates algorithmic performance by comparing PySB's wall time
  against BNG's self-reported CPU time.  This is the more honest comparison for
  large models where network generation itself dominates.
"""

import argparse
import importlib
import os
import re
import subprocess
import sys
import time

# Ensure pysb is importable when run as a script from any working directory
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

#: PySB example models known to pass network generation
PYSB_MODELS = [
    "bax_pore",
    "bax_pore_sequential",
    "bngwiki_egfr_simple",
    "bngwiki_enzymatic_cycle_mm",
    "bngwiki_simple",
    "earm_1_0",
    "earm_1_3",
    "explicit",
    "expression_observables",
    "fixed_initial",
    "fricker_2010_apoptosis",
    "hello_pysb",
    "kinase_cascade",
    "michment",
    "robertson",
    "schloegl",
    "synth_deg",
    "time",
    "tutorial_a",
    "tyson_oscillator",
]

#: BNG Validate models known to import cleanly via pysb.importers
BNG_VALIDATE_MODELS = [
    "CaOscillate_Func",
    "continue",
    "deleteMolecules",
    "empty_compartments_block",
    "gene_expr",
    "gene_expr_func",
    "gene_expr_simple",
    "isomerization",
    "michment",
    "motor",
    "simple_system",
    "test_fixed",
    "test_synthesis_complex",
    "test_synthesis_simple",
    # Larger models — meaningful network-generation workload for PySB
    "test_partial_dynamical_scaling",  # 34 sp, 63 rx
    "toy-jim",  # 51 sp, 221 rx
    "tlmr",
    "fceri_ji",
    "Repressilator",
]

#: BNG Validate models that need generate_network constraints
#: Each entry is (name, netgen_kwargs).
BNG_VALIDATE_CONSTRAINED_MODELS = [
    ("tlbr", {"max_iterations": 3}),  # 19 sp, 29 rx
    ("blbr", {"max_stoich": {"R": 5, "L": 5}}),  # 20 sp, 92 rx
]

#: BNG Models2 models known to import cleanly via pysb.importers
#: These live in the Models2/ sibling of the Validate/ directory.
BNG_MODELS2_MODELS = [
    "egfr_net_red",  # reduced EGFR network: 40 sp, 123 rx
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bng_validate_directory():
    """Return the path to BNG's Validate directory."""
    try:
        import pysb.pathfinder as pf

        bng_exec = os.path.realpath(pf.get_path("bng"))
        if bng_exec.endswith(".bat"):
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            return os.path.join(conda_prefix, "share", "bionetgen", "Validate")
        return os.path.join(os.path.dirname(bng_exec), "Validate")
    except Exception:
        return None


def _bng_models2_directory():
    """Return the path to BNG's Models2 directory, or None if not found."""
    validate = _bng_validate_directory()
    if validate is None:
        return None
    models2 = os.path.join(os.path.dirname(validate), "Models2")
    return models2 if os.path.isdir(models2) else None


def _load_pysb_example(name):
    """Import and return a freshly reset PySB example model."""
    import pysb.core

    # pysb/examples/explicit.py sets SelfExporter.do_export = False without
    # restoring it, which breaks all subsequent model imports that rely on
    # SelfExporter injecting component names into the module namespace.
    saved_do_export = pysb.core.SelfExporter.do_export
    try:
        mod = importlib.import_module(f"pysb.examples.{name}")
        importlib.reload(mod)
        model = mod.model
        model.reset_equations()
        return model
    finally:
        pysb.core.SelfExporter.do_export = saved_do_export


def _load_bngl_model(bngl_path):
    """Import a BNGL file and return a freshly reset PySB model."""
    from pysb.importers.bngl import model_from_bngl

    model = model_from_bngl(bngl_path)
    model.reset_equations()
    return model


def _time_pysb_netgen(model, netgen_kwargs=None):
    """Return (n_species, n_reactions, elapsed_s) for PySB NetworkGenerator."""
    from pysb.netgen import NetworkGenerator

    model.reset_equations()
    ng = NetworkGenerator(model)
    t0 = time.monotonic()
    ng.generate_network(populate=False, **(netgen_kwargs or {}))
    elapsed = time.monotonic() - t0
    return len(ng.species), len(ng.reactions), elapsed


def _format_bng_action(action_kw):
    """Format a generate_network action line with proper BNG syntax.

    Handles max_stoich dict values which need Perl hash syntax.
    """
    parts = []
    for k, v in action_kw.items():
        if isinstance(v, dict):
            # Perl hash syntax: {'key1'=>val1,'key2'=>val2}
            inner = ",".join(f"'{mk}'=>{mv}" for mk, mv in v.items())
            parts.append(f"{k}=>{{{inner}}}")
        elif isinstance(v, bool):
            parts.append(f"{k}=>{1 if v else 0}")
        elif isinstance(v, str):
            parts.append(f'{k}=>"{v}"')
        else:
            parts.append(f"{k}=>{v}")
    return f"\tgenerate_network({{{','.join(parts)}}})\n"


def _time_bng(model, bng_action_kwargs=None):
    """Return (n_species, n_reactions, elapsed_wall_s, bng_cpu_s) for BNG."""
    import pysb.bng as bng_mod
    import pysb.pathfinder as pf

    model.reset_equations()

    action_kw = {"overwrite": True, "verbose": True}
    if bng_action_kwargs:
        action_kw.update(bng_action_kwargs)

    # Use BngFileInterface to generate the BNGL file, then run BNG as a
    # subprocess ourselves so we can capture its stdout.  pysb.bng's own
    # execute() discards stdout (logs it at DEBUG), so we cannot retrieve
    # the "CPU TIME: generate_network" line through generate_network().
    bng_stdout = ""
    t0 = time.monotonic()
    try:
        with bng_mod.BngFileInterface(model, verbose=False, cleanup=True) as bngfile:
            # Write the BNGL content and action manually to support
            # dict-valued kwargs (like max_stoich) which need Perl hash
            # syntax that BngFileInterface._format_action_args can't handle.
            action_line = _format_bng_action(action_kw)
            with open(bngfile.bng_filename, "w") as fh:
                fh.write(bngfile.generator.get_content())
                fh.write("begin actions\n")
                fh.write(action_line)
                fh.write("end actions\n")

            bng_exec = pf.get_path("bng")
            args = [bng_exec, bngfile.bng_filename]
            if not bng_exec.endswith(".bat"):
                args = ["perl"] + args

            proc = subprocess.run(
                args,
                cwd=bngfile.base_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
            )
            bng_stdout = proc.stdout.decode("utf-8", errors="replace")

            # Parse the generated net file back into the model so that
            # model.species / model.reactions are populated.
            if proc.returncode == 0:
                bng_mod._parse_netfile(model, open(bngfile.net_filename))
    except Exception:
        pass
    elapsed_wall = time.monotonic() - t0

    # Parse BNG's own CPU time from its stdout.
    bng_cpu = None
    cpu_match = re.search(r"CPU TIME: generate_network\s+([\d.]+)\s+s", bng_stdout)
    if cpu_match:
        bng_cpu = float(cpu_match.group(1))

    n_species = len(model.species) if model.species else None
    n_reactions = len(model.reactions) if model.reactions else None
    return n_species, n_reactions, elapsed_wall, bng_cpu


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_benchmark(label, model_loader, verbose=False, netgen_kwargs=None):
    """Run both generators on a single model, return result dict."""
    result = {
        "model": label,
        "n_species": None,
        "n_reactions": None,
        "pysb_s": None,
        "bng_wall_s": None,
        "bng_cpu_s": None,
        "error": None,
    }
    try:
        model = model_loader()
        n_sp, n_rx, pysb_t = _time_pysb_netgen(model, netgen_kwargs)
        result["n_species"] = n_sp
        result["n_reactions"] = n_rx
        result["pysb_s"] = pysb_t
        if verbose:
            print(
                f"  PySB: {n_sp} species, {n_rx} reactions, {pysb_t:.3f}s", flush=True
            )
    except Exception as exc:
        result["error"] = f"PySB: {exc}"
        if verbose:
            print(f"  PySB ERROR: {exc}", flush=True)
        return result

    # Translate netgen_kwargs to BNG action kwargs
    bng_action_kw = None
    if netgen_kwargs:
        bng_action_kw = {}
        if "max_iterations" in netgen_kwargs:
            bng_action_kw["max_iter"] = netgen_kwargs["max_iterations"]
        if "max_stoich" in netgen_kwargs:
            bng_action_kw["max_stoich"] = netgen_kwargs["max_stoich"]

    try:
        model2 = model_loader()
        _, _, bng_wall, bng_cpu = _time_bng(model2, bng_action_kw)
        result["bng_wall_s"] = bng_wall
        result["bng_cpu_s"] = bng_cpu
        if verbose:
            cpu_str = f"{bng_cpu:.3f}s CPU" if bng_cpu is not None else "CPU N/A"
            print(f"  BNG:  {bng_wall:.3f}s wall, {cpu_str}", flush=True)
    except Exception as exc:
        result["error"] = (result.get("error") or "") + f"; BNG: {exc}"
        if verbose:
            print(f"  BNG ERROR: {exc}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Markdown table output
# ---------------------------------------------------------------------------


def _fmt(val, fmt=".3f", missing="N/A"):
    if val is None:
        return missing
    return format(val, fmt)


def _speedup(bng_t, pysb):
    """Format a speedup ratio, or 'N/A' if either value is missing or zero.

    Returns N/A when bng_t is zero because BNG reported no measurable CPU time
    (both sides are sub-millisecond and the ratio is not meaningful).
    """
    if bng_t is None or bng_t == 0 or pysb is None or pysb == 0:
        return "N/A"
    return f"{bng_t / pysb:.1f}x"


def _print_table(results):
    # Sort by descending reactions, then descending species as tiebreaker
    results = sorted(
        results,
        key=lambda r: (-(r["n_reactions"] or 0), -(r["n_species"] or 0)),
    )
    header = (
        f"| {'Model':<35} | {'N_sp':>6} | {'N_rx':>6} "
        f"| {'PySB (s)':>9} | {'BNG wall (s)':>12} "
        f"| {'BNG CPU (s)':>11} | {'Spdup(wall)':>11} | {'Spdup(CPU)':>10} |"
    )
    sep = (
        f"| {'-' * 35} | {'-' * 6} | {'-' * 6} "
        f"| {'-' * 9} | {'-' * 12} "
        f"| {'-' * 11} | {'-' * 11} | {'-' * 10} |"
    )
    print()
    print(header)
    print(sep)
    for r in results:
        err = f" ⚠ {r['error']}" if r["error"] else ""
        print(
            f"| {r['model']:<35} | {_fmt(r['n_species'], 'd'):>6} "
            f"| {_fmt(r['n_reactions'], 'd'):>6} "
            f"| {_fmt(r['pysb_s']):>9} "
            f"| {_fmt(r['bng_wall_s']):>12} "
            f"| {_fmt(r['bng_cpu_s']):>11} "
            f"| {_speedup(r['bng_wall_s'], r['pysb_s']):>11} "
            f"| {_speedup(r['bng_cpu_s'], r['pysb_s']):>10} |" + err
        )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark PySB netgen vs BioNetGen")
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Restrict to these model names (PySB example names)",
    )
    parser.add_argument(
        "--no-bng-validate", action="store_true", help="Skip BNG Validate models"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print per-model progress"
    )
    args = parser.parse_args()

    results = []

    # --- PySB example models ---
    pysb_models = PYSB_MODELS
    if args.models:
        pysb_models = [m for m in PYSB_MODELS if m in args.models]

    for name in pysb_models:
        if args.verbose:
            print(f"[pysb.examples.{name}]", flush=True)
        result = _run_benchmark(
            label=name,
            model_loader=lambda n=name: _load_pysb_example(n),
            verbose=args.verbose,
        )
        results.append(result)

    # --- BNG Validate models ---
    if not args.no_bng_validate:
        validate_dir = _bng_validate_directory()
        if validate_dir and os.path.isdir(validate_dir):
            bng_models = BNG_VALIDATE_MODELS
            if args.models:
                bng_models = [m for m in BNG_VALIDATE_MODELS if m in args.models]
            for name in bng_models:
                bngl = os.path.join(validate_dir, f"{name}.bngl")
                if not os.path.isfile(bngl):
                    continue
                label = f"[validate] {name}"
                if args.verbose:
                    print(f"{label}", flush=True)
                result = _run_benchmark(
                    label=label,
                    model_loader=lambda p=bngl: _load_bngl_model(p),
                    verbose=args.verbose,
                )
                results.append(result)

            # Constrained BNG Validate models (max_iter, max_stoich)
            constrained = BNG_VALIDATE_CONSTRAINED_MODELS
            if args.models:
                constrained = [(n, kw) for n, kw in constrained if n in args.models]
            for name, netgen_kw in constrained:
                bngl = os.path.join(validate_dir, f"{name}.bngl")
                if not os.path.isfile(bngl):
                    continue
                label = f"[validate] {name}"
                if args.verbose:
                    print(f"{label}", flush=True)
                result = _run_benchmark(
                    label=label,
                    model_loader=lambda p=bngl: _load_bngl_model(p),
                    verbose=args.verbose,
                    netgen_kwargs=netgen_kw,
                )
                results.append(result)
        else:
            print(
                "BNG Validate directory not found; skipping validate models.",
                file=sys.stderr,
            )

    # --- BNG Models2 models ---
    if not args.no_bng_validate:
        models2_dir = _bng_models2_directory()
        if models2_dir:
            bng_m2_models = BNG_MODELS2_MODELS
            if args.models:
                bng_m2_models = [m for m in BNG_MODELS2_MODELS if m in args.models]
            for name in bng_m2_models:
                bngl = os.path.join(models2_dir, f"{name}.bngl")
                if not os.path.isfile(bngl):
                    continue
                label = f"[models2] {name}"
                if args.verbose:
                    print(f"{label}", flush=True)
                result = _run_benchmark(
                    label=label,
                    model_loader=lambda p=bngl: _load_bngl_model(p),
                    verbose=args.verbose,
                )
                results.append(result)

    _print_table(results)

    # Summary stats
    ok_wall = [
        r for r in results if r["pysb_s"] is not None and r["bng_wall_s"] is not None
    ]
    ok_cpu = [
        r for r in ok_wall
        if r["bng_cpu_s"] is not None and r["pysb_s"] > 0
    ]
    if ok_wall:
        avg_wall = sum(r["bng_wall_s"] / r["pysb_s"] for r in ok_wall
                       if r["pysb_s"] > 0) / sum(
            1 for r in ok_wall if r["pysb_s"] > 0
        )
        print(f"Average speedup (wall): {avg_wall:.1f}x over {len(ok_wall)} models")
    if ok_cpu:
        avg_cpu = sum(r["bng_cpu_s"] / r["pysb_s"] for r in ok_cpu) / len(ok_cpu)
        print(f"Average speedup (CPU):  {avg_cpu:.1f}x over {len(ok_cpu)} models")
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\nModels with errors ({len(errors)}):")
        for r in errors:
            print(f"  {r['model']}: {r['error']}")


if __name__ == "__main__":
    main()
