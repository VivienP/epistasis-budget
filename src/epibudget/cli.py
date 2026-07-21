"""epibudget command-line interface. See docs/SPEC.md#8."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console
from rich.table import Table
from typer.core import TyperCommand

from epibudget.types import Mutation, Variant

if TYPE_CHECKING:
    from epibudget.graph import SelectionMethod
    from epibudget.scored_cache import CacheIdentity
    from epibudget.validate import Report

app = typer.Typer(
    add_completion=False,
    help="Experimental-design methods for budgeted protein-epistasis mapping.",
)
console = Console()

_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_TOKEN = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
_SEPARATORS = re.compile(r"[\s,;+]+")

# Short checkpoint names accepted on the CLI, resolved to their HuggingFace ids.
_MODEL_ALIASES = {
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
}

# Frozen confirmatory scoring identity (docs/VALIDATION.md "Outcome — frozen 650M headline"):
# `robustness`/`downstream` only ever analyse the completed 650M scored cache, so this is the one
# scoring configuration a cache may claim for those commands — checked against the sidecar rather
# than trusted from it.
_CONFIRMATORY_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
_CONFIRMATORY_SCORER_SEED = 0
_CONFIRMATORY_N_PERTURBATIONS = 16
_GATE2_ARGV_META_KEY = "epibudget.gate2.raw_argv"


class _Gate2Command(TyperCommand):
    """Retain this command's pre-parse argument tokens for exact run provenance."""

    # ctx is a click Context, but typer vendors its own click, so the concrete Context class differs
    # by typer version; Any keeps this override valid across typer versions.
    def parse_args(self, ctx: Any, args: list[str]) -> list[str]:
        ctx.meta[_GATE2_ARGV_META_KEY] = tuple(args)
        return super().parse_args(ctx, args)


def _resolve_model_id(name: str) -> str:
    """Map a short checkpoint name (``esm2_t12_35M``) to its HuggingFace id; pass full ids as-is."""
    return _MODEL_ALIASES.get(name, name)


def _variant_to_spec(variant: Variant) -> str:
    """Render a Variant as 1-indexed DMS notation (``V39A D40C``); the wild type is ``WT``."""
    muts = sorted(variant)
    return " ".join(f"{wt}{pos + 1}{mut}" for pos, wt, mut in muts) if muts else "WT"


def read_fasta_sequence(path: Path) -> str:
    """Read a single-record FASTA and return its residue sequence (upper-case, header dropped)."""
    body = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    seq = "".join(body).upper()
    if not seq:
        raise ValueError(f"no sequence found in {path}")
    return seq


def parse_variant(spec: str, wt: str) -> Variant:
    """Parse DMS notation (e.g. ``"V39A D40C"``, 1-indexed) into a Variant against ``wt``.

    Each token is ``<wt_aa><position_1indexed><mut_aa>``; the WT letter is checked against ``wt``
    so a wrong position or off-by-one is caught here. Empty / ``"WT"`` means the wild type.
    """
    stripped = spec.strip()
    if not stripped or stripped.upper() == "WT":
        return frozenset()
    muts: set[Mutation] = set()
    for token in _SEPARATORS.split(stripped):
        if not token:
            continue
        m = _TOKEN.match(token)
        if m is None:
            raise ValueError(f"malformed mutation token {token!r} (expected e.g. 'V39A')")
        wt_aa, pos_1, mut_aa = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        pos = pos_1 - 1
        if not 0 <= pos < len(wt):
            raise ValueError(
                f"position {pos_1} in {token!r} is out of range for a length-{len(wt)} WT"
            )
        if wt[pos] != wt_aa:
            raise ValueError(
                f"WT mismatch in {token!r}: sequence has {wt[pos]!r} at position {pos_1}"
            )
        if mut_aa not in _AA20:
            raise ValueError(f"{mut_aa!r} in {token!r} is not a valid amino acid")
        if mut_aa == wt_aa:
            raise ValueError(f"synonymous mutation {token!r}: mutant residue equals WT")
        muts.add((pos, wt_aa, mut_aa))
    return frozenset(muts)


def read_variant_specs(path: Path) -> list[str]:
    """Read variant specs from ``path`` — one per line, first comma-field, optional header row."""
    specs: list[str] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        cell = raw.split(",")[0].strip()
        if not cell:
            continue
        if i == 0 and cell.lower() in {"variant", "variants", "mutations"}:
            continue
        specs.append(cell)
    return specs


def _order_label(variant: Variant) -> str:
    return {1: "single", 2: "double", 3: "triple"}.get(len(variant), str(len(variant)))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


_METHODS = ("info", "fitness", "structural", "random", "practice")


def _print_validation_report(report: Report, run_dir: Path) -> None:
    """Print the invariant-#1 gate, provenance, and per-order Spearman recovery with coverage."""
    gate = "PASS" if report.predicted_epistasis_signal else "FAIL"
    console.print(
        f"[bold]Predicted Var[ε][/] = {report.var_predicted_epsilon:.4f}  "
        f"[{gate} invariant #1]  tolerance={report.predicted_epistasis_tolerance:.3e}   "
        f"candidates={report.n_candidates}  alphabet={report.candidate_alphabet!r}  "
        f"truth_terms={report.n_truth_terms}"
    )
    console.print(f"[bold]Truth Var[ε][/] = {report.var_epsilon:.4f}")
    by_key = {(r.method, r.budget): r for r in report.results}
    for order in ("pairwise", "third", "pooled"):
        label = "pooled (diagnostic only; cross-order)" if order == "pooled" else order
        table = Table(title=f"recovery — {label} (Spearman / Pearson, coverage)")
        table.add_column("B", justify="right")
        for method in _METHODS:
            table.add_column(method, justify="right")
        for budget in report.budgets:
            cells = []
            for method in _METHODS:
                metric = next(m for m in by_key[method, budget].metrics if m.order == order)
                cov = metric.coverage_fraction
                cells.append(f"{_fmt(metric.spearman)}/{_fmt(metric.pearson)} c{cov:.2f}")
            table.add_row(str(budget), *cells)
        console.print(table)

    hits = Table(title="hit-rate@B (top-fitness capture)")
    hits.add_column("B", justify="right")
    for method in _METHODS:
        hits.add_column(method, justify="right")
    for budget in report.budgets:
        hits.add_row(str(budget), *(f"{by_key[m, budget].hit_rate:.2f}" for m in _METHODS))
    console.print(hits)
    console.print(f"wrote {run_dir / 'metrics.json'}")


@app.command()
def allocate(
    fasta: str = typer.Option(..., help="Path to the wild-type sequence (FASTA)."),
    positions: str = typer.Option(
        ..., help="Comma-separated 1-indexed positions, e.g. 39,40,41,54."
    ),
    budget: int = typer.Option(..., help="Number of variants to select (B wells)."),
    model: str = typer.Option("esm2_t33_650M", help="ESM-2 checkpoint short name or HF id."),
    method: str = typer.Option(
        "info",
        help="Selection weighting: info (ESM dispersion x loops-braced) or structural (loops only)",
    ),
    lambda_: float = typer.Option(0.0, "--lambda", min=0.0, max=1.0, help="0=info, 1=fitness."),
    max_order: int = typer.Option(3, help="Max interaction order (2 or 3)."),
    alphabet: str = typer.Option(_AA20, help="Per-site candidate alphabet."),
    n_perturbations: int = typer.Option(16, help="Masking passes for var[ΔG] (the info prior)."),
    seed: int = typer.Option(0),
    device: str = typer.Option("cpu", help="Compute device: cpu, cuda, or auto (GPU if present)."),
    threads: int = typer.Option(0, help="torch CPU threads; 0 = library default."),
    out: str = typer.Option("allocation.json"),
) -> None:
    """Rank the B most epistasis-informative variants for a target."""
    from epibudget.acquisition import allocate as allocate_variants  # noqa: PLC0415
    from epibudget.data import enumerate_candidates  # noqa: PLC0415
    from epibudget.graph import SELECTION_METHODS, selection_graph  # noqa: PLC0415
    from epibudget.scoring import ConjointScorer  # noqa: PLC0415

    if method not in SELECTION_METHODS:
        raise typer.BadParameter(
            f"--method must be one of {', '.join(SELECTION_METHODS)}, got {method!r}"
        )

    wt = read_fasta_sequence(Path(fasta))
    sites = [int(p) - 1 for p in positions.split(",")]
    wt_at = tuple(wt[p] for p in sites)
    candidates = enumerate_candidates(sites, wt_at, allowed_aa=alphabet, max_order=max_order)

    console.print(
        f"[bold]allocate[/] scoring {len(candidates)} candidates with {model} on {device} …"
    )
    scorer = ConjointScorer(
        _resolve_model_id(model),
        device=device,
        n_perturbations=n_perturbations,
        seed=seed,
        num_threads=threads if threads > 0 else None,
    )
    scored = scorer.score_batch(wt, candidates)
    graph = selection_graph(scored, max_order, cast("SelectionMethod", method))
    result = allocate_variants(
        graph, scored, budget, lambda_=lambda_, seed=seed, model_id=model, method=method
    )

    dg_of = {sv.variant: sv.delta_g for sv in scored}
    table = Table(title=f"allocate B={budget} method={method} λ={lambda_} model={model}")
    for column, justify in (("rank", "right"), ("variant", "left"), ("order", "left")):
        table.add_column(column, justify=justify)  # type: ignore[arg-type]
    table.add_column("ΔG", justify="right")
    table.add_column("info-gain", justify="right")
    for rank, (variant, gain) in enumerate(
        zip(result.selected, result.expected_info_gain, strict=True), start=1
    ):
        table.add_row(
            str(rank),
            _variant_to_spec(variant),
            _order_label(variant),
            f"{dg_of[variant]:+.4f}",
            f"{gain:.4g}",
        )
    console.print(table)
    Path(out).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"wrote allocation to {out}")


@app.command()
def validate(
    dataset: str = typer.Option(
        "gb1_wu2016", help="Validation dataset id (gb1_wu2016 or trpb_johnston2024)."
    ),
    budgets: str = typer.Option("48,96,192", help="Comma-separated budgets."),
    model: str = typer.Option("esm2_t12_35M", help="ESM-2 checkpoint short name or HF id."),
    seeds: int = typer.Option(20, help="Random-baseline seeds."),
    out: str = typer.Option("report/", help="Report root; a run subdirectory is created."),
    data: str = typer.Option(
        "", help="Path to the dataset CSV; empty uses the selected dataset's default path."
    ),
    alphabet: str = typer.Option(
        _AA20, help="Per-site candidate alphabet; a reduced set keeps the fast-model run tractable."
    ),
    n_perturbations: int = typer.Option(16, help="Masking passes for var[ΔG] (the info prior)."),
    max_order: int = typer.Option(3, help="Max interaction order (2 or 3)."),
    device: str = typer.Option("cpu", help="Compute device: cpu, cuda, or auto (GPU if present)."),
    threads: int = typer.Option(0, help="torch CPU threads; 0 = library default."),
    batch_size: int = typer.Option(32, help="Scoring batch size (throughput only)."),
    scored_cache: str = typer.Option(
        "", help="JSONL scored-variant cache; resumes a long run after an interruption."
    ),
) -> None:
    """Run the validation harness; the frozen headline requires every explicit registered setting.

    ``--dataset`` selects the landscape (loader + reference construct) from the registry in
    ``epibudget.data``; an unregistered identifier is rejected before any scoring. GB1 is the
    default and its resolved path/reference are unchanged.
    """
    import hashlib  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    from epibudget.data import enumerate_candidates, resolve_dataset  # noqa: PLC0415
    from epibudget.scored_cache import build_cache_metadata, score_with_cache  # noqa: PLC0415
    from epibudget.scoring import ConjointScorer  # noqa: PLC0415
    from epibudget.validate import run_validation  # noqa: PLC0415

    try:
        spec = resolve_dataset(dataset)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    data_path = Path(data) if data else Path(spec.default_data_path)
    landscape = spec.loader(data_path)
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    candidates = enumerate_candidates(
        spec.sites, spec.wt_at_sites, allowed_aa=alphabet, max_order=max_order
    )
    console.print(
        f"[bold]validate[/] {spec.identifier}: scoring {len(candidates)} candidates "
        f"(alphabet={alphabet!r}) with {model} on {device} …"
    )
    scorer_seed = 0
    scorer = ConjointScorer(
        _resolve_model_id(model),
        device=device,
        n_perturbations=n_perturbations,
        seed=scorer_seed,
        batch_size=batch_size,
        num_threads=threads if threads > 0 else None,
    )
    if scored_cache:
        metadata = build_cache_metadata(
            scorer,
            spec.wt_sequence,
            candidates,
            candidate_alphabet=alphabet,
            max_order=max_order,
        )
        scored = score_with_cache(
            scorer,
            spec.wt_sequence,
            candidates,
            Path(scored_cache),
            metadata=metadata,
        )
    else:
        scored = scorer.score_batch(spec.wt_sequence, candidates)

    run_dir = Path(out) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = run_validation(
        scored,
        landscape,
        budgets=[int(b) for b in budgets.split(",")],
        seeds=seeds,
        model_id=model,
        out_dir=run_dir,
        dataset=spec.identifier,
        max_order=max_order,
        candidate_alphabet=alphabet,
        scorer_seed=scorer_seed,
        n_perturbations=n_perturbations,
        device=scorer.device,
        data_sha256=data_sha256,
        wt_sequence=spec.wt_sequence,
        sites=spec.sites,
    )
    _print_validation_report(report, run_dir)


@app.command()
def robustness(
    scored_cache: str = typer.Option(..., help="Completed JSONL scored-variant cache to analyse."),
    data: str = typer.Option("data/proteingym/gb1_wu2016.csv", help="Path to the GB1 CSV."),
    alphabet: str = typer.Option(_AA20, help="Per-site alphabet the cache was scored over."),
    budgets: str = typer.Option("48,96,192", help="Comma-separated budgets."),
    seeds: int = typer.Option(20, help="Random-baseline seeds."),
    max_order: int = typer.Option(3, help="Max interaction order (2 or 3)."),
    n_folds: int = typer.Option(5, help="Cross-fit folds for the scale-sensitivity analysis."),
    out: str = typer.Option("report/", help="Report root; a run subdirectory is created."),
) -> None:
    """Post-hoc robustness analyses on a completed run (no GPU); see docs/specs."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from epibudget.data import (  # noqa: PLC0415
        GB1_SITES,
        GB1_WT_AT_SITES,
        GB1_WT_SEQUENCE,
        enumerate_candidates,
        load_gb1,
    )
    from epibudget.provenance import write_json_exclusive  # noqa: PLC0415
    from epibudget.robustness import robustness_report  # noqa: PLC0415
    from epibudget.scored_cache import (  # noqa: PLC0415
        CacheIdentity,
        validate_cache_against_universe,
    )

    if n_folds < 2:  # noqa: PLR2004 — >= 2 for out-of-fold cross-fitting
        raise typer.BadParameter(f"--n-folds must be >= 2, got {n_folds}")

    landscape = load_gb1(Path(data))
    enumerated = enumerate_candidates(
        GB1_SITES, GB1_WT_AT_SITES, allowed_aa=alphabet, max_order=max_order
    )
    try:
        cache, metadata, expected_identity = validate_cache_against_universe(
            Path(scored_cache),
            enumerated,
            candidate_alphabet=alphabet,
            max_order=max_order,
            model_id=_CONFIRMATORY_MODEL_ID,
            scorer_seed=_CONFIRMATORY_SCORER_SEED,
            n_perturbations=_CONFIRMATORY_N_PERTURBATIONS,
            wt_sequence=GB1_WT_SEQUENCE,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    scored = [cache[v] for v in enumerated]  # enumeration order: reproduces the frozen selections
    model_id = metadata.model_id
    observed_identity = CacheIdentity.from_metadata(metadata)

    run_dir = Path(out) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = robustness_report(
        scored,
        landscape,
        budgets=[int(b) for b in budgets.split(",")],
        seeds=seeds,
        max_order=max_order,
        n_folds=n_folds,
        model_id=model_id,
        out_dir=run_dir,
    )
    # `robustness` is descriptive/non-decision-bearing (RobustnessReport carries no provenance
    # field), so the same independently-validated cache identity checked above is recorded as its
    # own sidecar rather than folded into robustness.json's schema.
    write_json_exclusive(
        run_dir / "robustness_cache_identity.json",
        {
            "scored_cache_identity_expected": expected_identity.model_dump(mode="json"),
            "scored_cache_identity_observed": observed_identity.model_dump(mode="json"),
        },
    )
    console.print(
        f"[bold]robustness[/] {report.n_candidates} candidates, {n_folds}-fold cross-fit; "
        f"wrote {run_dir / 'robustness.json'}"
    )
    for scale in report.scale_sensitivity:
        agree = "agrees" if scale.ranking_agrees else "DIFFERS"
        console.print(
            f"  {scale.order:<8} global={scale.global_ranking} crossfit={scale.crossfit_ranking} "
            f"({agree})"
        )


def _git_lines(repo: Path, *args: str) -> list[str]:
    import subprocess  # noqa: PLC0415

    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in result.stdout.splitlines() if line]


def _gate2_required_git_lines(repo: Path, *args: str) -> list[str]:
    """Run a required Gate 2 Git query, propagating every execution failure."""
    import subprocess  # noqa: PLC0415

    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in result.stdout.splitlines() if line]


def _gate2_run_stamp() -> str:
    from datetime import UTC, datetime  # noqa: PLC0415

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _gate2_provenance(
    repo: Path,
    cache_path: Path,
    sidecar_path: Path,
    data_path: Path,
    universe_sha256: str,
    argv: list[str],
    *,
    expected_identity: CacheIdentity,
    observed_identity: CacheIdentity,
    started_at_utc: str,
) -> dict[str, object]:
    """Assemble the cache-validated provenance required by the Gate 2 decision gate."""
    import hashlib  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    from epibudget.gate2 import PROTOCOL_VERSION, RUN_TYPE  # noqa: PLC0415
    from epibudget.provenance import (  # noqa: PLC0415
        changed_scientific_files,
        workspace_code_diff_sha256,
    )

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        head = _gate2_required_git_lines(repo, "rev-parse", "HEAD")
        if len(head) != 1 or re.fullmatch(r"[0-9a-fA-F]{40,64}", head[0]) is None:
            raise ValueError("Gate 2 git provenance requires one valid execution commit")
        execution_commit = head[0].lower()
        dirty = bool(_gate2_required_git_lines(repo, "status", "--porcelain"))
        code_diff = ""
        changed_files: list[str] = []
        if dirty:
            code_diff = workspace_code_diff_sha256(repo, execution_commit)
            changed_files = changed_scientific_files(repo, execution_commit)
            if re.fullmatch(r"[0-9a-f]{64}", code_diff) is None:
                raise ValueError("Gate 2 git provenance requires a valid dirty-tree diff hash")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Gate 2 git provenance could not be established: {exc}") from exc
    identity_expected = expected_identity.model_dump(mode="json")
    identity_observed = observed_identity.model_dump(mode="json")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_type": RUN_TYPE,
        "scored_cache_sha256": sha256(cache_path),
        "scored_cache_sidecar_sha256": sha256(sidecar_path),
        "dataset_sha256": sha256(data_path),
        "candidate_universe_sha256": universe_sha256,
        "scored_cache_identity_expected": identity_expected,
        "scored_cache_identity_observed": identity_observed,
        "scored_cache_validator_status": "passed",
        "execution_commit": execution_commit,
        "code_state": "dirty" if dirty else "clean",
        "code_diff_sha256": code_diff,
        "changed_scientific_files": changed_files,
        "argv": argv,
        "exact_command": subprocess.list2cmdline(argv),
        "started_at_utc": started_at_utc,
        # Structurally complete markers for the in-memory decision gate; only these three fields
        # are replaced after the analysis returns, so every eligibility-bearing field is stable.
        "completed_at_utc": started_at_utc,
        "elapsed_seconds": 0.0,
    }


@app.command(cls=_Gate2Command)
def gate2(
    ctx: typer.Context,
    scored_cache: str = typer.Option(
        "report/scored_650m.jsonl", help="Completed 650M JSONL score cache to analyse."
    ),
    data: str = typer.Option("data/proteingym/gb1_wu2016.csv", help="Path to the GB1 CSV."),
    alphabet: str = typer.Option(_AA20, help="Per-site alphabet the cache was scored over."),
    budgets: str = typer.Option("48,96,192", help="Comma-separated budgets."),
    random_seeds: int = typer.Option(20, min=0, help="Random-baseline seeds."),
    structural_seeds: int = typer.Option(100, min=0, help="Seeded structural tie-breaks."),
    n_folds: int = typer.Option(5, min=2, help="Cross-fit folds for slope attribution."),
    max_order: int = typer.Option(3, min=2, max=3, help="Maximum interaction order."),
    out: str = typer.Option("report/", help="Report root; a UTC run directory is created."),
) -> None:
    """Run the registered corrective Gate 2 analysis from a completed score cache (CPU-only)."""
    from datetime import UTC, datetime  # noqa: PLC0415
    from time import perf_counter  # noqa: PLC0415

    from epibudget.data import (  # noqa: PLC0415
        GB1_SITES,
        GB1_WT_AT_SITES,
        GB1_WT_SEQUENCE,
        enumerate_candidates,
        load_gb1,
    )
    from epibudget.gate2 import finalize_gate2_report, gate2_report  # noqa: PLC0415
    from epibudget.provenance import write_json_exclusive  # noqa: PLC0415
    from epibudget.scored_cache import (  # noqa: PLC0415
        CacheIdentity,
        cache_metadata_path,
        candidate_sha256,
        validate_cache_against_universe,
    )

    started_clock = perf_counter()
    started_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    budget_list = [int(value) for value in budgets.split(",")]
    cache_path = Path(scored_cache)
    data_path = Path(data)
    enumerated = enumerate_candidates(
        GB1_SITES, GB1_WT_AT_SITES, allowed_aa=alphabet, max_order=max_order
    )
    try:
        cache, metadata, expected_identity = validate_cache_against_universe(
            cache_path,
            enumerated,
            candidate_alphabet=alphabet,
            max_order=max_order,
            model_id=_CONFIRMATORY_MODEL_ID,
            scorer_seed=_CONFIRMATORY_SCORER_SEED,
            n_perturbations=_CONFIRMATORY_N_PERTURBATIONS,
            wt_sequence=GB1_WT_SEQUENCE,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    scored = [cache[variant] for variant in enumerated]
    observed_identity = CacheIdentity.from_metadata(metadata)
    sidecar_path = cache_metadata_path(cache_path)
    raw_argv = ctx.meta.get(_GATE2_ARGV_META_KEY)
    if not isinstance(raw_argv, tuple) or not all(isinstance(arg, str) for arg in raw_argv):
        raise typer.BadParameter("Gate 2 could not capture its exact command arguments")
    argv = ["epibudget", "gate2", *raw_argv]
    repo = Path(__file__).resolve().parents[2]
    try:
        provenance = _gate2_provenance(
            repo,
            cache_path,
            sidecar_path,
            data_path,
            candidate_sha256(enumerated),
            argv,
            expected_identity=expected_identity,
            observed_identity=observed_identity,
            started_at_utc=started_at_utc,
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    # Labels are loaded only after the complete scored universe has passed its independent gate.
    landscape = load_gb1(data_path)
    console.print(
        f"[bold]gate2[/] {len(scored)} cache-validated candidates; running corrective CPU analysis"
    )
    report = gate2_report(
        scored,
        landscape,
        budget_list,
        random_seeds=random_seeds,
        structural_seeds=structural_seeds,
        n_folds=n_folds,
        max_order=max_order,
        alphabet=alphabet,
        dataset="gb1_wu2016",
        model_id=metadata.model_id,
        provenance=provenance,
    )
    completed_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    final_provenance = {
        **provenance,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": perf_counter() - started_clock,
    }
    report = finalize_gate2_report(report, scored, final_provenance)
    run_dir = Path(out) / _gate2_run_stamp()
    report_path = run_dir / "gate2.json"
    write_json_exclusive(report_path, report.model_dump(mode="json"))
    console.print(
        f"decision={report.decision.decision}  "
        f"architecture_decision_eligible={report.architecture_decision_eligible}  "
        "public_claim_eligible=False"
    )
    console.print(f"wrote {report_path}")


def _downstream_provenance(
    repo: Path,
    cache_path: Path,
    sidecar: Path,
    data_path: Path,
    universe_sha256: str,
    headline: Path,
    command: str,
    *,
    partitions: int,
    budgets: list[int],
    seeds: int,
    n_folds: int,
    alphabet: str,
    max_order: int,
    n_perturbations: int,
    expected_identity: CacheIdentity,
    observed_identity: CacheIdentity,
    started_at_utc: str,
    completed_at_utc: str,
) -> dict[str, object]:
    """Assemble the full provenance dict for one ``downstream`` execution.

    ``completed_at_utc`` is the caller's own post-computation timestamp (captured after
    ``downstream_report`` returns), never computed here, so provenance never claims a completion
    time earlier than the benchmark actually finished. ``expected_identity``/``observed_identity``
    are the complete 8-field ``CacheIdentity`` pair ``validate_cache_against_universe`` actually
    checked the sidecar against — ``expected_identity`` is that call's own independently-computed
    return value (never re-derived from the sidecar), and ``observed_identity`` is read from the
    already cache-validated sidecar metadata, so a reader never has to trust that the earlier gate
    ran — both sides of every check are on the record.
    """
    import hashlib  # noqa: PLC0415

    from epibudget.downstream import (  # noqa: PLC0415
        _ESTIMANDS,
        _METHODS,
        _REGIMES,
        AMENDMENT_VERSION,
        GRID_MAIN,
        GRID_PAIR,
        N_INNER_FOLDS,
        PROTOCOL_VERSION,
        partition_salt,
        protocol_profile_conformance,
    )
    from epibudget.provenance import (  # noqa: PLC0415
        changed_scientific_files,
        workspace_code_diff_sha256,
    )

    def sha(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    head = _git_lines(repo, "rev-parse", "HEAD")
    dirty = bool(_git_lines(repo, "status", "--porcelain"))
    execution_commit = head[0] if head else ""
    code_diff = ""
    changed_files: list[str] = []
    if dirty and execution_commit:
        code_diff = workspace_code_diff_sha256(repo, execution_commit)
        changed_files = changed_scientific_files(repo, execution_commit)

    # CLI-boundary confirmatory-profile check: computed defensively again,
    # independently, inside downstream_report()/_decision_summary() itself — this is only an early,
    # operator-facing signal, never the authoritative decision gate.
    cli_conformance = protocol_profile_conformance(
        protocol_version=PROTOCOL_VERSION,
        partitions=partitions,
        outer_folds=n_folds,
        budgets=budgets,
        alphabet=alphabet,
        max_order=max_order,
        n_perturbations=n_perturbations,
        random_seeds=range(seeds),
        inner_folds=N_INNER_FOLDS,
        estimands=_ESTIMANDS,
        missingness_regimes=_REGIMES,
        methods=_METHODS,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "amendment_version": AMENDMENT_VERSION,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "scored_cache_sha256": sha(cache_path),
        "scored_cache_sidecar_sha256": sha(sidecar),
        "scored_cache_identity_expected": expected_identity.model_dump(mode="json"),
        "scored_cache_identity_observed": observed_identity.model_dump(mode="json"),
        "dataset_sha256": sha(data_path),
        "candidate_universe_sha256": universe_sha256,
        "headline_artifact_path": str(headline) if headline.is_file() else None,
        "headline_artifact_sha256": sha(headline),
        "execution_commit": execution_commit,
        "base_commit_sha": execution_commit,
        "code_state": "dirty" if dirty else "clean",
        "code_diff_sha256": code_diff,
        "changed_scientific_files": changed_files,
        "exact_command": command,
        "inner_fold_policy": (
            "identity-sorted balanced rank%n_inner over sha256(inner_salt:canonical_id); "
            "fallback to strongest-shrinkage grid corner below n_inner distinct identities"
        ),
        "inner_salt": "sha256(epibudget-downstream-inner:v1)",
        "n_inner_folds": N_INNER_FOLDS,
        "alpha_grid_main": list(GRID_MAIN),
        "alpha_grid_pair": list(GRID_PAIR),
        "partition_salts": [partition_salt(i) for i in range(partitions)],
        "estimands": list(_ESTIMANDS),
        "missingness_regimes": list(_REGIMES),
        "budgets": budgets,
        "seeds": seeds,
        "alpha_selection_source": "see deterministic_records[*]/random_records[*].alpha_*",
        "cli_protocol_profile_conforming": cli_conformance.conforming,
        "cli_protocol_profile_mismatches": cli_conformance.mismatches,
        "status": "provisional",
    }


@app.command()
def downstream(
    scored_cache: str = typer.Option(..., help="Completed JSONL scored-variant cache to analyse."),
    dataset: str = typer.Option(
        "gb1_wu2016", help="Landscape id (gb1_wu2016 or trpb_johnston2024)."
    ),
    data: str = typer.Option(
        "", help="Path to the dataset CSV; empty uses the selected dataset's default path."
    ),
    alphabet: str = typer.Option(_AA20, help="Per-site alphabet the cache was scored over."),
    n_perturbations: int = typer.Option(
        _CONFIRMATORY_N_PERTURBATIONS,
        help="Masking passes the cache carries; only 16 conforms (0 drops the info prior).",
    ),
    budgets: str = typer.Option("48,96,192", help="Comma-separated budgets."),
    seeds: int = typer.Option(20, help="Random-baseline seeds."),
    max_order: int = typer.Option(3, help="Max interaction order (2 or 3)."),
    n_folds: int = typer.Option(5, help="Outer held-out folds."),
    partitions: int = typer.Option(20, help="Independent salted fold partitions."),
    headline: str = typer.Option(
        "report/20260711T091947Z/metrics.json", help="Frozen headline artifact for provenance."
    ),
    out: str = typer.Option("report/", help="Report root; a run subdirectory is created."),
) -> None:
    """Post-registration downstream-impact benchmark on a completed cache (CPU-only, no GPU).

    A confirmatory run under protocol amendment 1 (docs/specs/downstream.md); ``decision_eligible``
    is false whenever ``--partitions`` is below the frozen ``EXPECTED_PARTITIONS=20`` register.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from epibudget.data import (  # noqa: PLC0415
        enumerate_candidates,
        resolve_dataset,
    )
    from epibudget.downstream import downstream_report  # noqa: PLC0415
    from epibudget.provenance import write_json_atomic  # noqa: PLC0415
    from epibudget.scored_cache import (  # noqa: PLC0415
        CacheIdentity,
        cache_metadata_path,
        candidate_sha256,
        validate_cache_against_universe,
    )

    if n_folds < 2:  # noqa: PLR2004 — need a held-out fold
        raise typer.BadParameter(f"--n-folds must be >= 2, got {n_folds}")
    if partitions < 1:
        raise typer.BadParameter(f"--partitions must be >= 1, got {partitions}")

    try:
        spec = resolve_dataset(dataset)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    cache_path = Path(scored_cache)
    data_path = Path(data) if data else Path(spec.default_data_path)
    landscape = spec.loader(data_path)
    enumerated = enumerate_candidates(
        spec.sites, spec.wt_at_sites, allowed_aa=alphabet, max_order=max_order
    )
    try:
        cache, metadata, expected_identity = validate_cache_against_universe(
            cache_path,
            enumerated,
            candidate_alphabet=alphabet,
            max_order=max_order,
            model_id=_CONFIRMATORY_MODEL_ID,
            scorer_seed=_CONFIRMATORY_SCORER_SEED,
            n_perturbations=n_perturbations,
            wt_sequence=spec.wt_sequence,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    scored = [cache[v] for v in enumerated]  # enumeration order: reproduces the frozen selections
    model_id = metadata.model_id
    observed_identity = CacheIdentity.from_metadata(metadata)

    sidecar = cache_metadata_path(cache_path)
    budget_list = [int(b) for b in budgets.split(",")]
    repo = Path(__file__).resolve().parents[2]
    command = (
        f"epibudget downstream --dataset {dataset} --scored-cache {scored_cache} --data {data} "
        f"--alphabet {alphabet} --n-perturbations {n_perturbations} --budgets {budgets} "
        f"--seeds {seeds} --max-order {max_order} --n-folds {n_folds} --partitions {partitions} "
        f"--headline {headline} --out {out}"
    )
    run_dir = Path(out) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    started_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    console.print(
        f"[bold]downstream[/] {len(scored)} candidates, {partitions}x{n_folds} fold-instances, "
        f"seeds={seeds}; running (CPU) ..."
    )
    # `provenance`/`out_dir` are intentionally withheld here: `completed_at_utc` below is only
    # correct once this call has returned, so the final report is written once, after that.
    report = downstream_report(
        scored,
        landscape,
        budget_list,
        seeds=seeds,
        n_folds=n_folds,
        partitions=partitions,
        max_order=max_order,
        sites=spec.sites,
        wt_at_sites=spec.wt_at_sites,
        alphabet=alphabet,
        n_perturbations=n_perturbations,
        dataset=spec.identifier,
        model_id=model_id,
    )
    completed_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    provenance = _downstream_provenance(
        repo,
        cache_path,
        sidecar,
        data_path,
        candidate_sha256(enumerated),
        Path(headline),
        command,
        partitions=partitions,
        budgets=budget_list,
        seeds=seeds,
        n_folds=n_folds,
        alphabet=alphabet,
        max_order=max_order,
        n_perturbations=n_perturbations,
        expected_identity=expected_identity,
        observed_identity=observed_identity,
        started_at_utc=started_at_utc,
        completed_at_utc=completed_at_utc,
    )
    if not provenance["cli_protocol_profile_conforming"]:
        console.print(
            "[yellow]warning:[/] executed configuration does not match the frozen confirmatory "
            f"profile — mismatches={provenance['cli_protocol_profile_mismatches']}; this run can "
            "never be decision_eligible regardless of partition coverage"
        )
    report = report.model_copy(update={"provenance": provenance})
    write_json_atomic(run_dir / "downstream.json", report.model_dump(mode="json"))
    decision = report.decision
    console.print(
        f"  decision_eligible(structural)={decision.structural_gate.decision_eligible}  "
        f"structural_downstream_supported={decision.structural_downstream_supported}  "
        f"esm_uncertainty_supported={decision.esm_uncertainty_supported}"
    )
    for gate, label in (
        (decision.structural_gate, "structural-fitness"),
        (decision.esm_gate, "info-structural"),
    ):
        console.print(
            f"  {label:<18} status={gate.status:<28} coverage={gate.observed_valid_partitions}/"
            f"{gate.expected_partitions} sign={gate.sign_positive}/{gate.sign_threshold} "
            f"mean={_fmt(gate.global_mean_delta)}"
        )
    console.print(f"wrote {run_dir / 'downstream.json'} (status={provenance['status']})")


@app.command()
def score(
    fasta: str = typer.Option(..., help="Path to the wild-type sequence (FASTA)."),
    variants: str = typer.Option(..., help="CSV of variants to score (DMS notation, e.g. V39A)."),
    model: str = typer.Option(
        "facebook/esm2_t12_35M_UR50D", help="ESM-2 checkpoint id (HuggingFace)."
    ),
    n_perturbations: int = typer.Option(16, help="Masking passes for var[ΔG]; 0 disables."),
    seed: int = typer.Option(0),
) -> None:
    """Debug: dump conjoint ΔG + masking-perturbation variance for a variant list."""
    from epibudget.scoring import ConjointScorer  # noqa: PLC0415  # heavy import, deferred

    wt = read_fasta_sequence(Path(fasta))
    specs = read_variant_specs(Path(variants))
    parsed = [parse_variant(spec, wt) for spec in specs]

    scorer = ConjointScorer(model, n_perturbations=n_perturbations, seed=seed)
    scored = scorer.score_batch(wt, parsed)

    table = Table(title=f"conjoint scores — {model}")
    table.add_column("variant")
    table.add_column("order")
    table.add_column("ΔG", justify="right")
    table.add_column("var[ΔG]", justify="right")
    for spec, sv in zip(specs, scored, strict=True):
        table.add_row(
            spec or "WT", _order_label(sv.variant), f"{sv.delta_g:+.4f}", f"{sv.var_delta_g:.4f}"
        )
    console.print(table)


if __name__ == "__main__":
    app()
