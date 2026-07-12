"""epibudget command-line interface. See docs/SPEC.md#8."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from epibudget.types import Mutation, Variant

if TYPE_CHECKING:
    from epibudget.scored_cache import CacheIdentity
    from epibudget.validate import Report

app = typer.Typer(
    add_completion=False,
    help="Information-optimal experimental budget allocation for mapping protein epistasis.",
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
    gate = "PASS" if report.var_epsilon > 0.0 else "FAIL"
    console.print(
        f"[bold]Var[ε][/] = {report.var_epsilon:.4f}  [{gate} invariant #1]   "
        f"candidates={report.n_candidates}  alphabet={report.candidate_alphabet!r}  "
        f"truth_terms={report.n_truth_terms}"
    )
    by_key = {(r.method, r.budget): r for r in report.results}
    for order in ("pairwise", "third", "pooled"):
        table = Table(title=f"recovery — {order} (Spearman / Pearson, coverage)")
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
    from epibudget.epistasis import predicted_epistasis  # noqa: PLC0415
    from epibudget.graph import EpistasisFactorGraph  # noqa: PLC0415
    from epibudget.scoring import ConjointScorer  # noqa: PLC0415

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
    graph = EpistasisFactorGraph(
        predicted_epistasis(scored, max_order), {sv.variant: sv.var_delta_g for sv in scored}
    )
    result = allocate_variants(graph, scored, budget, lambda_=lambda_, seed=seed, model_id=model)

    dg_of = {sv.variant: sv.delta_g for sv in scored}
    table = Table(title=f"allocate B={budget} λ={lambda_} model={model}")
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
    dataset: str = typer.Option("gb1_wu2016", help="Validation dataset id."),
    budgets: str = typer.Option("48,96,192", help="Comma-separated budgets."),
    model: str = typer.Option("esm2_t12_35M", help="ESM-2 checkpoint short name or HF id."),
    seeds: int = typer.Option(20, help="Random-baseline seeds."),
    out: str = typer.Option("report/", help="Report root; a run subdirectory is created."),
    data: str = typer.Option("data/proteingym/gb1_wu2016.csv", help="Path to the GB1 CSV."),
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
    """Run the GB1 harness; the frozen headline requires every explicit registered setting."""
    import hashlib  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    from epibudget.data import (  # noqa: PLC0415
        GB1_SITES,
        GB1_WT_AT_SITES,
        GB1_WT_SEQUENCE,
        enumerate_candidates,
        load_gb1,
    )
    from epibudget.scored_cache import build_cache_metadata, score_with_cache  # noqa: PLC0415
    from epibudget.scoring import ConjointScorer  # noqa: PLC0415
    from epibudget.validate import run_validation  # noqa: PLC0415

    data_path = Path(data)
    landscape = load_gb1(data_path)
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    candidates = enumerate_candidates(
        GB1_SITES, GB1_WT_AT_SITES, allowed_aa=alphabet, max_order=max_order
    )
    console.print(
        f"[bold]validate[/] scoring {len(candidates)} candidates "
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
            GB1_WT_SEQUENCE,
            candidates,
            candidate_alphabet=alphabet,
            max_order=max_order,
        )
        scored = score_with_cache(
            scorer,
            GB1_WT_SEQUENCE,
            candidates,
            Path(scored_cache),
            metadata=metadata,
        )
    else:
        scored = scorer.score_batch(GB1_WT_SEQUENCE, candidates)

    run_dir = Path(out) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report = run_validation(
        scored,
        landscape,
        budgets=[int(b) for b in budgets.split(",")],
        seeds=seeds,
        model_id=model,
        out_dir=run_dir,
        dataset=dataset,
        max_order=max_order,
        candidate_alphabet=alphabet,
        scorer_seed=scorer_seed,
        n_perturbations=n_perturbations,
        device=scorer.device,
        data_sha256=data_sha256,
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
    data: str = typer.Option("data/proteingym/gb1_wu2016.csv", help="Path to the GB1 CSV."),
    alphabet: str = typer.Option(_AA20, help="Per-site alphabet the cache was scored over."),
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
        GB1_SITES,
        GB1_WT_AT_SITES,
        GB1_WT_SEQUENCE,
        enumerate_candidates,
        load_gb1,
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

    cache_path = Path(scored_cache)
    data_path = Path(data)
    landscape = load_gb1(data_path)
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
    scored = [cache[v] for v in enumerated]  # enumeration order: reproduces the frozen selections
    model_id = metadata.model_id
    observed_identity = CacheIdentity.from_metadata(metadata)

    sidecar = cache_metadata_path(cache_path)
    budget_list = [int(b) for b in budgets.split(",")]
    repo = Path(__file__).resolve().parents[2]
    command = (
        f"epibudget downstream --scored-cache {scored_cache} --data {data} --alphabet {alphabet} "
        f"--budgets {budgets} --seeds {seeds} --max-order {max_order} --n-folds {n_folds} "
        f"--partitions {partitions} --headline {headline} --out {out}"
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
        sites=GB1_SITES,
        wt_at_sites=GB1_WT_AT_SITES,
        alphabet=alphabet,
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
