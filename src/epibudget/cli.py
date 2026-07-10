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
