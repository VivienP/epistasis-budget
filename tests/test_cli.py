"""Offline tests for the CLI's pure helpers (FASTA reading, variant parsing).

The ``score`` command itself needs an ESM-2 forward pass, so it is not exercised here; only the
input-parsing helpers are, which is where the surprising failure modes (indexing, WT mismatch) live.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from math import exp
from pathlib import Path

import pytest
from typer.testing import CliRunner

from epibudget import cli as cli_module
from epibudget import data as data_module
from epibudget import downstream as downstream_module
from epibudget import gate2 as gate2_module
from epibudget import provenance as provenance_module
from epibudget import scored_cache as scored_cache_module
from epibudget import scoring
from epibudget.cli import (
    _AA20,
    _CONFIRMATORY_MODEL_ID,
    _CONFIRMATORY_N_PERTURBATIONS,
    _CONFIRMATORY_SCORER_SEED,
    app,
    parse_variant,
    read_fasta_sequence,
    read_variant_specs,
)
from epibudget.data import (
    GB1_SITES,
    GB1_WT_AT_SITES,
    GB1_WT_SEQUENCE,
    TRPB_SITES,
    TRPB_WT_AT_SITES,
    TRPB_WT_SEQUENCE,
    apply_mutations,
    enumerate_candidates,
    load_gb1,
)
from epibudget.scored_cache import CacheMetadata, cache_metadata_path
from epibudget.types import ScoredVariant, Variant

_ALLOC_BUDGET = 4
_PAIR_ORDER = 2
_GATE2_BUDGETS = [48, 96, 192]
_GATE2_RANDOM_SEEDS = 20
_GATE2_STRUCTURAL_SEEDS = 100
_GATE2_FOLDS = 5
_GATE2_MAX_ORDER = 3


def test_read_fasta_sequence_concatenates_body_and_drops_header(tmp_path: Path) -> None:
    f = tmp_path / "wt.fasta"
    f.write_text(">gb1 B1 domain\nMTYKLILNGK\nTLKGETTTEA\n", encoding="utf-8")
    assert read_fasta_sequence(f) == "MTYKLILNGKTLKGETTTEA"


def test_read_fasta_sequence_rejects_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.fasta"
    f.write_text(">only a header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no sequence"):
        read_fasta_sequence(f)


def test_parse_variant_single_and_multi() -> None:
    # 1-indexed DMS notation; V39 is 0-indexed position 38 in the GB1 WT.
    v = parse_variant("V39A", GB1_WT_SEQUENCE)
    assert v == frozenset({(38, "V", "A")})
    v2 = parse_variant("V39A D40C", GB1_WT_SEQUENCE)
    assert v2 == frozenset({(38, "V", "A"), (39, "D", "C")})


def test_parse_variant_accepts_common_separators() -> None:
    expected = frozenset({(38, "V", "A"), (39, "D", "C")})
    for spec in ("V39A;D40C", "V39A,D40C", "V39A+D40C", "V39A  D40C"):
        assert parse_variant(spec, GB1_WT_SEQUENCE) == expected


def test_parse_variant_empty_or_wt_is_wild_type() -> None:
    assert parse_variant("", GB1_WT_SEQUENCE) == frozenset()
    assert parse_variant("WT", GB1_WT_SEQUENCE) == frozenset()


def test_parse_variant_rejects_wt_letter_mismatch() -> None:
    # WT at position 39 (1-indexed) is V, not Q.
    with pytest.raises(ValueError, match="WT mismatch"):
        parse_variant("Q39A", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_synonymous() -> None:
    with pytest.raises(ValueError, match="synonymous"):
        parse_variant("V39V", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_out_of_range_position() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_variant("V999A", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_unknown_residue() -> None:
    with pytest.raises(ValueError, match="not a valid amino acid"):
        parse_variant("V39Z", GB1_WT_SEQUENCE)


def test_parse_variant_rejects_malformed_token() -> None:
    with pytest.raises(ValueError, match="malformed"):
        parse_variant("hello", GB1_WT_SEQUENCE)


def test_read_variant_specs_with_header(tmp_path: Path) -> None:
    f = tmp_path / "variants.csv"
    f.write_text("variant\nV39A\nV39A D40C\n\n", encoding="utf-8")
    assert read_variant_specs(f) == ["V39A", "V39A D40C"]


def test_read_variant_specs_headerless(tmp_path: Path) -> None:
    f = tmp_path / "variants.csv"
    f.write_text("V39A\nD40C G41W\n", encoding="utf-8")
    assert read_variant_specs(f) == ["V39A", "D40C G41W"]


class _FakeScorer:
    """Stand-in for ConjointScorer: no ESM-2 forward pass, deterministic stub scores."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        n_perturbations: int = 16,
        seed: int = 0,
        batch_size: int = 32,
        num_threads: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.n_perturbations = n_perturbations
        self.seed = seed
        self.mask_fraction = 0.15
        self.batch_size = batch_size
        self.num_threads = num_threads

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        return [
            ScoredVariant(variant=v, delta_g=0.1 * i, var_delta_g=0.01)
            for i, v in enumerate(variants)
        ]


class _AdditiveScorer(_FakeScorer):
    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        return [
            ScoredVariant(
                variant=variant,
                delta_g=sum(
                    0.1 * (GB1_SITES.index(position) + 1) for position, _wt, _mut in variant
                ),
                var_delta_g=0.01,
            )
            for variant in variants
        ]


class _ConstantPairEpistasisScorer(_FakeScorer):
    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        return [
            ScoredVariant(
                variant=variant,
                delta_g=sum(float(GB1_SITES.index(position) + 1) for position, _wt, _mut in variant)
                + (1.0 if len(variant) == _PAIR_ORDER else 0.0),
                var_delta_g=0.01,
            )
            for variant in variants
        ]


def test_score_command_runs_offline_with_a_stubbed_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(f">gb1\n{GB1_WT_SEQUENCE}\n", encoding="utf-8")
    variants_file = tmp_path / "variants.csv"
    variants_file.write_text("V39A\nV39A D40C\n", encoding="utf-8")

    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)

    result = CliRunner().invoke(
        app, ["score", "--fasta", str(fasta), "--variants", str(variants_file)]
    )
    assert result.exit_code == 0, result.output
    assert "single" in result.output  # one row per spec, labelled by order
    assert "double" in result.output


def test_allocate_command_runs_offline_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)
    fasta = tmp_path / "wt.fasta"
    fasta.write_text(">toy\nADG\n", encoding="utf-8")  # WT residues A/D/G at positions 1..3
    out = tmp_path / "allocation.json"

    result = CliRunner().invoke(
        app,
        [
            "allocate",
            "--fasta",
            str(fasta),
            "--positions",
            "1,2,3",
            "--budget",
            "4",
            "--alphabet",
            "ACG",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["budget"] == _ALLOC_BUDGET
    assert len(written["selected"]) == _ALLOC_BUDGET


def _write_gb1_csv(path: Path, variants: list[Variant]) -> None:
    """Write a minimal GB1-format CSV (protein sequence, label) with a non-additive landscape."""

    def true_dg(variant: Variant) -> float:
        sites = {pos for pos, _, _ in variant}
        value = 0.5 * len(sites)
        if {GB1_SITES[0], GB1_SITES[1]} <= sites:
            value += 0.8  # a genuine pairwise interaction ⇒ Var[ε] > 0
        return value

    rows = [(GB1_WT_SEQUENCE, 1.0)]
    rows += [(apply_mutations(GB1_WT_SEQUENCE, v), exp(true_dg(v))) for v in variants]
    lines = ["protein,label"] + [f"{seq},{label}" for seq, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_flat_gb1_csv(path: Path, variants: list[Variant]) -> None:
    rows = [(GB1_WT_SEQUENCE, 1.0)]
    rows += [(apply_mutations(GB1_WT_SEQUENCE, variant), 1.0) for variant in variants]
    lines = ["protein,label"] + [f"{seq},{label}" for seq, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_command_runs_offline_and_writes_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 30)  # keep the offline run snappy

    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    scored_cache = tmp_path / "scored.jsonl"

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--seeds",
            "2",
            "--out",
            str(tmp_path / "report"),
            "--scored-cache",
            str(scored_cache),
        ],
    )
    assert result.exit_code == 0, result.output
    runs = list((tmp_path / "report").iterdir())
    assert len(runs) == 1  # one run directory created
    written = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert written["var_epsilon"] > 0.0
    assert written["candidate_alphabet"] == "AC"
    cache_metadata = json.loads(
        scored_cache.with_name("scored.jsonl.meta.json").read_text(encoding="utf-8")
    )
    assert cache_metadata["candidate_count"] == len(candidates)
    assert cache_metadata["candidate_alphabet"] == "AC"
    assert {r["method"] for r in written["results"]} == {
        "info",
        "fitness",
        "structural",
        "random",
        "practice",
    }


def test_validate_cli_fails_invariant_gate_for_additive_esm_even_when_truth_is_nonadditive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scoring, "ConjointScorer", _AdditiveScorer)
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 10)
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "nonadditive.csv"
    _write_gb1_csv(csv, candidates)

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--seeds",
            "2",
            "--out",
            str(tmp_path / "report"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Predicted Var[ε] = 0.0000  [FAIL invariant #1]" in result.output
    assert "tolerance=" in result.output
    assert "Truth Var[ε]" in result.output
    assert "pooled (diagnostic only" in result.output
    run_dir = next((tmp_path / "report").iterdir())
    written = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert written["var_predicted_epsilon"] > 0.0
    assert written["predicted_epistasis_signal"] is False
    assert written["predicted_epistasis_tolerance"] > 0.0
    assert written["var_epsilon"] > 0.0


def test_validate_cli_passes_invariant_gate_for_constant_nonzero_predicted_epistasis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scoring, "ConjointScorer", _ConstantPairEpistasisScorer)
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 10)
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=2)
    csv = tmp_path / "additive.csv"
    _write_flat_gb1_csv(csv, candidates)

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--max-order",
            "2",
            "--seeds",
            "2",
            "--out",
            str(tmp_path / "report"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "[PASS invariant #1]" in result.output
    assert "Truth Var[ε]" in result.output
    run_dir = next((tmp_path / "report").iterdir())
    written = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert written["var_predicted_epsilon"] == 0.0
    assert written["predicted_epistasis_signal"] is True
    assert written["var_epsilon"] == 0.0


# ---------------------------------------------------------- native dataset routing


def _write_landscape_csv(
    path: Path, wt_sequence: str, sites: tuple[int, ...], variants: list[Variant]
) -> None:
    """Write a minimal (protein, label) CSV for any four-site landscape, non-additively coupled.

    Same construction as ``_write_gb1_csv`` but parameterised on the reference sequence and sites,
    so a TrpB fixture can be built without pointing the GB1 helper at a foreign reference.
    """

    def true_dg(variant: Variant) -> float:
        positions = {pos for pos, _, _ in variant}
        value = 0.5 * len(positions)
        if {sites[0], sites[1]} <= positions:
            value += 0.8  # a genuine pairwise interaction ⇒ Var[ε] > 0
        return value

    rows = [(wt_sequence, 1.0)]
    rows += [(apply_mutations(wt_sequence, v), exp(true_dg(v))) for v in variants]
    lines = ["protein,label"] + [f"{seq},{label}" for seq, label in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_run_validation_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Wrap ``run_validation`` with a call counter, patched at the module attribute the CLI's
    deferred ``from epibudget.validate import run_validation`` reads at call time."""
    from epibudget import validate as validate_module  # noqa: PLC0415

    real_run_validation = validate_module.run_validation
    calls = {"n": 0}

    def _counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_run_validation(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(validate_module, "run_validation", _counting)
    return calls


def test_validate_gb1_records_gb1_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GB1 stamps the GB1 identifier, sites, and WT reference into provenance."""
    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 30)
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--dataset",
            "gb1_wu2016",
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--seeds",
            "2",
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    runs = list((tmp_path / "report").iterdir())
    assert len(runs) == 1
    written = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert written["dataset"] == "gb1_wu2016"
    assert written["sites"] == list(GB1_SITES)
    assert written["wt_sha256"] == hashlib.sha256(GB1_WT_SEQUENCE.encode("ascii")).hexdigest()


def test_validate_trpb_records_trpb_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TrpB stamps the TrpB identifier, sites, and Tm9D8* WT reference into provenance."""
    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 30)
    candidates = enumerate_candidates(TRPB_SITES, TRPB_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "trpb.csv"
    _write_landscape_csv(csv, TRPB_WT_SEQUENCE, TRPB_SITES, candidates)

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--dataset",
            "trpb_johnston2024",
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--seeds",
            "2",
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    runs = list((tmp_path / "report").iterdir())
    assert len(runs) == 1
    written = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert written["dataset"] == "trpb_johnston2024"
    assert written["sites"] == list(TRPB_SITES)
    assert written["wt_sha256"] == hashlib.sha256(TRPB_WT_SEQUENCE.encode("ascii")).hexdigest()


def test_validate_rejects_unknown_dataset_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unregistered identifier fails as a bad parameter, before any landscape load or scoring,
    and writes no report at all."""
    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)
    calls = _count_run_validation_calls(monkeypatch)
    report_root = tmp_path / "report"

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--dataset",
            "does_not_exist",
            "--data",
            str(tmp_path / "missing.csv"),
            "--out",
            str(report_root),
        ],
    )
    assert result.exit_code != 0
    assert "unknown dataset" in result.output
    assert calls["n"] == 0  # validation never started
    assert not report_root.exists()  # nothing was written


def test_validate_trpb_csv_not_loaded_via_load_gb1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TrpB run must route the CSV through ``load_trpb``, never ``load_gb1``.

    Proven two ways: ``load_gb1`` outright rejects the TrpB CSV (its sequences are not
    length-compatible with the GB1 reference), and a spy on the registry's TrpB loader confirms the
    CLI sent exactly this CSV to ``load_trpb``.
    """
    monkeypatch.setattr(scoring, "ConjointScorer", _FakeScorer)
    monkeypatch.setattr("epibudget.validate._N_BOOTSTRAP", 30)
    candidates = enumerate_candidates(TRPB_SITES, TRPB_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "trpb.csv"
    _write_landscape_csv(csv, TRPB_WT_SEQUENCE, TRPB_SITES, candidates)

    # If the CLI wrongly routed the TrpB CSV through the GB1 loader, this is the error it would hit.
    with pytest.raises(ValueError, match="length"):
        load_gb1(csv)

    load_trpb_paths: list[Path] = []
    real_load_trpb = data_module.load_trpb

    def _spy_load_trpb(path: Path) -> dict[Variant, float]:
        load_trpb_paths.append(path)
        return real_load_trpb(path)

    trpb_spec = data_module.DATASETS["trpb_johnston2024"]
    monkeypatch.setitem(
        data_module.DATASETS,
        "trpb_johnston2024",
        dataclasses.replace(trpb_spec, loader=_spy_load_trpb),
    )

    result = CliRunner().invoke(
        app,
        [
            "validate",
            "--dataset",
            "trpb_johnston2024",
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--seeds",
            "2",
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    # load_trpb ran exactly once, on the TrpB CSV — the GB1 loader was never in the path.
    assert load_trpb_paths == [csv]


def _build_scored_cache(
    path: Path,
    variants_written: list[Variant],
    universe: list[Variant],
    *,
    alphabet: str,
    max_order: int = 3,
) -> None:
    """Write a JSONL scored cache plus a metadata sidecar declaring ``universe``'s identity.

    ``variants_written`` may be a strict subset of ``universe`` to simulate a truncated run: the
    sidecar still declares the full intended universe, so ``validate_cache_against_universe``
    rejects it as missing entries rather than as a sidecar/universe mismatch.
    """
    from epibudget.provenance import write_json_exclusive  # noqa: PLC0415
    from epibudget.scored_cache import (  # noqa: PLC0415
        CacheMetadata,
        append_cache,
        cache_metadata_path,
        candidate_sha256,
    )

    scored = [
        ScoredVariant(variant=v, delta_g=float(i) - 5.0, var_delta_g=0.1 + 0.01 * i)
        for i, v in enumerate(variants_written)
    ]
    append_cache(path, scored)
    metadata = CacheMetadata(
        # Matches the confirmatory identity the CLI checks against: a
        # mismatch here would make `robustness`/`downstream` reject this fixture before running.
        model_id=_CONFIRMATORY_MODEL_ID,
        wt_sha256=hashlib.sha256(GB1_WT_SEQUENCE.encode("ascii")).hexdigest(),
        candidate_sha256=candidate_sha256(universe),
        candidate_count=len(universe),
        candidate_alphabet=alphabet,
        max_order=max_order,
        scorer_seed=_CONFIRMATORY_SCORER_SEED,
        n_perturbations=_CONFIRMATORY_N_PERTURBATIONS,
        device="cpu",
        mask_fraction=0.15,
        batch_size=32,
        num_threads=None,
    )
    write_json_exclusive(cache_metadata_path(path), metadata.model_dump(mode="json"))


def _independent_candidate_sha256(candidates: list[Variant]) -> str:
    """Recompute the candidate-universe hash from raw fixture data.

    Deliberately duplicates ``scored_cache.candidate_sha256``'s documented canonicalization
    (sorted nested lists, ``json.dumps`` with ``(",", ":")`` separators, ASCII, SHA-256 hex) rather
    than importing and calling it, so an exact-equality assertion built from this helper would
    still catch a hashing bug in the production function itself.
    """
    canonical = sorted([sorted([list(mutation) for mutation in v]) for v in candidates])
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _valid_sidecar_payload(
    candidates: list[Variant], *, alphabet: str, max_order: int
) -> dict[str, object]:
    """The exact sidecar JSON a cache matching the frozen confirmatory identity would declare."""
    from epibudget.scored_cache import candidate_sha256  # noqa: PLC0415

    metadata = CacheMetadata(
        model_id=_CONFIRMATORY_MODEL_ID,
        wt_sha256=hashlib.sha256(GB1_WT_SEQUENCE.encode("ascii")).hexdigest(),
        candidate_sha256=candidate_sha256(candidates),
        candidate_count=len(candidates),
        candidate_alphabet=alphabet,
        max_order=max_order,
        scorer_seed=_CONFIRMATORY_SCORER_SEED,
        n_perturbations=_CONFIRMATORY_N_PERTURBATIONS,
        device="cpu",
        mask_fraction=0.15,
        batch_size=32,
        num_threads=None,
    )
    return metadata.model_dump(mode="json")


def _write_cache_with_sidecar_payload(
    path: Path, candidates: list[Variant], sidecar_payload: dict[str, object]
) -> None:
    """Write a JSONL cache for ``candidates`` plus a caller-supplied (possibly mutated) sidecar."""
    from epibudget.provenance import write_json_exclusive  # noqa: PLC0415
    from epibudget.scored_cache import append_cache  # noqa: PLC0415

    scored = [
        ScoredVariant(variant=v, delta_g=float(i) - 5.0, var_delta_g=0.1 + 0.01 * i)
        for i, v in enumerate(candidates)
    ]
    append_cache(path, scored)
    write_json_exclusive(cache_metadata_path(path), sidecar_payload)


def test_robustness_command_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("epibudget.robustness._N_BOOTSTRAP", 30)  # keep the offline run snappy
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet="AC")

    result = CliRunner().invoke(
        app,
        [
            "robustness",
            "--scored-cache",
            str(cache),
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--seeds",
            "2",
            "--n-folds",
            "3",
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    runs = list((tmp_path / "report").iterdir())
    assert len(runs) == 1
    report = json.loads((runs[0] / "robustness.json").read_text(encoding="utf-8"))
    assert report["n_candidates"] == len(candidates)
    assert "does not alter the frozen decision rule" in report["note"]
    assert report["scale_sensitivity"]  # at least the pairwise order was analysed


def test_robustness_command_writes_complete_cache_identity_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``robustness`` is descriptive/non-decision-bearing, but must still record the same
    8-field expected/observed cache identity that ``validate_cache_against_universe`` checked."""
    monkeypatch.setattr("epibudget.robustness._N_BOOTSTRAP", 30)
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet="AC")

    result = CliRunner().invoke(
        app,
        [
            "robustness",
            "--scored-cache",
            str(cache),
            "--data",
            str(csv),
            "--alphabet",
            "AC",
            "--budgets",
            "4",
            "--seeds",
            "2",
            "--n-folds",
            "3",
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    runs = list((tmp_path / "report").iterdir())
    identity_report = json.loads(
        (runs[0] / "robustness_cache_identity.json").read_text(encoding="utf-8")
    )

    expected_identity = {
        "model_id": _CONFIRMATORY_MODEL_ID,
        "scorer_seed": _CONFIRMATORY_SCORER_SEED,
        "n_perturbations": _CONFIRMATORY_N_PERTURBATIONS,
        "candidate_sha256": _independent_candidate_sha256(candidates),
        "candidate_count": len(candidates),
        "candidate_alphabet": "AC",
        "max_order": 3,
        "wt_sha256": hashlib.sha256(GB1_WT_SEQUENCE.encode("ascii")).hexdigest(),
    }
    assert identity_report["scored_cache_identity_expected"] == expected_identity
    assert identity_report["scored_cache_identity_observed"] == expected_identity


def test_robustness_command_rejects_a_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("epibudget.robustness._N_BOOTSTRAP", 30)
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="AC", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates[:-1], candidates, alphabet="AC")  # one row short

    result = CliRunner().invoke(
        app,
        ["robustness", "--scored-cache", str(cache), "--data", str(csv), "--alphabet", "AC"],
    )
    assert result.exit_code != 0
    assert "missing" in result.output


def test_downstream_command_writes_provisional_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 3-letter alphabet gives enough doubles/triples per fold for the order-stratified metrics.
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet="ACD")

    result = CliRunner().invoke(
        app,
        [
            "downstream",
            "--scored-cache",
            str(cache),
            "--data",
            str(csv),
            "--alphabet",
            "ACD",
            "--budgets",
            "8,12",
            "--seeds",
            "2",
            "--n-folds",
            "2",
            "--partitions",
            "1",
            "--headline",
            str(csv),
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    runs = list((tmp_path / "report").iterdir())
    assert len(runs) == 1
    report = json.loads((runs[0] / "downstream.json").read_text(encoding="utf-8"))
    assert report["n_candidates"] == len(candidates)
    assert "does not alter the frozen" in report["note"]
    assert {s["method"] for s in report["method_budget"]} == {
        "info",
        "fitness",
        "structural",
        "random",
        "practice",
    }
    assert report["provenance"]["status"] == "provisional"
    assert "structural_downstream_supported" in report["decision"]
    # this fixture's toy alphabet/budgets never match the frozen
    # confirmatory profile, so both the CLI-boundary and decision-layer checks must flag it.
    assert report["provenance"]["cli_protocol_profile_conforming"] is False
    assert "alphabet" in report["provenance"]["cli_protocol_profile_mismatches"]
    assert report["decision"]["protocol_profile_conforming"] is False
    assert report["decision"]["structural_gate"]["status"] == "nonconforming_protocol_profile"


def test_downstream_command_captures_completed_at_utc_after_the_report_call_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``completed_at_utc`` must reflect completion of the benchmark itself,
    not the moment the provenance object was constructed."""
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet="ACD")

    real_downstream_report = downstream_module.downstream_report
    delay_seconds = 0.2

    def _delayed_downstream_report(*args: object, **kwargs: object) -> object:
        result = real_downstream_report(*args, **kwargs)  # type: ignore[arg-type]
        time.sleep(delay_seconds)  # stand in for the benchmark computation itself taking time
        return result

    monkeypatch.setattr(downstream_module, "downstream_report", _delayed_downstream_report)

    result = CliRunner().invoke(
        app,
        [
            "downstream",
            "--scored-cache",
            str(cache),
            "--data",
            str(csv),
            "--alphabet",
            "ACD",
            "--budgets",
            "8,12",
            "--seeds",
            "2",
            "--n-folds",
            "2",
            "--partitions",
            "1",
            "--headline",
            str(csv),
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 0, result.output
    runs = list((tmp_path / "report").iterdir())
    report = json.loads((runs[0] / "downstream.json").read_text(encoding="utf-8"))
    provenance = report["provenance"]
    started = datetime.fromisoformat(provenance["started_at_utc"].replace("Z", "+00:00"))
    completed = datetime.fromisoformat(provenance["completed_at_utc"].replace("Z", "+00:00"))
    assert started <= completed
    # completed_at_utc was captured no earlier than the (delayed) report call's own return.
    assert (completed - started).total_seconds() >= delay_seconds


def test_downstream_command_rejects_a_partial_cache(tmp_path: Path) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates[:-1], candidates, alphabet="ACD")  # one row short

    result = CliRunner().invoke(
        app,
        ["downstream", "--scored-cache", str(cache), "--data", str(csv), "--alphabet", "ACD"],
    )
    assert result.exit_code != 0
    assert "missing" in result.output


# ---------------------------------------------------------- complete cache identity


def _downstream_cli_args(cache: Path, csv: Path, out_dir: Path) -> list[str]:
    return [
        "downstream",
        "--scored-cache",
        str(cache),
        "--data",
        str(csv),
        "--alphabet",
        "ACD",
        "--budgets",
        "8,12",
        "--seeds",
        "2",
        "--n-folds",
        "2",
        "--partitions",
        "1",
        "--headline",
        str(csv),
        "--out",
        str(out_dir),
    ]


def _count_downstream_report_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Wrap the real ``downstream_report`` with a call counter, patched in at the module attribute
    ``cli.py``'s deferred ``from epibudget.downstream import downstream_report`` reads at call
    time (the same pattern as the completion-timestamp test above)."""
    real_downstream_report = downstream_module.downstream_report
    calls = {"n": 0}

    def _counting(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real_downstream_report(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(downstream_module, "downstream_report", _counting)
    return calls


def test_downstream_rejects_unknown_dataset_before_any_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unregistered ``--dataset`` fails as a bad parameter before any cache/landscape load, so
    the benchmark never runs (mirrors the ``validate`` guard, now that ``downstream`` is
    dataset-generic)."""
    calls = _count_downstream_report_calls(monkeypatch)
    result = CliRunner().invoke(
        app,
        [
            "downstream",
            "--scored-cache",
            str(tmp_path / "missing.jsonl"),
            "--dataset",
            "does_not_exist",
            "--data",
            str(tmp_path / "missing.csv"),
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown dataset" in result.output
    assert calls["n"] == 0  # benchmark never started


def test_downstream_command_serializes_complete_cache_identity_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a valid run's provenance must serialize all 8 identity fields on both the
    ``expected`` and ``observed`` side, with values matching the fixture's known raw inputs
    (not re-derived from the report's own numbers)."""
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet="ACD")

    calls = _count_downstream_report_calls(monkeypatch)
    result = CliRunner().invoke(app, _downstream_cli_args(cache, csv, tmp_path / "report"))

    assert result.exit_code == 0, result.output
    assert calls["n"] == 1
    runs = list((tmp_path / "report").iterdir())
    assert len(runs) == 1
    report = json.loads((runs[0] / "downstream.json").read_text(encoding="utf-8"))

    expected_identity = {
        "model_id": _CONFIRMATORY_MODEL_ID,
        "scorer_seed": _CONFIRMATORY_SCORER_SEED,
        "n_perturbations": _CONFIRMATORY_N_PERTURBATIONS,
        "candidate_sha256": _independent_candidate_sha256(candidates),
        "candidate_count": len(candidates),
        "candidate_alphabet": "ACD",
        "max_order": 3,
        "wt_sha256": hashlib.sha256(GB1_WT_SEQUENCE.encode("ascii")).hexdigest(),
    }
    # The fixture cache's sidecar declares exactly the confirmatory identity, so observed and
    # expected coincide here; a mismatch in either would fail this exact-equality check.
    assert report["provenance"]["scored_cache_identity_expected"] == expected_identity
    assert report["provenance"]["scored_cache_identity_observed"] == expected_identity


def _wrong(field: str, value: object) -> Callable[[dict[str, object]], None]:
    def _apply(payload: dict[str, object]) -> None:
        payload[field] = value

    return _apply


def _missing(field: str) -> Callable[[dict[str, object]], None]:
    def _apply(payload: dict[str, object]) -> None:
        del payload[field]

    return _apply


_IDENTITY_MUTATION_CASES: list[tuple[str, Callable[[dict[str, object]], None]]] = [
    ("wrong_model_id", _wrong("model_id", "facebook/esm2_t30_150M_UR50D")),
    ("wrong_scorer_seed", _wrong("scorer_seed", _CONFIRMATORY_SCORER_SEED + 1)),
    ("wrong_n_perturbations", _wrong("n_perturbations", _CONFIRMATORY_N_PERTURBATIONS + 1)),
    ("wrong_candidate_sha256", _wrong("candidate_sha256", "0" * 64)),
    ("wrong_candidate_count", _wrong("candidate_count", -1)),
    ("wrong_candidate_alphabet", _wrong("candidate_alphabet", "ZZZZ")),
    ("wrong_max_order", _wrong("max_order", 1)),
    ("wrong_wt_sha256", _wrong("wt_sha256", "f" * 64)),
    ("missing_required_identity_field", _missing("model_id")),
    ("malformed_identity_type", _wrong("scorer_seed", "not-an-int")),
]


@pytest.mark.parametrize(
    "mutate",
    [case[1] for case in _IDENTITY_MUTATION_CASES],
    ids=[c[0] for c in _IDENTITY_MUTATION_CASES],
)
def test_downstream_command_rejects_cache_identity_mutations_before_computing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Callable[[dict[str, object]], None]
) -> None:
    """mutation matrix: any single-field cache-identity mismatch (wrong value, missing
    field, or malformed type) must reject before ``downstream_report`` is ever called, must write
    no ``downstream.json``, and must fail loudly (non-zero exit) rather than warn-and-continue."""
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    payload = _valid_sidecar_payload(candidates, alphabet="ACD", max_order=3)
    mutate(payload)
    _write_cache_with_sidecar_payload(cache, candidates, payload)

    calls = _count_downstream_report_calls(monkeypatch)
    report_root = tmp_path / "report"
    result = CliRunner().invoke(app, _downstream_cli_args(cache, csv, report_root))

    assert result.exit_code != 0, result.output
    assert result.exception is not None
    assert calls["n"] == 0  # downstream_report was never reached
    assert not report_root.exists() or list(report_root.iterdir()) == []  # no downstream.json


class _StubGate2Decision:
    def __init__(self, decision: str) -> None:
        self.decision = decision


class _StubGate2Report:
    """Small report double: Gate 2 computation is covered by ``test_gate2.py``."""

    def __init__(
        self,
        provenance: dict[str, object],
        *,
        architecture_decision_eligible: bool = False,
        decision: str = "inconclusive_zero_gpu",
    ) -> None:
        self.status = "provisional" if provenance["code_state"] == "dirty" else "final"
        self.public_claim_eligible = False
        self.architecture_decision_eligible = architecture_decision_eligible
        self.architecture_eligibility_reasons = ["synthetic CLI fixture"]
        self.decision = _StubGate2Decision(decision)
        self.provenance = provenance

    def model_copy(self, *, update: dict[str, object]) -> _StubGate2Report:
        copied = _StubGate2Report(self.provenance)
        copied.__dict__.update(self.__dict__)
        copied.__dict__.update(update)
        return copied

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "protocol_version": "gate2-v1",
            "run_type": "post_hoc_corrective_gate2",
            "status": self.status,
            "public_claim_eligible": self.public_claim_eligible,
            "architecture_decision_eligible": self.architecture_decision_eligible,
            "architecture_eligibility_reasons": self.architecture_eligibility_reasons,
            "decision": {
                "decision": self.decision.decision,
                "architecture_decision_eligible": self.architecture_decision_eligible,
            },
            "provenance": self.provenance,
        }


def _gate2_cli_args(cache: Path, csv: Path, out_dir: Path) -> list[str]:
    return [
        "gate2",
        "--scored-cache",
        str(cache),
        "--data",
        str(csv),
        "--out",
        str(out_dir),
    ]


def _patch_small_gate2_universe(monkeypatch: pytest.MonkeyPatch, candidates: list[Variant]) -> None:
    monkeypatch.setattr(data_module, "enumerate_candidates", lambda *args, **kwargs: candidates)


def _patch_stub_gate2_report(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], list[str]]:
    captured: dict[str, object] = {}
    events: list[str] = []

    def _stub_report(*args: object, **kwargs: object) -> _StubGate2Report:
        events.append("gate2_report")
        captured["args"] = args
        captured["kwargs"] = kwargs
        provenance = kwargs["provenance"]
        assert isinstance(provenance, dict)
        return _StubGate2Report(provenance)

    def _stub_finalize(
        report: _StubGate2Report,
        scored: Sequence[ScoredVariant],
        provenance: dict[str, object],
    ) -> _StubGate2Report:
        del scored
        events.append("finalize_gate2_report")
        return report.model_copy(update={"provenance": provenance})

    monkeypatch.setattr(gate2_module, "gate2_report", _stub_report)
    monkeypatch.setattr(gate2_module, "finalize_gate2_report", _stub_finalize)
    return captured, events


def _patch_dirty_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_git_lines(repo: Path, *args: str) -> list[str]:
        del repo
        if args == ("rev-parse", "HEAD"):
            return ["a" * 40]
        if args == ("status", "--porcelain"):
            return [" M src/epibudget/cli.py"]
        return []

    monkeypatch.setattr(cli_module, "_gate2_required_git_lines", _fake_git_lines)
    monkeypatch.setattr(
        provenance_module, "workspace_code_diff_sha256", lambda repo, commit: "b" * 64
    )
    monkeypatch.setattr(
        provenance_module,
        "changed_scientific_files",
        lambda repo, commit: ["src/epibudget/cli.py", "tests/test_cli.py"],
    )


def _assert_gate2_default_call(captured: dict[str, object]) -> None:
    kwargs = captured["kwargs"]
    args = captured["args"]
    assert isinstance(kwargs, dict)
    assert isinstance(args, tuple)
    assert args[2] == _GATE2_BUDGETS
    assert kwargs["random_seeds"] == _GATE2_RANDOM_SEEDS
    assert kwargs["structural_seeds"] == _GATE2_STRUCTURAL_SEEDS
    assert kwargs["n_folds"] == _GATE2_FOLDS
    assert kwargs["max_order"] == _GATE2_MAX_ORDER
    assert kwargs["alphabet"] == _AA20
    assert kwargs["dataset"] == "gb1_wu2016"
    assert kwargs["model_id"] == _CONFIRMATORY_MODEL_ID


def _assert_gate2_provenance(
    written: dict[str, object],
    csv: Path,
    cache: Path,
    candidates: list[Variant],
    out_dir: Path,
) -> None:
    provenance = written["provenance"]
    assert isinstance(provenance, dict)
    expected_identity = {
        "model_id": _CONFIRMATORY_MODEL_ID,
        "scorer_seed": _CONFIRMATORY_SCORER_SEED,
        "n_perturbations": _CONFIRMATORY_N_PERTURBATIONS,
        "candidate_sha256": _independent_candidate_sha256(candidates),
        "candidate_count": len(candidates),
        "candidate_alphabet": _AA20,
        "max_order": _GATE2_MAX_ORDER,
        "wt_sha256": hashlib.sha256(GB1_WT_SEQUENCE.encode("ascii")).hexdigest(),
    }
    assert provenance["protocol_version"] == "gate2-v1"
    assert provenance["run_type"] == "post_hoc_corrective_gate2"
    assert provenance["scored_cache_validator_status"] == "passed"
    assert provenance["scored_cache_identity_expected"] == expected_identity
    assert provenance["scored_cache_identity_observed"] == expected_identity
    assert provenance["candidate_universe_sha256"] == expected_identity["candidate_sha256"]
    assert provenance["execution_commit"] == "a" * 40
    assert provenance["code_state"] == "dirty"
    assert provenance["code_diff_sha256"] == "b" * 64
    assert provenance["changed_scientific_files"] == [
        "src/epibudget/cli.py",
        "tests/test_cli.py",
    ]
    expected_argv = [
        "epibudget",
        "gate2",
        "--scored-cache",
        str(cache),
        "--data",
        str(csv),
        "--out",
        str(out_dir),
    ]
    actual_argv = provenance["argv"]
    exact_command = provenance["exact_command"]
    assert isinstance(actual_argv, list)
    assert isinstance(exact_command, str)
    assert actual_argv == expected_argv
    assert exact_command == subprocess.list2cmdline(expected_argv)
    assert '"' in exact_command
    for omitted_default in (
        "--alphabet",
        "--budgets",
        "--random-seeds",
        "--structural-seeds",
        "--n-folds",
        "--max-order",
    ):
        assert omitted_default not in actual_argv
    assert provenance["dataset_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert provenance["scored_cache_sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert (
        provenance["scored_cache_sidecar_sha256"]
        == hashlib.sha256(cache_metadata_path(cache).read_bytes()).hexdigest()
    )
    started = datetime.fromisoformat(str(provenance["started_at_utc"]).replace("Z", "+00:00"))
    completed = datetime.fromisoformat(str(provenance["completed_at_utc"]).replace("Z", "+00:00"))
    assert started <= completed
    assert isinstance(provenance["elapsed_seconds"], float)
    assert provenance["elapsed_seconds"] >= 0.0


def test_gate2_help_exposes_the_frozen_cache_only_profile() -> None:
    result = CliRunner().invoke(app, ["gate2", "--help"], terminal_width=200)

    assert result.exit_code == 0, result.output
    for option in (
        "--scored-cache",
        "--data",
        "--alphabet",
        "--budgets",
        "--random-seeds",
        "--structural-seeds",
        "--n-folds",
        "--max-order",
        "--out",
    ):
        assert option in result.output
    for default in ("48,96,192", "20", "100"):
        assert default in result.output
    parameters = inspect.signature(cli_module.gate2).parameters
    assert parameters["scored_cache"].default.default == "report/scored_650m.jsonl"
    assert parameters["data"].default.default == "data/proteingym/gb1_wu2016.csv"
    assert parameters["alphabet"].default.default == _AA20
    assert parameters["budgets"].default.default == "48,96,192"
    assert parameters["random_seeds"].default.default == _GATE2_RANDOM_SEEDS
    assert parameters["structural_seeds"].default.default == _GATE2_STRUCTURAL_SEEDS
    assert parameters["n_folds"].default.default == _GATE2_FOLDS
    assert parameters["max_order"].default.default == _GATE2_MAX_ORDER
    assert parameters["out"].default.default == "report/"
    assert "device" not in result.output.lower()


def test_gate2_command_validates_cache_before_labels_and_writes_complete_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    input_dir = tmp_path / "gate 2 inputs"
    input_dir.mkdir()
    csv = input_dir / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = input_dir / "scored.jsonl"
    report_root = tmp_path / "gate 2 report"
    _build_scored_cache(cache, candidates, candidates, alphabet=_AA20)
    _patch_small_gate2_universe(monkeypatch, candidates)
    captured, events = _patch_stub_gate2_report(monkeypatch)
    _patch_dirty_git(monkeypatch)

    real_validate = scored_cache_module.validate_cache_against_universe
    real_load_gb1 = data_module.load_gb1

    def _validate(*args: object, **kwargs: object) -> object:
        events.append("validate_cache")
        return real_validate(*args, **kwargs)  # type: ignore[arg-type]

    def _load(path: Path) -> dict[Variant, float]:
        assert events == ["validate_cache"]
        events.append("load_labels")
        return real_load_gb1(path)

    monkeypatch.setattr(scored_cache_module, "validate_cache_against_universe", _validate)
    monkeypatch.setattr(data_module, "load_gb1", _load)

    class _ForbiddenScorer:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("Gate 2 must never instantiate a scorer")

    monkeypatch.setattr(scoring, "ConjointScorer", _ForbiddenScorer)

    result = CliRunner().invoke(app, _gate2_cli_args(cache, csv, report_root))

    assert result.exit_code == 0, result.output
    assert events == [
        "validate_cache",
        "load_labels",
        "gate2_report",
        "finalize_gate2_report",
    ]
    _assert_gate2_default_call(captured)

    runs = list(report_root.iterdir())
    assert len(runs) == 1
    report_path = runs[0] / "gate2.json"
    written = json.loads(report_path.read_text(encoding="utf-8"))
    _assert_gate2_provenance(written, csv, cache, candidates, report_root)
    assert written["status"] == "provisional"
    assert written["public_claim_eligible"] is False
    assert "public_claim_eligible=False" in result.output
    assert str(report_path) in result.output.replace("\n", "")


def test_gate2_command_rejects_partial_cache_before_loading_labels_or_running_gate2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates[:-1], candidates, alphabet=_AA20)
    _patch_small_gate2_universe(monkeypatch, candidates)
    calls = {"labels": 0, "report": 0}
    monkeypatch.setattr(
        data_module,
        "load_gb1",
        lambda path: calls.__setitem__("labels", calls["labels"] + 1),
    )
    monkeypatch.setattr(
        gate2_module,
        "gate2_report",
        lambda *args, **kwargs: calls.__setitem__("report", calls["report"] + 1),
    )

    result = CliRunner().invoke(
        app, _gate2_cli_args(cache, csv, tmp_path / "report"), terminal_width=300
    )

    assert result.exit_code != 0
    assert "missing" in result.output
    assert calls == {"labels": 0, "report": 0}


def test_gate2_command_rejects_missing_sidecar_before_running_gate2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet=_AA20)
    cache_metadata_path(cache).unlink()
    _patch_small_gate2_universe(monkeypatch, candidates)
    calls = {"report": 0}
    monkeypatch.setattr(
        gate2_module,
        "gate2_report",
        lambda *args, **kwargs: calls.__setitem__("report", calls["report"] + 1),
    )

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        _gate2_cli_args(Path("scored.jsonl"), Path("gb1.csv"), Path("report")),
        terminal_width=300,
    )

    assert result.exit_code != 0
    assert "no metadata sidecar" in result.output
    assert calls["report"] == 0


def test_gate2_command_rejects_wrong_cache_identity_before_running_gate2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    payload = _valid_sidecar_payload(candidates, alphabet=_AA20, max_order=3)
    payload["model_id"] = "facebook/esm2_t30_150M_UR50D"
    _write_cache_with_sidecar_payload(cache, candidates, payload)
    _patch_small_gate2_universe(monkeypatch, candidates)
    calls = {"report": 0}
    monkeypatch.setattr(
        gate2_module,
        "gate2_report",
        lambda *args, **kwargs: calls.__setitem__("report", calls["report"] + 1),
    )

    result = CliRunner().invoke(
        app, _gate2_cli_args(cache, csv, tmp_path / "report"), terminal_width=300
    )

    assert result.exit_code != 0
    assert "model_id" in result.output
    assert calls["report"] == 0


def test_gate2_command_never_overwrites_a_timestamp_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet=_AA20)
    _patch_small_gate2_universe(monkeypatch, candidates)
    _patch_stub_gate2_report(monkeypatch)
    monkeypatch.setattr(cli_module, "_gate2_run_stamp", lambda: "20260714T120000Z", raising=False)
    args = _gate2_cli_args(cache, csv, tmp_path / "report")

    first = CliRunner().invoke(app, args)
    report_path = tmp_path / "report" / "20260714T120000Z" / "gate2.json"
    original = report_path.read_bytes()
    second = CliRunner().invoke(app, args)

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert isinstance(second.exception, FileExistsError)
    assert report_path.read_bytes() == original


@pytest.mark.parametrize(
    ("failing_query", "empty_result"),
    [
        (("rev-parse", "HEAD"), False),
        (("rev-parse", "HEAD"), True),
        (("status", "--porcelain"), False),
    ],
    ids=["head-error", "head-empty", "status-error"],
)
def test_gate2_command_fails_closed_when_required_git_query_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_query: tuple[str, ...],
    empty_result: bool,
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet=_AA20)
    _patch_small_gate2_universe(monkeypatch, candidates)
    _, events = _patch_stub_gate2_report(monkeypatch)

    def _required_git_lines(repo: Path, *args: str) -> list[str]:
        del repo
        if args == failing_query:
            if empty_result:
                return []
            raise subprocess.CalledProcessError(128, ["git", *args])
        if args == ("rev-parse", "HEAD"):
            return ["a" * 40]
        return []

    monkeypatch.setattr(cli_module, "_gate2_required_git_lines", _required_git_lines, raising=False)
    report_root = tmp_path / "report"

    result = CliRunner().invoke(app, _gate2_cli_args(cache, csv, report_root))

    assert result.exit_code != 0
    assert "Gate 2 git provenance" in result.output
    assert "gate2_report" not in events
    assert not report_root.exists()


def test_gate2_command_propagates_dirty_diff_hash_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet=_AA20)
    _patch_small_gate2_universe(monkeypatch, candidates)
    _, events = _patch_stub_gate2_report(monkeypatch)

    def _required_git_lines(repo: Path, *args: str) -> list[str]:
        del repo
        if args == ("rev-parse", "HEAD"):
            return ["a" * 40]
        return [" M src/epibudget/cli.py"]

    def _failed_diff(repo: Path, commit: str) -> str:
        del repo
        raise subprocess.CalledProcessError(128, ["git", "diff", commit])

    monkeypatch.setattr(cli_module, "_gate2_required_git_lines", _required_git_lines, raising=False)
    monkeypatch.setattr(provenance_module, "workspace_code_diff_sha256", _failed_diff)
    report_root = tmp_path / "report"

    result = CliRunner().invoke(app, _gate2_cli_args(cache, csv, report_root))

    assert result.exit_code != 0
    assert "Gate 2 git provenance" in result.output
    assert "gate2_report" not in events
    assert not report_root.exists()


def test_gate2_command_serializes_the_post_completion_finalized_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = enumerate_candidates(GB1_SITES, GB1_WT_AT_SITES, allowed_aa="ACD", max_order=3)
    csv = tmp_path / "gb1.csv"
    _write_gb1_csv(csv, candidates)
    cache = tmp_path / "scored.jsonl"
    _build_scored_cache(cache, candidates, candidates, alphabet=_AA20)
    _patch_small_gate2_universe(monkeypatch, candidates)
    calls: list[tuple[_StubGate2Report, Sequence[ScoredVariant], dict[str, object]]] = []

    def _initial_report(*args: object, **kwargs: object) -> _StubGate2Report:
        provenance = kwargs["provenance"]
        assert isinstance(provenance, dict)
        return _StubGate2Report(
            provenance,
            architecture_decision_eligible=True,
            decision="repair_current_core",
        )

    def _finalize(
        report: _StubGate2Report,
        scored: Sequence[ScoredVariant],
        provenance: dict[str, object],
    ) -> _StubGate2Report:
        calls.append((report, scored, provenance))
        return _StubGate2Report(
            provenance,
            architecture_decision_eligible=False,
            decision="inconclusive_zero_gpu",
        )

    monkeypatch.setattr(gate2_module, "gate2_report", _initial_report)
    monkeypatch.setattr(gate2_module, "finalize_gate2_report", _finalize, raising=False)

    result = CliRunner().invoke(app, _gate2_cli_args(cache, csv, tmp_path / "report"))

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    initial, finalized_scored, final_provenance = calls[0]
    assert initial.architecture_decision_eligible is True
    assert [item.variant for item in finalized_scored] == candidates
    elapsed_seconds = final_provenance["elapsed_seconds"]
    assert isinstance(elapsed_seconds, float)
    assert final_provenance["completed_at_utc"] != final_provenance["started_at_utc"] or (
        elapsed_seconds > 0.0
    )
    run_dir = next((tmp_path / "report").iterdir())
    written = json.loads((run_dir / "gate2.json").read_text(encoding="utf-8"))
    assert written["architecture_decision_eligible"] is False
    assert written["decision"]["architecture_decision_eligible"] is False
    assert written["decision"]["decision"] == "inconclusive_zero_gpu"
