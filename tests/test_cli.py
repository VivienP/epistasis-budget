"""Offline tests for the CLI's pure helpers (FASTA reading, variant parsing).

The ``score`` command itself needs an ESM-2 forward pass, so it is not exercised here; only the
input-parsing helpers are, which is where the surprising failure modes (indexing, WT mismatch) live.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from math import exp
from pathlib import Path

import pytest
from typer.testing import CliRunner

from epibudget import downstream as downstream_module
from epibudget import scoring
from epibudget.cli import (
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
    apply_mutations,
    enumerate_candidates,
)
from epibudget.scored_cache import CacheMetadata, cache_metadata_path
from epibudget.types import ScoredVariant, Variant

_ALLOC_BUDGET = 4


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
