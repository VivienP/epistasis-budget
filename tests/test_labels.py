"""Independent oracles for the plate label accounting (audit finding H-3).

The expected buckets are derived here from the label value alone, never by calling the production
classifier, so a change of predicate in ``labels.py`` cannot make these tests agree with it.
"""

from __future__ import annotations

import math

import pytest

from epibudget.labels import (
    LabelAccounting,
    LabelAccountingError,
    account,
    is_trainable,
    training_target,
)
from epibudget.types import Variant

_N_NON_POSITIVE_ENCODINGS = 2  # exact zero and small negative
_N_MIXED_ROWS = 3
_N_ACTIVE_ROWS = 4
_N_PLATE_ROWS = 10
_N_MISSING_ROWS = 2


def _v(index: int) -> Variant:
    """A distinct single-mutation variant per index (identity only; no landscape semantics)."""
    return frozenset({(index, "A", "C")})


def _expected_bucket(value: float | None) -> str:
    """Oracle: bucket a label from its value alone, independent of ``labels.account``."""
    if value is None:
        return "missing"
    if math.isnan(value) or math.isinf(value):
        return "non_finite"
    if value > 0.0:
        return "valid_positive"
    if value == 0.0:
        return "valid_zero"
    if value > -1.0:
        return "valid_negative_in_domain"
    return "outside_transform_domain"


# One representative of every bucket, including zero and negative score encodings.
_CASES: tuple[tuple[float | None, str], ...] = (
    (2.5, "valid_positive"),
    (1e-12, "valid_positive"),
    (0.0, "valid_zero"),
    (-0.164070661, "valid_negative_in_domain"),  # observed TrpB minimum
    (-0.999999, "valid_negative_in_domain"),
    (-1.0, "outside_transform_domain"),  # log1p(-1) = -inf
    (-2.0, "outside_transform_domain"),
    (float("nan"), "non_finite"),
    (float("inf"), "non_finite"),
    (float("-inf"), "non_finite"),
    (None, "missing"),
)


@pytest.mark.parametrize(("value", "bucket"), _CASES)
def test_every_label_lands_in_its_oracle_bucket(value: float | None, bucket: str) -> None:
    """Each representative value is counted in exactly the bucket the oracle predicts."""
    assert _expected_bucket(value) == bucket  # the oracle itself is self-consistent
    selected = [_v(0)]
    revealed = {} if value is None else {_v(0): value}
    accounting, trainable = account(selected, revealed)
    counts = accounting.model_dump()
    assert counts[bucket] == 1
    assert sum(counts[name] for name in counts if name != "selected") == 1
    trains = bucket in {"valid_positive", "valid_zero", "valid_negative_in_domain"}
    assert len(trainable) == (1 if trains else 0)
    assert is_trainable(value) is trains if value is not None else True


def test_accounting_identity_holds_over_every_bucket_at_once() -> None:
    """The buckets partition a mixed plate exactly; nothing is silently discarded."""
    values = [v for v, _ in _CASES]
    selected = [_v(i) for i in range(len(values))]
    revealed = {_v(i): v for i, v in enumerate(values) if v is not None}
    accounting, trainable = account(selected, revealed)

    expected: dict[str, int] = {}
    for value in values:
        expected[_expected_bucket(value)] = expected.get(_expected_bucket(value), 0) + 1

    assert accounting.selected == len(values)
    for name, count in expected.items():
        assert getattr(accounting, name) == count, name
    assert accounting.effective_train_size == len(trainable)
    assert accounting.effective_train_size == (
        expected["valid_positive"] + expected["valid_zero"] + expected["valid_negative_in_domain"]
    )
    accounting.check()  # must not raise


def test_non_positive_scores_span_zero_and_negative_encodings() -> None:
    """An exact zero and a small negative are counted as the same score-sign class."""
    selected = [_v(0), _v(1), _v(2)]
    revealed = {_v(0): 0.0, _v(1): -0.01, _v(2): 3.0}
    accounting, trainable = account(selected, revealed)
    assert accounting.non_positive_count == _N_NON_POSITIVE_ENCODINGS
    # Non-positive rows are training data, not missing data.
    assert accounting.effective_train_size == _N_MIXED_ROWS
    assert len(trainable) == _N_MIXED_ROWS
    assert accounting.positive_score_fraction == pytest.approx(1 / _N_MIXED_ROWS)


def test_positive_score_fraction_is_not_one_when_the_plate_contains_non_positive_rows() -> None:
    """Regression for the retired ``train_live_fraction``: it read 1.000 on every TrpB plate.

    Under the old positive-vs-exact-zero split a negative label entered neither bucket, so the
    denominator lost it and the fraction was 1.0 however many non-positive rows the plate held.
    """
    selected = [_v(i) for i in range(_N_PLATE_ROWS)]
    revealed = {_v(i): (1.0 if i < _N_ACTIVE_ROWS else -0.02) for i in range(_N_PLATE_ROWS)}
    accounting, _ = account(selected, revealed)
    assert accounting.positive_score_fraction == pytest.approx(_N_ACTIVE_ROWS / _N_PLATE_ROWS)
    assert accounting.effective_train_size == _N_PLATE_ROWS


def test_missing_identities_are_counted_not_dropped() -> None:
    """A selected variant absent from the landscape is ``missing`` and still sums into the plate."""
    selected = [_v(0), _v(1), _v(2)]
    accounting, trainable = account(selected, {_v(0): 1.0})
    assert (accounting.missing, accounting.valid_positive) == (_N_MISSING_ROWS, 1)
    assert accounting.selected == _N_MIXED_ROWS
    assert len(trainable) == 1


def test_check_fails_closed_on_a_broken_partition() -> None:
    """A hand-built accounting whose buckets miss the plate total raises, never reports."""
    broken = LabelAccounting(
        selected=_N_PLATE_ROWS,
        valid_positive=_N_ACTIVE_ROWS,
        valid_zero=0,
        valid_negative_in_domain=0,
        outside_transform_domain=0,
        non_finite=0,
        missing=0,
    )
    with pytest.raises(LabelAccountingError, match="fell into no bucket"):
        broken.check()


def test_training_target_is_strictly_increasing_across_zero() -> None:
    """``log1p`` preserves score ordering across zero, so Spearman evaluation stays sound."""
    values = [-0.5, -0.164070661, -0.01, 0.0, 1e-9, 1.0, 8.76]
    targets = [training_target(v) for v in values]
    assert targets == sorted(targets)
    assert all(math.isfinite(t) for t in targets)
    assert training_target(0.0) == 0.0


@pytest.mark.parametrize("value", [-1.0, -1.5, float("nan"), float("inf")])
def test_training_target_refuses_values_outside_the_transform_domain(value: float) -> None:
    """No caller can obtain a nan/-inf training label by accident."""
    assert not is_trainable(value)
    with pytest.raises(ValueError, match="log1p training domain"):
        training_target(value)
