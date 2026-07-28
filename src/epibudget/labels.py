"""Exhaustive accounting of measured labels revealed by a selected plate (audit finding H-3).

Every selected variant lands in exactly one bucket, and the buckets sum to the plate size. The
previous downstream accounting split revealed labels into positive / exact-zero / non-finite and
silently discarded everything else, so a finite negative label vanished from the training set and
from every count. On TrpB — 35,643 of 160,000 values are negative and none is exactly zero — that
dropped up to 17% of a plate with no record and reported a training "live fraction" of 1.000.

Label semantics come from the committed dataset provenance, not from the sign alone.
``scripts/fetch_trpb.py`` records the TrpB label as "an aggregated catalytic-fitness score
(Kowalsky et al.); <= 0 is inactive (like a dead row)". A non-positive TrpB value is therefore a
valid measurement of an inactive variant — the same biological category as a GB1 fitness-zero row,
recorded by an assay whose inactive readout scatters slightly below zero rather than resting exactly
at it. Such rows are training data, not missing data.

The training transform is ``log1p``, defined and strictly increasing on (-1, inf). Every observed
TrpB label satisfies f > -1 (the minimum is -0.164), so the whole inactive class is inside the
domain. A label at or below -1 has no ``log1p`` image and is bucketed as out-of-domain rather than
silently coerced; none occurs in either registered landscape, and the bucket exists so a future
dataset cannot reintroduce the silent drop.

This module is about the *downstream* training target only. The map-recovery path uses a log-ratio
that is undefined at f <= 0, so it conditions on the strictly-positive sub-landscape; that is a
different, separately documented eligible population (see ``recovery.eligible_population``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, log1p

from pydantic import BaseModel

from epibudget.types import Variant

# log1p is defined on (-1, inf); -1 itself maps to -inf, so the domain is strictly greater than -1.
LOG1P_DOMAIN_LOWER_BOUND = -1.0

# Bucket names, in report order. Exhaustive and mutually exclusive over a selected plate.
BUCKET_NAMES: tuple[str, ...] = (
    "valid_positive",
    "valid_zero",
    "valid_negative_in_domain",
    "outside_transform_domain",
    "non_finite",
    "missing",
)


class LabelAccountingError(RuntimeError):
    """A plate's label buckets do not sum to the number of selected variants."""


class LabelAccounting(BaseModel):
    """Exhaustive per-plate label accounting; the buckets sum to ``selected`` exactly.

    ``valid_positive``/``valid_zero``/``valid_negative_in_domain`` are the training rows: all three
    are real measurements with a finite ``log1p`` image. ``valid_zero`` and
    ``valid_negative_in_domain`` are the inactive class (GB1 records it as exact zero, TrpB as a
    small negative), kept together in ``inactive_count`` so the two landscapes are comparable.

    ``outside_transform_domain`` (f <= -1) and ``non_finite`` are real rows that cannot be
    transformed; ``missing`` are selected identities absent from the landscape. None of the three is
    a training row, and each is counted rather than dropped.
    """

    model_config = {"frozen": True}

    selected: int
    valid_positive: int
    valid_zero: int
    valid_negative_in_domain: int
    outside_transform_domain: int
    non_finite: int
    missing: int

    @property
    def effective_train_size(self) -> int:
        """Rows that reach the learner: every real measurement inside the ``log1p`` domain."""
        return self.valid_positive + self.valid_zero + self.valid_negative_in_domain

    @property
    def inactive_count(self) -> int:
        """Measured-inactive rows, however the assay encodes inactivity (exact zero or negative)."""
        return self.valid_zero + self.valid_negative_in_domain

    @property
    def active_fraction(self) -> float | None:
        """Fraction of training rows that are active (f > 0); ``None`` when nothing trains.

        Replaces the previous ``train_live_fraction``, which counted only positive-vs-exact-zero and
        therefore reported 1.000 on every TrpB plate while a fifth of the plate was inactive.
        """
        total = self.effective_train_size
        return self.valid_positive / total if total else None

    def check(self) -> None:
        """Raise unless the buckets partition the plate exactly (fail closed, never silent)."""
        total = (
            self.valid_positive
            + self.valid_zero
            + self.valid_negative_in_domain
            + self.outside_transform_domain
            + self.non_finite
            + self.missing
        )
        if total != self.selected:
            raise LabelAccountingError(
                f"label buckets sum to {total} but {self.selected} variants were selected; "
                f"a revealed label fell into no bucket (buckets: "
                f"positive={self.valid_positive}, zero={self.valid_zero}, "
                f"negative_in_domain={self.valid_negative_in_domain}, "
                f"outside_domain={self.outside_transform_domain}, "
                f"non_finite={self.non_finite}, missing={self.missing})"
            )


def is_trainable(value: float) -> bool:
    """True iff ``value`` is a real measurement with a finite ``log1p`` image."""
    return isfinite(value) and value > LOG1P_DOMAIN_LOWER_BOUND


def training_target(value: float) -> float:
    """``log1p(fitness)`` — strictly increasing on the trainable domain, so ranking is preserved."""
    if not is_trainable(value):
        raise ValueError(f"{value!r} is outside the log1p training domain (-1, inf)")
    return log1p(value)


def account(
    selected: Sequence[Variant], revealed: Mapping[Variant, float]
) -> tuple[LabelAccounting, list[Variant]]:
    """Bucket every selected variant and return the accounting plus the trainable variants.

    ``revealed`` holds the labels the landscape actually returned; a selected identity absent from
    it is ``missing``. The returned variant list is sorted by the caller's canonical order upstream,
    so this function never imposes an ordering of its own.
    """
    positive = zero = negative = out_of_domain = non_finite = missing = 0
    trainable: list[Variant] = []
    for variant in selected:
        if variant not in revealed:
            missing += 1
            continue
        value = revealed[variant]
        if not isfinite(value):
            non_finite += 1
        elif value > 0.0:
            positive += 1
            trainable.append(variant)
        elif value == 0.0:
            zero += 1
            trainable.append(variant)
        elif value > LOG1P_DOMAIN_LOWER_BOUND:
            negative += 1
            trainable.append(variant)
        else:
            out_of_domain += 1
    accounting = LabelAccounting(
        selected=len(selected),
        valid_positive=positive,
        valid_zero=zero,
        valid_negative_in_domain=negative,
        outside_transform_domain=out_of_domain,
        non_finite=non_finite,
        missing=missing,
    )
    accounting.check()
    return accounting, trainable
