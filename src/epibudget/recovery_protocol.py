"""The registered TrpB Fourier recovery protocol and the policy used to execute it.

These are two different things and they are kept apart on purpose.

:class:`RecoveryScientificProtocol` is what the result means: dataset, methods, budgets, seeds,
folds, Fourier orders, the estimand and its dimensions, and the canonical order of the acquisition
sequences. Its ``semantic_sha256`` identifies the science. Changing it changes what the report is a
report *of*, so it invalidates every recorded state.

:class:`RecoveryExecutionPolicy` is how the run is carried out: how often the D-optimal selection
publishes a resumable state, what a checkpoint contains, the on-disk formats, and the scheduling
order. Its ``policy_sha256`` identifies the machinery. Changing it invalidates intermediate states
that were written under the old machinery, but it must never change the identity of the final
scientific report.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Final

_MINIMUM_FOLD_COUNT: Final = 2
_PAIRWISE_ORDER: Final = 2


@dataclass(frozen=True)
class RecoveryScientificProtocol:
    """What the recovery result means, independently of how it is computed."""

    version: str
    dataset: str
    deterministic_methods: tuple[str, ...]
    stochastic_methods: tuple[str, ...]
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    n_folds: int
    selection_max_order: int
    estimation_max_order: int
    coefficient_count: int
    feature_count: int
    label_transform: str = "log1p(fitness)"

    def __post_init__(self) -> None:
        """Reject any protocol whose own fields are mutually inconsistent."""
        if not self.version or not self.dataset:
            raise ValueError("recovery protocol requires a version and a dataset")
        methods = self.methods
        if len(set(methods)) != len(methods) or not methods:
            raise ValueError("recovery protocol methods must be unique and non-empty")
        if not self.budgets or any(budget < 1 for budget in self.budgets):
            raise ValueError("recovery protocol budgets must be non-empty and positive")
        if tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("recovery protocol budgets must be strictly increasing")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("recovery protocol seeds must be unique")
        if self.n_folds < _MINIMUM_FOLD_COUNT:
            raise ValueError("recovery protocol requires at least two folds")
        if self.estimation_max_order != _PAIRWISE_ORDER:
            raise ValueError("recovery protocol estimation order must be 2")
        if self.selection_max_order < self.estimation_max_order:
            raise ValueError("recovery protocol selection order must cover the estimation order")
        if self.coefficient_count < 1 or self.feature_count <= self.coefficient_count:
            raise ValueError("recovery protocol feature count must exceed its coefficient count")

    @property
    def methods(self) -> tuple[str, ...]:
        """Canonical method order: deterministic methods first, then stochastic methods."""
        return (*self.deterministic_methods, *self.stochastic_methods)

    @property
    def sequence_keys(self) -> tuple[tuple[str, int | None], ...]:
        """Every acquisition sequence identity in canonical order."""
        return (
            *((method, None) for method in self.deterministic_methods),
            *((method, seed) for seed in self.seeds for method in self.stochastic_methods),
        )

    @property
    def sequence_count(self) -> int:
        """Number of registered acquisition sequences."""
        return len(self.sequence_keys)

    @property
    def cell_count(self) -> int:
        """Number of registered method-budget-seed cells."""
        return self.sequence_count * len(self.budgets)

    def fit_count(self, budgets: Sequence[int], seeds: Sequence[int]) -> int:
        """Number of estimator fits for an arbitrary budget and seed grid under this protocol."""
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        sequences = len(self.deterministic_methods) + len(self.stochastic_methods) * len(seeds)
        return len(budgets) * sequences

    def semantic_payload(self) -> dict[str, object]:
        """Return the fields that define the scientific meaning, excluding the version label."""
        return {
            "dataset": self.dataset,
            "deterministic_methods": list(self.deterministic_methods),
            "stochastic_methods": list(self.stochastic_methods),
            "budgets": list(self.budgets),
            "seeds": list(self.seeds),
            "n_folds": self.n_folds,
            "selection_max_order": self.selection_max_order,
            "estimation_max_order": self.estimation_max_order,
            "coefficient_count": self.coefficient_count,
            "feature_count": self.feature_count,
            "label_transform": self.label_transform,
        }

    @cached_property
    def semantic_sha256(self) -> str:
        """SHA-256 over the canonical encoding of every scientific field."""
        return _digest(self.semantic_payload())

    def identity_payload(self) -> dict[str, object]:
        """Return the scientific block embedded in a durable run identity."""
        return {
            "version": self.version,
            "semantic_sha256": self.semantic_sha256,
            "sequence_keys": [
                {"method": method, "seed": seed} for method, seed in self.sequence_keys
            ],
            "cell_count": self.cell_count,
            **self.semantic_payload(),
        }


@dataclass(frozen=True)
class RecoveryExecutionPolicy:
    """How a registered protocol is executed, checkpointed, stored, and scheduled."""

    version: str
    doptimal_block_size: int
    lasso_checkpoint_unit: str
    cell_checkpoint_unit: str
    scheduling: str
    manifest_format: str
    array_format: str
    legacy_budget_block_size: int

    def __post_init__(self) -> None:
        """Reject a policy that cannot describe a resumable run."""
        if not self.version:
            raise ValueError("execution policy requires a version")
        if self.doptimal_block_size < 1:
            raise ValueError("execution policy D-optimal block size must be positive")
        if self.legacy_budget_block_size < 1:
            raise ValueError("execution policy legacy budget block size must be positive")
        if not self.scheduling or not self.manifest_format or not self.array_format:
            raise ValueError("execution policy requires scheduling and storage formats")
        if not self.lasso_checkpoint_unit or not self.cell_checkpoint_unit:
            raise ValueError("execution policy requires explicit checkpoint units")

    def budget_blocks(self, budgets: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Chunk registered budgets into the prototype's fixed-size checkpoint blocks."""
        size = self.legacy_budget_block_size
        if len(budgets) % size:
            raise ValueError(f"{len(budgets)} budgets do not divide into blocks of {size}")
        return tuple(tuple(budgets[start : start + size]) for start in range(0, len(budgets), size))

    def policy_payload(self) -> dict[str, object]:
        """Return the fields that define the execution machinery, excluding the version label."""
        return {
            "doptimal_block_size": self.doptimal_block_size,
            "lasso_checkpoint_unit": self.lasso_checkpoint_unit,
            "cell_checkpoint_unit": self.cell_checkpoint_unit,
            "scheduling": self.scheduling,
            "manifest_format": self.manifest_format,
            "array_format": self.array_format,
            "legacy_budget_block_size": self.legacy_budget_block_size,
        }

    @cached_property
    def policy_sha256(self) -> str:
        """SHA-256 over the canonical encoding of every execution field."""
        return _digest(self.policy_payload())

    def identity_payload(self) -> dict[str, object]:
        """Return the execution block embedded in a durable run identity."""
        return {
            "version": self.version,
            "policy_sha256": self.policy_sha256,
            **self.policy_payload(),
        }


def _digest(payload: dict[str, object]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


REGISTERED_RECOVERY_PROTOCOL: Final = RecoveryScientificProtocol(
    version="epibudget-fourier-recovery-protocol-v1",
    dataset="trpb_johnston2024",
    deterministic_methods=("info", "fitness", "doptimal_reduced_pairwise"),
    stochastic_methods=("random", "structural"),
    budgets=(48, 96, 192, 384, 768, 1536, 2242, 3072),
    seeds=tuple(range(20)),
    n_folds=5,
    selection_max_order=3,
    estimation_max_order=2,
    coefficient_count=2_166,
    feature_count=2_242,
)

REGISTERED_EXECUTION_POLICY: Final = RecoveryExecutionPolicy(
    version="epibudget-fourier-recovery-execution-v1",
    doptimal_block_size=64,
    lasso_checkpoint_unit="fold",
    cell_checkpoint_unit="cell",
    scheduling="breadth-first-budget-major",
    manifest_format="epibudget-run-store-manifest-v1",
    array_format="epibudget-run-store-array-v1",
    legacy_budget_block_size=4,
)
