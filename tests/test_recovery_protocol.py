"""Offline tests for the registered scientific protocol and its execution policy.

The two identities are deliberately separate: changing how the run is checkpointed or scheduled must
not change what the report means, and therefore must not change the scientific fingerprint.
"""

# ruff: noqa: PLR2004

from __future__ import annotations

import dataclasses

import pytest

from epibudget.fourier_recovery import registered_fit_count
from epibudget.recovery_checkpoint import REGISTERED_BUDGET_BLOCKS
from epibudget.recovery_protocol import (
    REGISTERED_EXECUTION_POLICY,
    REGISTERED_RECOVERY_PROTOCOL,
    RecoveryExecutionPolicy,
    RecoveryScientificProtocol,
)


def _protocol(**changes: object) -> RecoveryScientificProtocol:
    return dataclasses.replace(REGISTERED_RECOVERY_PROTOCOL, **changes)  # type: ignore[arg-type]


def _policy(**changes: object) -> RecoveryExecutionPolicy:
    return dataclasses.replace(REGISTERED_EXECUTION_POLICY, **changes)  # type: ignore[arg-type]


def test_registered_protocol_matches_the_frozen_a1_grid() -> None:
    protocol = REGISTERED_RECOVERY_PROTOCOL

    assert protocol.dataset == "trpb_johnston2024"
    assert protocol.budgets == (48, 96, 192, 384, 768, 1536, 2242, 3072)
    assert protocol.seeds == tuple(range(20))
    assert protocol.methods == (
        "info",
        "fitness",
        "doptimal_reduced_pairwise",
        "random",
        "structural",
    )
    assert protocol.n_folds == 5
    assert protocol.selection_max_order == 3
    assert protocol.estimation_max_order == 2
    assert protocol.coefficient_count == 2_166
    assert protocol.feature_count == 2_242
    assert protocol.sequence_count == 43
    assert protocol.cell_count == 344


def test_execution_policy_holds_every_machinery_knob() -> None:
    policy = REGISTERED_EXECUTION_POLICY

    assert policy.doptimal_block_size == 64
    assert policy.lasso_checkpoint_unit == "fold"
    assert policy.cell_checkpoint_unit == "cell"
    assert policy.scheduling == "breadth-first-budget-major"
    assert policy.legacy_budget_block_size == 4
    assert policy.budget_blocks(REGISTERED_RECOVERY_PROTOCOL.budgets) == (
        (48, 96, 192, 384),
        (768, 1536, 2242, 3072),
    )


def test_the_scientific_protocol_carries_no_execution_knob() -> None:
    scientific = set(REGISTERED_RECOVERY_PROTOCOL.semantic_payload())
    execution = set(REGISTERED_EXECUTION_POLICY.policy_payload())

    assert scientific & execution == set()
    assert "doptimal_block_size" not in scientific
    assert "budget_blocks" not in scientific
    assert "legacy_budget_block_size" not in scientific
    assert not hasattr(REGISTERED_RECOVERY_PROTOCOL, "doptimal_block_size")
    assert not hasattr(REGISTERED_RECOVERY_PROTOCOL, "budget_blocks")


def test_changing_execution_cadence_leaves_the_scientific_identity_untouched() -> None:
    scientific = REGISTERED_RECOVERY_PROTOCOL.semantic_sha256
    policy = REGISTERED_EXECUTION_POLICY.policy_sha256

    faster = _policy(doptimal_block_size=128)
    rescheduled = _policy(scheduling="depth-first-sequence-major")
    relegacy = _policy(legacy_budget_block_size=8)

    assert REGISTERED_RECOVERY_PROTOCOL.semantic_sha256 == scientific
    assert faster.policy_sha256 != policy
    assert rescheduled.policy_sha256 != policy
    assert relegacy.policy_sha256 != policy


def test_changing_the_science_changes_only_the_scientific_identity() -> None:
    baseline = REGISTERED_RECOVERY_PROTOCOL.semantic_sha256

    assert _protocol(version="protocol-v2").semantic_sha256 == baseline
    assert _protocol(n_folds=6).semantic_sha256 != baseline
    assert _protocol(seeds=tuple(range(19))).semantic_sha256 != baseline
    assert _protocol(budgets=(48, 96)).semantic_sha256 != baseline
    assert (
        _policy(version="execution-v2").policy_sha256 == REGISTERED_EXECUTION_POLICY.policy_sha256
    )


def test_sequence_keys_follow_the_canonical_execution_order() -> None:
    keys = REGISTERED_RECOVERY_PROTOCOL.sequence_keys

    assert len(keys) == 43
    assert len(set(keys)) == 43
    assert keys[:3] == (("info", None), ("fitness", None), ("doptimal_reduced_pairwise", None))
    assert keys[3:7] == (("random", 0), ("structural", 0), ("random", 1), ("structural", 1))
    assert keys[-1] == ("structural", 19)


def test_registered_fit_count_reads_the_protocol() -> None:
    protocol = REGISTERED_RECOVERY_PROTOCOL

    assert registered_fit_count(protocol.budgets, protocol.seeds) == protocol.cell_count
    assert registered_fit_count((1,), protocol.seeds) == protocol.sequence_count
    with pytest.raises(ValueError, match="unique"):
        registered_fit_count(protocol.budgets, (0, 0))


def test_prototype_budget_blocks_come_from_the_execution_policy() -> None:
    blocks = REGISTERED_EXECUTION_POLICY.budget_blocks(REGISTERED_RECOVERY_PROTOCOL.budgets)

    assert blocks == REGISTERED_BUDGET_BLOCKS


def test_identity_payloads_expose_both_fingerprints_separately() -> None:
    scientific = REGISTERED_RECOVERY_PROTOCOL.identity_payload()
    execution = REGISTERED_EXECUTION_POLICY.identity_payload()

    assert scientific["semantic_sha256"] == REGISTERED_RECOVERY_PROTOCOL.semantic_sha256
    assert scientific["cell_count"] == 344
    assert scientific["sequence_keys"][:1] == [{"method": "info", "seed": None}]  # type: ignore[index]
    assert execution["policy_sha256"] == REGISTERED_EXECUTION_POLICY.policy_sha256
    assert "semantic_sha256" not in execution
    assert "policy_sha256" not in scientific


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"budgets": (96, 48)}, "strictly increasing"),
        ({"budgets": ()}, "non-empty"),
        ({"seeds": (0, 0)}, "unique"),
        ({"n_folds": 1}, "at least two folds"),
        ({"estimation_max_order": 3}, "estimation order must be 2"),
        ({"selection_max_order": 1}, "cover the estimation order"),
        ({"feature_count": 2_166}, "must exceed its coefficient count"),
        ({"stochastic_methods": ("random", "info")}, "unique"),
        ({"dataset": ""}, "version and a dataset"),
    ],
)
def test_incoherent_protocol_fields_fail_closed(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _protocol(**changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"doptimal_block_size": 0}, "D-optimal block size must be positive"),
        ({"legacy_budget_block_size": 0}, "legacy budget block size must be positive"),
        ({"scheduling": ""}, "scheduling and storage formats"),
        ({"array_format": ""}, "scheduling and storage formats"),
        ({"lasso_checkpoint_unit": ""}, "explicit checkpoint units"),
        ({"version": ""}, "requires a version"),
    ],
)
def test_incoherent_policy_fields_fail_closed(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _policy(**changes)


def test_budget_blocks_reject_a_grid_that_does_not_divide() -> None:
    with pytest.raises(ValueError, match="do not divide into blocks"):
        REGISTERED_EXECUTION_POLICY.budget_blocks((48, 96, 192))
