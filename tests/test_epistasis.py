"""Tests for the WT-referenced epistasis terms — the mathematical heart, and invariant #1.

These pass today (pure functions). They encode the property that additive ΔG ⇒ ε ≡ 0, which is why
conjoint (non-additive) ESM-2 scoring is mandatory
(docs/RESEARCH_EPISTASIS.md#5, CLAUDE.md invariant #1).
"""

from __future__ import annotations

from itertools import chain, combinations
from math import exp

import numpy as np
import pytest

from epibudget.epistasis import (
    _epsilon,
    _landscape_tensor,
    _orthonormal_contrast_basis,
    _wht_forward,
    epsilon_pairwise,
    epsilon_third,
    ground_truth_epistasis,
    interaction_loop,
    predicted_epistasis,
    wht_spectrum,
    wt_centered_log_fitness,
)
from epibudget.scoring import additive_delta_g
from epibudget.types import Mutation, ScoredVariant, Variant

MUT_I: Mutation = (0, "A", "C")
MUT_J: Mutation = (1, "D", "E")
MUT_K: Mutation = (2, "F", "G")
MUT_L: Mutation = (3, "H", "I")

_PAIRWISE_LOOP = 3  # {i}, {j}, {i,j}
_THIRD_LOOP = 7  # 3 singles + 3 pairs + the triple
_PAIRWISE_ORDER = 2
_THIRD_ORDER = 3
_N_TOY_SITES = 3
_N_PAIRS_3SITES = 3  # C(3,2) pairwise instances over three sites
_MODE_TOL = 1e-6  # a real injected mode clears this; numerical zero stays well below


def _v(*muts: Mutation) -> Variant:
    return frozenset(muts)


def _additive_landscape(
    single_effects: dict[Variant, float], muts: list[Mutation]
) -> dict[Variant, float]:
    """ΔG map where every variant's score is the sum of its single-mutant effects.

    The landscape is perfectly additive by construction.
    """
    dg: dict[Variant, float] = {}
    for order in range(1, len(muts) + 1):
        for combo in combinations(muts, order):
            dg[frozenset(combo)] = additive_delta_g(single_effects, frozenset(combo))
    return dg


def test_epsilon_pairwise_detects_interaction() -> None:
    dg = {_v(MUT_I): 1.0, _v(MUT_J): 1.0, _v(MUT_I, MUT_J): 2.5}
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(0.5)


def test_epsilon_pairwise_is_zero_for_additive_landscape() -> None:
    # Invariant #1, in miniature: additivity ⇒ no epistasis signal.
    singles = {_v(MUT_I): 1.5, _v(MUT_J): -0.7}
    dg = _additive_landscape(singles, [MUT_I, MUT_J])
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(0.0)


def test_epsilon_third_is_zero_for_additive_landscape() -> None:
    singles = {_v(MUT_I): 1.5, _v(MUT_J): -0.7, _v(MUT_K): 0.4}
    dg = _additive_landscape(singles, [MUT_I, MUT_J, MUT_K])
    assert epsilon_third(dg, MUT_I, MUT_J, MUT_K) == pytest.approx(0.0)


def test_epsilon_third_recovers_injected_interaction() -> None:
    # Start additive, then inject a pure third-order term of +0.3 into the triple.
    singles = {_v(MUT_I): 1.5, _v(MUT_J): -0.7, _v(MUT_K): 0.4}
    dg = _additive_landscape(singles, [MUT_I, MUT_J, MUT_K])
    dg[_v(MUT_I, MUT_J, MUT_K)] += 0.3
    assert epsilon_third(dg, MUT_I, MUT_J, MUT_K) == pytest.approx(0.3)


def test_epsilon_symmetric_in_its_sites() -> None:
    dg = {_v(MUT_I): 0.2, _v(MUT_J): 0.9, _v(MUT_I, MUT_J): 1.7}
    assert epsilon_pairwise(dg, MUT_I, MUT_J) == pytest.approx(epsilon_pairwise(dg, MUT_J, MUT_I))


# --- measured-fitness WT anchor ---------------------------------------------------------------


def test_wt_centered_log_fitness_has_exact_wt_zero_and_is_scale_invariant() -> None:
    fitness = {
        frozenset(): 2.5,
        _v(MUT_I): 5.0,
        _v(MUT_J): 1.25,
    }
    centered = wt_centered_log_fitness(fitness)
    scaled = wt_centered_log_fitness({variant: 7.0 * value for variant, value in fitness.items()})

    assert centered[frozenset()].hex() == "0x0.0p+0"
    assert centered[_v(MUT_I)] == pytest.approx(np.log(2.0))
    assert centered[_v(MUT_J)] == pytest.approx(np.log(0.5))
    assert scaled == pytest.approx(centered)


@pytest.mark.parametrize(
    "fitness",
    [
        {},
        {frozenset(): 0.0},
        {frozenset(): -1.0},
        {frozenset(): float("inf")},
        {frozenset(): float("nan")},
    ],
)
def test_wt_centered_log_fitness_rejects_missing_or_invalid_reference(
    fitness: dict[Variant, float],
) -> None:
    with pytest.raises(ValueError):
        wt_centered_log_fitness(fitness)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_wt_centered_log_fitness_rejects_any_nonfinite_value(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        wt_centered_log_fitness({frozenset(): 1.0, _v(MUT_I): value})


def test_wt_centered_log_fitness_drops_nonpositive_nonreference_values() -> None:
    centered = wt_centered_log_fitness(
        {frozenset(): 2.0, _v(MUT_I): 0.0, _v(MUT_J): -3.0, _v(MUT_K): 4.0}
    )
    assert set(centered) == {frozenset(), _v(MUT_K)}


def test_wt_centered_logs_recover_pair_third_and_arbitrary_inclusion_exclusion() -> None:
    mutations = (MUT_I, MUT_J, MUT_K, MUT_L)
    raw_log: dict[Variant, float] = {frozenset(): -0.6}
    for order in range(1, len(mutations) + 1):
        for index, combo in enumerate(combinations(mutations, order), start=1):
            raw_log[frozenset(combo)] = 0.2 * order + 0.07 * index * index
    centered = wt_centered_log_fitness({variant: exp(value) for variant, value in raw_log.items()})

    def expected(term: tuple[Mutation, ...]) -> float:
        return sum(
            (-1.0 if (len(term) - order) % 2 else 1.0) * raw_log[frozenset(combo)]
            for order in range(len(term) + 1)
            for combo in combinations(term, order)
        )

    assert epsilon_pairwise(centered, MUT_I, MUT_J) == pytest.approx(expected((MUT_I, MUT_J)))
    assert epsilon_third(centered, MUT_I, MUT_J, MUT_K) == pytest.approx(
        expected((MUT_I, MUT_J, MUT_K))
    )
    assert _epsilon(centered, mutations) == pytest.approx(expected(mutations))


# --- general inclusion-exclusion helper agrees with the hardcoded order-2/3 forms ---------------


def test_general_epsilon_matches_pairwise_and_third() -> None:
    rng = np.random.default_rng(0)
    all_muts = [MUT_I, MUT_J, MUT_K]
    dg: dict[Variant, float] = {}
    for r in range(1, 4):
        for combo in combinations(all_muts, r):
            dg[frozenset(combo)] = float(rng.normal())
    assert _epsilon(dg, (MUT_I, MUT_J)) == pytest.approx(epsilon_pairwise(dg, MUT_I, MUT_J))
    assert _epsilon(dg, (MUT_I, MUT_J, MUT_K)) == pytest.approx(
        epsilon_third(dg, MUT_I, MUT_J, MUT_K)
    )


def test_interaction_loop_sizes() -> None:
    assert len(interaction_loop((MUT_I, MUT_J))) == _PAIRWISE_LOOP
    assert len(interaction_loop((MUT_I, MUT_J, MUT_K))) == _THIRD_LOOP


# --- predicted_epistasis: epsilon_hat + seed sigma^2 from ESM dispersion ------------------------


def _scored(variant: Variant, delta_g: float, var_delta_g: float) -> ScoredVariant:
    return ScoredVariant(variant=variant, delta_g=delta_g, var_delta_g=var_delta_g)


def test_predicted_epistasis_pairwise_value_and_variance() -> None:
    scored = [
        _scored(_v(MUT_I), 1.0, 0.10),
        _scored(_v(MUT_J), 2.0, 0.20),
        _scored(_v(MUT_I, MUT_J), 3.5, 0.05),
    ]
    (interaction,) = predicted_epistasis(scored)
    assert interaction.sites == (0, 1)
    assert interaction.order == _PAIRWISE_ORDER
    assert interaction.epsilon_hat == pytest.approx(3.5 - 1.0 - 2.0)  # 0.5
    assert interaction.sigma2 == pytest.approx(0.10 + 0.20 + 0.05)  # loop sum


def test_predicted_epistasis_third_order_variance_sums_seven_terms() -> None:
    variants = {
        _v(MUT_I): 0.11,
        _v(MUT_J): 0.12,
        _v(MUT_K): 0.13,
        _v(MUT_I, MUT_J): 0.21,
        _v(MUT_I, MUT_K): 0.22,
        _v(MUT_J, MUT_K): 0.23,
        _v(MUT_I, MUT_J, MUT_K): 0.31,
    }
    scored = [_scored(v, 1.0, var) for v, var in variants.items()]
    interactions = predicted_epistasis(scored)
    triple = next(i for i in interactions if i.order == _THIRD_ORDER)
    assert triple.sigma2 == pytest.approx(sum(variants.values()))


def test_predicted_epistasis_raises_on_missing_lower_order() -> None:
    # The double is present but a constituent single is not — a wiring bug, not a data gap.
    scored = [_scored(_v(MUT_I), 1.0, 0.1), _scored(_v(MUT_I, MUT_J), 3.5, 0.05)]
    with pytest.raises(KeyError, match="missing lower-order"):
        predicted_epistasis(scored)


def _complete_three_site_scored() -> list[ScoredVariant]:
    variants = [
        _v(MUT_I),
        _v(MUT_J),
        _v(MUT_K),
        _v(MUT_I, MUT_J),
        _v(MUT_I, MUT_K),
        _v(MUT_J, MUT_K),
        _v(MUT_I, MUT_J, MUT_K),
    ]
    return [_scored(v, float(len(v)), 0.1) for v in variants]


def test_max_order_2_excludes_third_order_interactions() -> None:
    scored = _complete_three_site_scored()
    predicted = predicted_epistasis(scored, max_order=2)
    assert {i.order for i in predicted} == {_PAIRWISE_ORDER}
    assert len(predicted) == _N_PAIRS_3SITES  # pairwise instances only, no triple

    dg = {sv.variant: sv.delta_g for sv in scored}
    truth = ground_truth_epistasis(dg, max_order=2)
    assert {i.order for i in truth} == {_PAIRWISE_ORDER}
    assert len(truth) == _N_PAIRS_3SITES


def test_max_order_3_includes_the_triple() -> None:
    predicted = predicted_epistasis(_complete_three_site_scored(), max_order=3)
    assert {i.order for i in predicted} == {_PAIRWISE_ORDER, _THIRD_ORDER}
    assert sum(1 for i in predicted if i.order == _THIRD_ORDER) == 1


# --- ground_truth_epistasis: sigma^2 == 0, drops incomplete loops (never imputes) ---------------


def test_ground_truth_epistasis_value_and_zero_variance() -> None:
    dg = {_v(MUT_I): 1.0, _v(MUT_J): 2.0, _v(MUT_I, MUT_J): 3.5}
    (interaction,) = ground_truth_epistasis(dg)
    assert interaction.epsilon_hat == pytest.approx(0.5)
    assert interaction.sigma2 == 0.0


def test_ground_truth_epistasis_drops_interaction_with_missing_constituent() -> None:
    # The double is present but a single is missing (e.g. dead upstream): drop, never impute.
    dg = {_v(MUT_I): 1.0, _v(MUT_I, MUT_J): 3.5}
    assert ground_truth_epistasis(dg) == []


# --- Walsh-Hadamard spectrum -------------------------------------------------------------------

_M0: Mutation = (0, "A", "C")
_M1: Mutation = (1, "A", "C")
_M2: Mutation = (2, "A", "C")
_TOY_SITES = (0, 1, 2)


def _toy_landscape(effects: dict[int, float], c: float = 0.0) -> dict[Variant, float]:
    """A complete 3-site, 2-letter landscape: additive part + a pure order-2 mode on sites (0,1).

    ``c`` scales the pure order-2 contribution ``c·s0·s1`` (sᵢ = +1 if site i is WT, −1 if mutated);
    it engages only sites 0 and 1, so it is a pure pairwise mode there. WT-anchored (ΔG(∅)=0).
    """
    muts = {0: _M0, 1: _M1, 2: _M2}
    raw: dict[Variant, float] = {}
    for combo in chain.from_iterable(combinations(_TOY_SITES, r) for r in range(4)):
        present = set(combo)
        sign0 = -1.0 if 0 in present else 1.0
        sign1 = -1.0 if 1 in present else 1.0
        raw[frozenset(muts[s] for s in present)] = (
            sum(effects[s] for s in present) + c * sign0 * sign1
        )
    wt_value = raw[frozenset()]
    return {variant: value - wt_value for variant, value in raw.items()}


def test_wht_forward_is_orthonormal_roundtrip_and_parseval() -> None:
    rng = np.random.default_rng(1)
    tensor = rng.normal(size=(3, 4, 2)).astype(np.float64)
    bases = [_orthonormal_contrast_basis(q) for q in tensor.shape]
    coeffs = _wht_forward(tensor, bases)
    reconstructed = _wht_forward(coeffs, [b.T for b in bases])
    assert np.allclose(reconstructed, tensor)
    assert float(np.square(coeffs).sum()) == pytest.approx(float(np.square(tensor).sum()))


def test_wht_spectrum_parseval_equals_population_variance() -> None:
    dg = _toy_landscape({0: 0.5, 1: -0.3, 2: 0.8}, c=0.7)
    spectrum = wht_spectrum(dg, _TOY_SITES)
    population_var = float(np.var(np.array(list(dg.values()), dtype=np.float64)))
    assert sum(spectrum.values()) == pytest.approx(population_var)


def test_wht_additive_landscape_and_epsilon_both_vanish_above_order_one() -> None:
    dg = _toy_landscape({0: 0.5, 1: -0.3, 2: 0.8}, c=0.0)
    spectrum = wht_spectrum(dg, _TOY_SITES)
    assert spectrum[1] > 0.0  # additive main effects are real
    assert spectrum[2] == pytest.approx(0.0, abs=1e-12)
    assert spectrum[3] == pytest.approx(0.0, abs=1e-12)
    # The other formalism agrees: every WT-referenced epsilon of order >= 2 is exactly zero.
    for interaction in ground_truth_epistasis(dg):
        assert interaction.epsilon_hat == pytest.approx(0.0, abs=1e-12)


def test_wht_pure_order2_mode_shows_in_both_spectrum_and_epsilon() -> None:
    c = 0.7
    dg = _toy_landscape({0: 0.5, 1: -0.3, 2: 0.8}, c=c)
    spectrum = wht_spectrum(dg, _TOY_SITES)
    assert spectrum[2] > _MODE_TOL  # the injected pairwise mode is visible in the WHT spectrum
    assert spectrum[3] == pytest.approx(0.0, abs=1e-12)  # nothing leaks to third order
    by_sites = {i.sites: i.epsilon_hat for i in ground_truth_epistasis(dg)}
    assert by_sites[(0, 1)] == pytest.approx(4.0 * c)  # the interacting pair
    assert by_sites[(0, 2)] == pytest.approx(0.0, abs=1e-12)  # non-interacting pairs
    assert by_sites[(1, 2)] == pytest.approx(0.0, abs=1e-12)
    assert by_sites[(0, 1, 2)] == pytest.approx(0.0, abs=1e-12)  # no third-order structure


def test_landscape_tensor_wt_cell_is_zero_and_shape_matches_alphabet() -> None:
    dg = _toy_landscape({0: 0.5, 1: -0.3, 2: 0.8}, c=0.4)
    tensor, bases = _landscape_tensor(dg, _TOY_SITES)
    assert tensor.shape == (2, 2, 2)  # 2-letter alphabet per site
    assert tensor[0, 0, 0] == pytest.approx(0.0)  # all-WT cell == ΔG(∅) == 0
    assert len(bases) == _N_TOY_SITES


def test_wht_spectrum_raises_on_incomplete_landscape() -> None:
    dg = _toy_landscape({0: 0.5, 1: -0.3, 2: 0.8}, c=0.4)
    del dg[frozenset({_M0, _M1, _M2})]  # punch a hole in the complete grid
    with pytest.raises(ValueError, match="incomplete landscape"):
        wht_spectrum(dg, _TOY_SITES)


def _multiallelic_landscape(bump: float = 0.0) -> dict[Variant, float]:
    """A complete 2-site landscape with a 3-letter site 0 (A/C/D) and 2-letter site 1 (A/C).

    Exercises the q>2 order-aggregation path of wht_spectrum, which the binary toy never touches.
    ``bump`` adds a non-additive term to a single (D,C) cell.
    """
    f0 = {"A": 0.0, "C": 0.5, "D": -0.3}
    f1 = {"A": 0.0, "C": 0.8}
    dg: dict[Variant, float] = {}
    for a0 in ("A", "C", "D"):
        for a1 in ("A", "C"):
            muts = {(s, "A", a) for s, a in ((0, a0), (1, a1)) if a != "A"}
            value = f0[a0] + f1[a1] + (bump if (a0 == "D" and a1 == "C") else 0.0)
            dg[frozenset(muts)] = value
    return dg


def test_wht_multiallelic_additive_has_no_order2_and_obeys_parseval() -> None:
    dg = _multiallelic_landscape(bump=0.0)
    spectrum = wht_spectrum(dg, (0, 1))
    assert spectrum[1] > 0.0
    assert spectrum[2] == pytest.approx(0.0, abs=1e-12)  # additive: no interaction on a q=3 axis
    population_var = float(np.var(np.array(list(dg.values()), dtype=np.float64)))
    assert sum(spectrum.values()) == pytest.approx(population_var)


def test_wht_multiallelic_interaction_shows_at_order2_and_obeys_parseval() -> None:
    dg = _multiallelic_landscape(bump=0.7)
    spectrum = wht_spectrum(dg, (0, 1))
    assert spectrum[2] > _MODE_TOL  # the non-additive term surfaces at order 2 on the q=3 grid
    population_var = float(np.var(np.array(list(dg.values()), dtype=np.float64)))
    assert sum(spectrum.values()) == pytest.approx(population_var)


def test_landscape_tensor_raises_on_inconsistent_wt_residue() -> None:
    dg = {frozenset(): 0.0, _v((0, "A", "C")): 1.0, _v((0, "V", "D")): 2.0}
    with pytest.raises(ValueError, match="inconsistent WT residue"):
        wht_spectrum(dg, (0,))


def test_landscape_tensor_raises_on_site_with_no_observed_mutation() -> None:
    dg = {frozenset(): 0.0, _v((0, "A", "C")): 1.0}  # site 1 is never mutated
    with pytest.raises(ValueError, match="no mutations observed"):
        wht_spectrum(dg, (0, 1))
