"""Linear-Gaussian factor graph over epistatic interactions. See docs/SPEC.md#5.

Each candidate variant's ΔG carries an independent Gaussian prior N(ΔG_hat, τ²), τ² = var_delta_g
(the ESM masking-perturbation dispersion). Each interaction's coefficient ε(S) is the fixed ±1
inclusion–exclusion combination of the ΔG of its loop L(S), so its prior variance is the plain sum
σ²(ε(S)) = Σ_{T∈L(S)} τ_T². Measuring a variant reveals its ΔG exactly, collapsing that τ² to 0 —
standard linear-Gaussian conditioning at zero observation noise. Uncertainty is therefore reduced by
dropping the measured variants' contributions from every loop they brace.

Submodularity claim (honest wording). Under this model — independent priors across distinct
variants and exact measurement — info_gain(M, v) = τ_v² · n(v), where n(v) is the number of
interactions whose loop contains v. This is independent of M: info_gain is **modular**, a strict
special case of submodular in which the diminishing-returns inequality
info_gain(A,v) ≥ info_gain(B,v) (A ⊆ B, v ∉ B) holds with EQUALITY, not strictly. Modular functions
are submodular, so greedy is not
merely (1−1/e)-near-optimal but EXACTLY optimal for a fixed budget (it coincides with sorting
candidates by the fixed weight τ_v²·n(v) and taking the top B). This is a consequence of the
independent-noise assumption; it is not a general theorem about A-optimal (sum-of-variance) design,
which is not submodular in general once correlated priors or non-zero observation noise are added.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from epibudget.epistasis import interaction_loop
from epibudget.types import Interaction, Mutation, Variant


class EpistasisFactorGraph:
    """Tracks posterior uncertainty of interaction terms as variants are hypothetically measured.

    The API depends only on which variants are *measured* (membership), never on any revealed
    fitness value, so it cannot leak labels into the uncertainty model (docs/VALIDATION.md threats
    table); labels enter only via ``data.reveal_measured_fitness`` in the validation harness.
    """

    def __init__(
        self, interactions: Sequence[Interaction], var_delta_g: Mapping[Variant, float]
    ) -> None:
        self.interactions = list(interactions)
        keys = [interaction.mutations for interaction in self.interactions]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "duplicate interactions (same mutations) would make total_uncertainty and "
                "posterior_variance disagree; deduplicate before constructing the graph"
            )
        # Per-interaction loop (its non-empty sub-variants) and the τ² of each loop member.
        self._loops: list[list[Variant]] = []
        self._tau2: dict[Variant, float] = {}
        for interaction in self.interactions:
            loop = interaction_loop(interaction.mutations)
            for member in loop:
                if member not in var_delta_g:
                    raise KeyError(
                        f"var_delta_g is missing sub-variant {sorted(member)} required by "
                        f"interaction {list(interaction.mutations)}"
                    )
                self._tau2[member] = var_delta_g[member]
            self._loops.append(loop)
        # weight[v] = τ_v² · n(v), n(v) = number of interaction loops containing v.
        counts: Counter[Variant] = Counter()
        for loop in self._loops:
            counts.update(loop)
        self._weight: dict[Variant, float] = {v: self._tau2[v] * n for v, n in counts.items()}
        self._prior_total = sum(self._weight.values())

    def posterior_variance(self, measured: frozenset[Variant]) -> dict[tuple[Mutation, ...], float]:
        """Posterior σ² of each interaction given ``measured`` (keyed by its mutation tuple)."""
        return {
            tuple(sorted(interaction.mutations)): sum(
                self._tau2[member] for member in loop if member not in measured
            )
            for interaction, loop in zip(self.interactions, self._loops, strict=True)
        }

    def total_uncertainty(self, measured: frozenset[Variant]) -> float:
        """Σ σ² over all interactions given ``measured`` (the objective we minimise).

        Equal to Σ over interactions of ``posterior_variance``; computed in O(|measured|) via the
        prior total minus the weight of each measured variant (a swap of the double sum).
        """
        return self._prior_total - sum(self._weight.get(v, 0.0) for v in measured)

    def info_gain(self, measured: frozenset[Variant], candidate: Variant) -> float:
        """Reduction in total uncertainty from measuring ``candidate`` (≥ 0, modular)."""
        return self.total_uncertainty(measured) - self.total_uncertainty(measured | {candidate})
