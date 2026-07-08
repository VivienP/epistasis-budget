"""Conjoint ESM-2 scoring. INVARIANT #1 lives here (docs/CLAUDE.md, RESEARCH §5, SPEC §3).

Multi-mutant scores MUST be computed conjointly: apply all of a variant's mutations to the background,
then read the conditional log-likelihood of each mutated residue IN THE MUTATED CONTEXT. Never score
each mutation independently on the wild-type background and sum — that makes every epistasis term
identically zero by construction.
"""

from __future__ import annotations

from collections.abc import Sequence

from epibudget.types import ScoredVariant, Variant


def additive_delta_g(single_effects: dict[Variant, float], variant: Variant) -> float:
    """Reference implementation of the FORBIDDEN additive score, for tests only.

    ΔG_additive(S) = Σ_{m∈S} ΔG({m}). Used exclusively by ``tests/test_scoring.py`` to demonstrate that
    additive scoring yields ε ≡ 0. Never call this from the scoring path.
    """
    return sum(single_effects[frozenset({m})] for m in variant)


class ConjointScorer:
    """Scores variants with ESM-2 using conjoint conditional log-likelihoods.

    Parameters mirror docs/SPEC.md#3.3. CPU-only; deterministic given ``seed``.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        n_perturbations: int = 16,
        seed: int = 0,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.n_perturbations = n_perturbations
        self.seed = seed
        # Model/tokenizer loaded lazily on first score() — see Week 0 in docs/ROADMAP.md.

    def score(self, wt: str, variant: Variant) -> ScoredVariant:
        """Conjoint conditional score of ``variant`` against ``wt`` (+ masking-perturbation variance).

        Contract (enforced by tests):
          * mutations are applied to the background BEFORE scoring (conjoint, not additive);
          * the residue read at each position matches the intended mutant residue (no tokenizer
            off-by-one — ESM prepends a BOS token);
          * deterministic given ``self.seed``.
        """
        raise NotImplementedError("Week 0 — see docs/ROADMAP.md and .claude/agents/esm-scoring-engineer")

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        """Score many variants, caching shared mutated-context forward passes."""
        raise NotImplementedError("Week 0 — see docs/ROADMAP.md")
