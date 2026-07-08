"""Conjoint ESM-2 scoring. INVARIANT #1 lives here (docs/CLAUDE.md, RESEARCH §5, SPEC §3).

Multi-mutant scores MUST be computed conjointly: apply all of a variant's mutations to the
background, then read the conditional log-likelihood of each mutated residue IN THE MUTATED CONTEXT.
Never score each mutation independently on the wild-type background and sum — that makes every
epistasis term identically zero by construction.

``delta_g`` is the deterministic conjoint score (this is what feeds the epistasis terms and the
de-risk gate). ``var_delta_g`` is a zero-shot uncertainty proxy: the dispersion of that score across
stochastic background-context masking perturbations. ESM-2 ships with dropout probability 0, so
MC-dropout would be identically zero; the masking-perturbation dispersion of SPEC §3.2 is used
instead, calibrated against real prediction error in the Step 4 uncertainty-prior scatter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from epibudget.data import apply_mutations
from epibudget.types import Mutation, ScoredVariant, Variant

if TYPE_CHECKING:
    import torch

# The 20 standard amino acids, in ESM-2's single-letter token vocabulary.
_AA20 = "ACDEFGHIKLMNPQRSTVWY"


def additive_delta_g(single_effects: dict[Variant, float], variant: Variant) -> float:
    """Reference implementation of the FORBIDDEN additive score, for tests only.

    ΔG_additive(S) = Σ_{m∈S} ΔG({m}). Used exclusively by ``tests/test_scoring.py`` to demonstrate
    that additive scoring yields ε ≡ 0. Never call this from the scoring path.
    """
    return sum(single_effects[frozenset({m})] for m in variant)


class ConjointScorer:
    """Scores variants with ESM-2 using conjoint conditional log-likelihoods.

    Parameters mirror docs/SPEC.md#3.3. CPU-only; deterministic given ``seed``.

    ``n_perturbations`` background-masking passes estimate ``var_delta_g``; set it to 0 to skip the
    uncertainty estimate (the de-risk gate needs only the deterministic ``delta_g``).
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        n_perturbations: int = 16,
        seed: int = 0,
        mask_fraction: float = 0.15,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.n_perturbations = n_perturbations
        self.seed = seed
        self.mask_fraction = mask_fraction
        # ESM-2 model + tokenizer, treated as Any: transformers is intentionally untyped here
        # (docs pyproject mypy overrides), and the checkpoint is loaded lazily by _ensure_loaded.
        self._model: Any = None
        self._tokenizer: Any = None
        self._mask_id: int = -1

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Deferred so importing this module (and the offline test suite) never pulls in transformers
        # or triggers a model download; the checkpoint loads on first score() only.
        from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: PLC0415

        tokenizer: Any = AutoTokenizer.from_pretrained(self.model_id)
        model: Any = EsmForMaskedLM.from_pretrained(self.model_id)
        model.eval()
        model.to(self.device)
        self._tokenizer = tokenizer
        self._model = model
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise RuntimeError(f"tokenizer for {self.model_id} has no mask token")
        self._mask_id = int(mask_id)

    def _aa_id(self, aa: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(aa)
        if tid is None or tid == self._tokenizer.unk_token_id:
            raise ValueError(f"amino acid {aa!r} is not in the ESM-2 vocabulary")
        return int(tid)

    def _encode(self, seq: str) -> torch.Tensor:
        enc = self._tokenizer(seq, return_tensors="pt")
        input_ids: torch.Tensor = enc["input_ids"][0].to(self.device)
        return input_ids

    def score(self, wt: str, variant: Variant) -> ScoredVariant:
        """Conjoint conditional score of ``variant`` vs ``wt`` (+ masking-perturbation variance).

        Contract (enforced by tests):
          * mutations are applied to the background BEFORE scoring (conjoint, not additive);
          * the residue read at each position matches the intended mutant residue (no tokenizer
            off-by-one — ESM prepends a BOS token);
          * deterministic given ``self.seed``.
        """
        self._ensure_loaded()
        return self._score_one(wt, variant)

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        """Score many variants, reusing identical forward passes across repeated variants."""
        self._ensure_loaded()
        cache: dict[Variant, ScoredVariant] = {}
        out: list[ScoredVariant] = []
        for v in variants:
            if v not in cache:
                cache[v] = self._score_one(wt, v)
            out.append(cache[v])
        return out

    # ------------------------------------------------------------------ internals

    def _score_one(self, wt: str, variant: Variant) -> ScoredVariant:
        if not variant:  # order 0 == wild type == reference; ΔG(∅) ≡ 0
            return ScoredVariant(variant=variant, delta_g=0.0, var_delta_g=0.0)

        mutant_seq = apply_mutations(wt, variant)
        muts: list[Mutation] = sorted(variant, key=lambda m: (m[0], m[2]))
        sites = [m[0] for m in muts]
        wt_ids = [self._aa_id(m[1]) for m in muts]
        mut_ids = [self._aa_id(m[2]) for m in muts]

        base_ids = self._encode(mutant_seq)
        self._assert_token_alignment(base_ids, sites, mut_ids)

        # Deterministic conjoint ΔG: mask each mutated site in turn (others revealed as mutants).
        delta_g = self._delta_g_pass(base_ids, sites, wt_ids, mut_ids, extra_mask=())
        var_delta_g = self._var_delta_g(base_ids, sites, wt_ids, mut_ids, len(mutant_seq), muts)
        return ScoredVariant(variant=variant, delta_g=delta_g, var_delta_g=var_delta_g)

    def _var_delta_g(
        self,
        base_ids: torch.Tensor,
        sites: Sequence[int],
        wt_ids: Sequence[int],
        mut_ids: Sequence[int],
        seq_len: int,
        muts: Sequence[Mutation],
    ) -> float:
        """Dispersion of the conjoint ΔG across ``n_perturbations`` background-context maskings."""
        if self.n_perturbations <= 0:
            return 0.0
        bg = [q for q in range(seq_len) if q not in set(sites)]
        n_mask = min(len(bg), max(1, round(self.mask_fraction * len(bg)))) if bg else 0
        if n_mask == 0:
            return 0.0
        rng = np.random.default_rng(self.seed + _variant_key(muts))
        passes = []
        for _ in range(self.n_perturbations):
            extra = tuple(int(q) for q in rng.choice(bg, size=n_mask, replace=False))
            passes.append(self._delta_g_pass(base_ids, sites, wt_ids, mut_ids, extra_mask=extra))
        return float(np.var(passes))

    def _delta_g_pass(
        self,
        base_ids: torch.Tensor,
        sites: Sequence[int],
        wt_ids: Sequence[int],
        mut_ids: Sequence[int],
        extra_mask: tuple[int, ...],
    ) -> float:
        """One conjoint pass: Σ_p [log P(mut_p) − log P(wt_p)], site p masked in the mutant context.

        Each mutated position is scored in its own row (masked there, all other mutations still
        present), so a single forward pass covers the whole variant. ``extra_mask`` optionally hides
        background positions to perturb the context for the uncertainty estimate.
        """
        import torch  # noqa: PLC0415  # deferred heavy dependency (see _ensure_loaded)

        assert self._model is not None
        rows = []
        for p in sites:
            row = base_ids.clone()
            row[p + 1] = self._mask_id  # +1: ESM prepends a <cls>/BOS token
            for q in extra_mask:
                row[q + 1] = self._mask_id
            rows.append(row)
        batch = torch.stack(rows)
        with torch.no_grad():
            logits = self._model(batch).logits
        logp = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for i, p in enumerate(sites):
            total += float(logp[i, p + 1, mut_ids[i]] - logp[i, p + 1, wt_ids[i]])
        return total

    def _assert_token_alignment(
        self, base_ids: torch.Tensor, sites: Sequence[int], mut_ids: Sequence[int]
    ) -> None:
        """Guard the ESM BOS off-by-one: token at index p+1 must be the intended mutant residue."""
        for p, mid in zip(sites, mut_ids, strict=True):
            got = int(base_ids[p + 1].item())
            if got != mid:
                raise AssertionError(
                    f"tokenizer alignment error at position {p}: token id {got} at index {p + 1} "
                    f"is not the intended mutant residue id {mid} (BOS off-by-one?)"
                )


def _variant_key(muts: Sequence[Mutation]) -> int:
    """Stable, salt-free integer key for a variant, for reproducible per-variant RNG seeding."""
    key = 0
    for pos, wt_aa, mut_aa in muts:
        key = key * 1_000_003 + pos * 400 + _AA20.index(wt_aa) * 20 + _AA20.index(mut_aa)
    return key % (2**31)
