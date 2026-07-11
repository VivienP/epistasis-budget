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

``score`` is the reference per-variant path and the parity oracle. ``score_batch`` returns the same
numbers via de-duplicated, cross-variant batched forwards (``scoring_plan``): masking a site erases
its residue, so one masked forward serves every substitution there, and identical masked inputs are
run once. Batching/de-dup/threads/device are throughput only — the slow parity test proves it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from epibudget.data import apply_mutations
from epibudget.scoring_plan import (
    MASK_CHAR,
    Consumer,
    Pass,
    Row,
    dedup,
    finalize,
    plan_variant,
    variant_key,
)
from epibudget.types import Mutation, ScoredVariant, Variant

if TYPE_CHECKING:
    import torch

# The 20 standard amino acids, in ESM-2's single-letter token vocabulary.
_AA20 = "ACDEFGHIKLMNPQRSTVWY"


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when a GPU is present, else ``"cpu"``; pass others through.

    Prefer CPU/free tiers for reproducibility, but use available compute when a run is otherwise
    blocked (docs/CLAUDE.md). ``"cpu"`` (the default) and an explicit ``"cuda"`` are honoured as-is.
    """
    if device == "auto":
        import torch  # noqa: PLC0415  # deferred heavy dependency (see _ensure_loaded)

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def additive_delta_g(single_effects: dict[Variant, float], variant: Variant) -> float:
    """Reference implementation of the FORBIDDEN additive score, for tests only.

    ΔG_additive(S) = Σ_{m∈S} ΔG({m}). Used exclusively by ``tests/test_scoring.py`` to demonstrate
    that additive scoring yields ε ≡ 0. Never call this from the scoring path.
    """
    return sum(single_effects[frozenset({m})] for m in variant)


class ConjointScorer:
    """Scores variants with ESM-2 using conjoint conditional log-likelihoods.

    Parameters mirror docs/SPEC.md#3.3. Deterministic given ``seed``. ``device`` defaults to
    ``"cpu"`` (``"auto"`` selects a GPU when present, ``"cuda"`` forces it); ``batch_size`` and
    ``num_threads`` tune throughput only and never change the numbers.

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
        batch_size: int = 32,
        num_threads: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.n_perturbations = n_perturbations
        self.seed = seed
        self.mask_fraction = mask_fraction
        self.batch_size = batch_size
        self.num_threads = num_threads
        # ESM-2 model + tokenizer, treated as Any: transformers is intentionally untyped here
        # (docs pyproject mypy overrides), and the checkpoint is loaded lazily by _ensure_loaded.
        self._model: Any = None
        self._tokenizer: Any = None
        self._mask_id: int = -1
        self._bos_id: int = -1
        self._eos_id: int = -1
        self._aa_ids: dict[str, int] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Deferred so importing this module (and the offline test suite) never pulls in transformers
        # or triggers a model download; the checkpoint loads on first score() only.
        import torch  # noqa: PLC0415
        from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: PLC0415

        if self.num_threads is not None:
            torch.set_num_threads(self.num_threads)
        self.device = resolve_device(self.device)

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
        self._bos_id = int(tokenizer.cls_token_id)
        self._eos_id = int(tokenizer.eos_token_id)
        self._aa_ids = {aa: self._aa_id(aa) for aa in _AA20}
        # The batched path builds token rows char-by-char; guard that ESM tokenisation really is
        # per-residue (one token each, + BOS/EOS) so a row built this way equals tokenizer(seq).
        probe = "MTYK"
        built = [self._bos_id, *[self._aa_ids[c] for c in probe], self._eos_id]
        if self._encode(probe).tolist() != built:
            raise RuntimeError(
                "ESM tokenizer is not char-level; batched scoring cannot match score()"
            )

    def _aa_id(self, aa: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(aa)
        if tid is None or tid == self._tokenizer.unk_token_id:
            raise ValueError(f"amino acid {aa!r} is not in the ESM-2 vocabulary")
        return int(tid)

    def _encode(self, seq: str) -> torch.Tensor:
        enc = self._tokenizer(seq, return_tensors="pt")
        input_ids: torch.Tensor = enc["input_ids"][0].to(self.device)
        return input_ids

    def _encode_masked(self, masked_seq: str) -> torch.Tensor:
        """Token ids for a planned masked row: BOS, per-residue ids (mask char → mask id), EOS.

        Built char-by-char (guarded char-level in _ensure_loaded) so it equals cloning the encoded
        revealed sequence and masking the hidden positions, without a tokenizer call per row.
        """
        import torch  # noqa: PLC0415

        ids = [self._bos_id]
        for c in masked_seq:
            ids.append(self._mask_id if c == MASK_CHAR else self._aa_ids[c])
        ids.append(self._eos_id)
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def score(self, wt: str, variant: Variant) -> ScoredVariant:
        """Conjoint conditional score of ``variant`` vs ``wt`` (+ masking-perturbation variance).

        The per-variant reference path and the parity oracle for ``score_batch``.

        Contract (enforced by tests):
          * mutations are applied to the background BEFORE scoring (conjoint, not additive);
          * the residue read at each position matches the intended mutant residue (no tokenizer
            off-by-one — ESM prepends a BOS token);
          * deterministic given ``self.seed``.
        """
        self._ensure_loaded()
        return self._score_one(wt, variant)

    def score_batch(self, wt: str, variants: Sequence[Variant]) -> list[ScoredVariant]:
        """Score many variants via de-duplicated cross-variant batches; same numbers as ``score``.

        The masked forward rows are planned (``scoring_plan``) and de-duplicated by
        ``(masked_seq, read_pos)`` — identical masked inputs forward once — then run in
        ``batch_size`` chunks; the per-variant ΔG and masking-perturbation variance are reassembled
        to match the per-variant reference exactly (slow parity test), in input order.
        """
        self._ensure_loaded()

        unique_variants: list[Variant] = []
        seen: set[Variant] = set()
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)

        global_passes: list[Pass] = []
        all_rows: list[Row] = []
        for v in unique_variants:
            passes, rows = plan_variant(
                wt,
                v,
                seed=self.seed,
                n_perturbations=self.n_perturbations,
                mask_fraction=self.mask_fraction,
            )
            base = len(global_passes)
            global_passes.extend(passes)
            for masked_seq, read_pos, mut_aa, wt_aa, pass_id, site_index in rows:
                all_rows.append((masked_seq, read_pos, mut_aa, wt_aa, pass_id + base, site_index))

        unique_seqs, unique_read_pos, consumers = dedup(all_rows)
        pass_partials: list[list[float]] = [[0.0] * n_sites for (_v, _vi, n_sites) in global_passes]
        self._resolve_rows(unique_seqs, unique_read_pos, consumers, pass_partials)

        scored = finalize(global_passes, pass_partials)
        result: dict[Variant, ScoredVariant] = {}
        for v in unique_variants:
            if not v:  # order 0 == wild type == reference; ΔG(∅) ≡ 0, no forward
                result[v] = ScoredVariant(variant=v, delta_g=0.0, var_delta_g=0.0)
            else:
                delta_g, var_delta_g = scored[v]
                result[v] = ScoredVariant(variant=v, delta_g=delta_g, var_delta_g=var_delta_g)
        return [result[v] for v in variants]

    # ------------------------------------------------------------------ internals

    def _resolve_rows(
        self,
        unique_seqs: Sequence[str],
        unique_read_pos: Sequence[int],
        consumers: Sequence[list[Consumer]],
        pass_partials: list[list[float]],
    ) -> None:
        """Forward the unique masked rows in ``batch_size`` chunks; write each consumer's ΔlogP.

        All rows share one WT length, so a chunk stacks without padding. Each consumer's value is
        ``logP(mut) − logP(wt)`` at the row's read position — the subtraction done in the model's
        float dtype before casting, matching ``_delta_g_pass``.
        """
        import torch  # noqa: PLC0415

        assert self._model is not None
        n = len(unique_seqs)
        for start in range(0, n, self.batch_size):
            stop = min(start + self.batch_size, n)
            batch_ids = torch.stack(
                [self._encode_masked(unique_seqs[u]) for u in range(start, stop)]
            )
            with torch.no_grad():
                logits = self._model(batch_ids).logits
            read_idx = torch.tensor(
                [unique_read_pos[u] + 1 for u in range(start, stop)], device=self.device
            )
            rows_ax = torch.arange(stop - start, device=self.device)
            logp = torch.log_softmax(logits[rows_ax, read_idx, :], dim=-1)  # [chunk, vocab]
            for local, u in enumerate(range(start, stop)):
                row_logp = logp[local]
                for pass_id, site_index, mut_aa, wt_aa in consumers[u]:
                    delta = float(row_logp[self._aa_ids[mut_aa]] - row_logp[self._aa_ids[wt_aa]])
                    pass_partials[pass_id][site_index] = delta

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
        rng = np.random.default_rng(self.seed + variant_key(muts))
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
