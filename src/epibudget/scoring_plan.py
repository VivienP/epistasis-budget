"""Deterministic planning for batched conjoint scoring — the pure half of ``ConjointScorer``.

``score_batch``'s masked-marginal forward passes are enumerated here as plain data (masked residue
strings + the position to read), de-duplicated by ``(masked_seq, read_pos)``, and reassembled into
per-variant ``delta_g`` / ``var_delta_g``. There is no torch or transformers import: the model is
never touched, so the planning + de-dup + finalisation logic is exercised by the offline test suite
without an ESM-2 download.

The plan reproduces the exact masking and per-variant RNG of ``ConjointScorer._score_one`` /
``_var_delta_g``, so the batched path returns the same numbers as the untouched per-variant
reference (``ConjointScorer.score``), which the slow parity test asserts. Masking site ``p`` erases
the residue there, so one forward of ``mutant_seq`` with ``p`` masked yields the conditional
distribution for every substitution at ``p`` — which is why the 19 substitutions at a site collapse
to a single row.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from epibudget.data import apply_mutations
from epibudget.types import Mutation, Variant

# The 20 standard amino acids, in ESM-2's single-letter token vocabulary.
_AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Marks a masked residue position in a planned row. Not a valid amino-acid letter, so it can never
# collide with a real residue; the impure half replaces it with the tokenizer's mask id.
MASK_CHAR = "*"

# A planned pass: (variant, var_index, n_sites). var_index == -1 is the deterministic ΔG pass;
# 0..n-1 are the var_delta_g perturbation passes, in draw order. n_sites == order == read slots.
Pass = tuple[Variant, int, int]

# A planned row: (masked_seq, read_pos, mut_aa, wt_aa, pass_id, site_index). read_pos is the
# 0-indexed residue position to read (the masked query site); pass_id indexes into the pass list;
# site_index is the slot within that pass. mut_aa varies per consumer; wt_aa is fixed by read_pos.
Row = tuple[str, int, str, str, int, int]

# One resolved consumer of a unique row: (pass_id, site_index, mut_aa, wt_aa).
Consumer = tuple[int, int, str, str]


def variant_key(muts: Sequence[Mutation]) -> int:
    """Stable, salt-free integer key for a variant, for reproducible per-variant RNG seeding.

    The single source of truth for both the reference ``_var_delta_g`` and the batched planner, so
    their perturbation draws are guaranteed identical. Order-sensitive: feed ``muts`` sorted by
    ``(pos, mut_aa)`` (the order ``_score_one`` uses).
    """
    key = 0
    for pos, wt_aa, mut_aa in muts:
        key = key * 1_000_003 + pos * 400 + _AA20.index(wt_aa) * 20 + _AA20.index(mut_aa)
    return key % (2**31)


def _masked_row(
    mutant_seq: str,
    masked_positions: tuple[int, ...],
    read_pos: int,
    mut_aa: str,
    wt_aa: str,
    pass_id: int,
    site_index: int,
) -> Row:
    """One row: ``mutant_seq`` with ``masked_positions`` hidden, read at ``read_pos``."""
    chars = list(mutant_seq)
    for q in masked_positions:
        chars[q] = MASK_CHAR
    return ("".join(chars), read_pos, mut_aa, wt_aa, pass_id, site_index)


def plan_variant(
    wt: str,
    variant: Variant,
    *,
    seed: int,
    n_perturbations: int,
    mask_fraction: float,
) -> tuple[list[Pass], list[Row]]:
    """Plan the forward rows for one variant, reproducing ``_score_one`` + ``_var_delta_g`` exactly.

    Returns ``(passes, rows)`` with ``pass_id`` in each row indexing into ``passes`` (local,
    0-based; the caller offsets them when merging variants). The empty variant yields ``([], [])`` —
    it is ΔG(∅) ≡ 0, scored with no forward.
    """
    if not variant:
        return [], []

    mutant_seq = apply_mutations(wt, variant)
    muts: list[Mutation] = sorted(variant, key=lambda m: (m[0], m[2]))
    sites = [m[0] for m in muts]
    seq_len = len(mutant_seq)

    passes: list[Pass] = []
    rows: list[Row] = []

    # Deterministic ΔG pass: mask each mutated site in turn, other mutations revealed as mutants.
    det_id = len(passes)
    passes.append((variant, -1, len(sites)))
    for site_index, m in enumerate(muts):
        rows.append(_masked_row(mutant_seq, (m[0],), m[0], m[2], m[1], det_id, site_index))

    # var_delta_g: n_perturbations background-context maskings (skipped when disabled/degenerate).
    if n_perturbations > 0:
        bg = [q for q in range(seq_len) if q not in set(sites)]
        n_mask = min(len(bg), max(1, round(mask_fraction * len(bg)))) if bg else 0
        if n_mask > 0:
            rng = np.random.default_rng(seed + variant_key(muts))
            for t in range(n_perturbations):
                extra = tuple(int(q) for q in rng.choice(bg, size=n_mask, replace=False))
                pass_id = len(passes)
                passes.append((variant, t, len(sites)))
                for site_index, m in enumerate(muts):
                    rows.append(
                        _masked_row(
                            mutant_seq, (m[0], *extra), m[0], m[2], m[1], pass_id, site_index
                        )
                    )
    return passes, rows


def dedup(rows: Sequence[Row]) -> tuple[list[str], list[int], list[list[Consumer]]]:
    """De-duplicate planned rows by ``(masked_seq, read_pos)``.

    Returns ``(unique_seqs, unique_read_pos, consumers)`` where ``consumers[u]`` lists every
    ``(pass_id, site_index, mut_aa, wt_aa)`` reading unique row ``u``. Identical masked input rows
    forward once; their shared read-position log-probabilities serve every consumer (each picks its
    own ``mut_aa`` — ``wt_aa`` is fixed by ``read_pos``).
    """
    index: dict[tuple[str, int], int] = {}
    unique_seqs: list[str] = []
    unique_read_pos: list[int] = []
    consumers: list[list[Consumer]] = []
    for masked_seq, read_pos, mut_aa, wt_aa, pass_id, site_index in rows:
        key = (masked_seq, read_pos)
        u = index.get(key)
        if u is None:
            u = len(unique_seqs)
            index[key] = u
            unique_seqs.append(masked_seq)
            unique_read_pos.append(read_pos)
            consumers.append([])
        consumers[u].append((pass_id, site_index, mut_aa, wt_aa))
    return unique_seqs, unique_read_pos, consumers


def finalize(
    passes: Sequence[Pass],
    pass_partials: Sequence[Sequence[float]],
) -> dict[Variant, tuple[float, float]]:
    """Assemble per-variant ``(delta_g, var_delta_g)`` from resolved per-pass, per-site values.

    ``pass_partials[pass_id][site_index]`` is ``logP(mut) - logP(wt)`` for that row. The
    deterministic pass (``var_index == -1``) sums its site slots in slot order to give ``delta_g``
    (matching ``_delta_g_pass``); the perturbation-pass sums, in draw order, feed
    ``float(np.var(...))`` for ``var_delta_g`` (matching ``_var_delta_g``, a population variance).
    """
    delta_g: dict[Variant, float] = {}
    var_sums: dict[Variant, dict[int, float]] = {}
    for pass_id, (variant, var_index, n_sites) in enumerate(passes):
        pass_sum = 0.0
        for site_index in range(n_sites):
            pass_sum += pass_partials[pass_id][site_index]
        if var_index < 0:
            delta_g[variant] = pass_sum
        else:
            var_sums.setdefault(variant, {})[var_index] = pass_sum

    out: dict[Variant, tuple[float, float]] = {}
    for variant, dg in delta_g.items():
        sums = var_sums.get(variant)
        if sums:
            ordered = [sums[t] for t in range(len(sums))]
            var_dg = float(np.var(np.array(ordered, dtype=np.float64)))
        else:
            var_dg = 0.0
        out[variant] = (dg, var_dg)
    return out
