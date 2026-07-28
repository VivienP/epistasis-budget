"""One shared tie-handling algorithm for every acquisition ranking (audit finding H-1).

Why this exists. On a four-site landscape the loop-coverage weight n(v) takes exactly three values —
1140 for every single, 39 for every double, 1 for every triple (``analytic_loop_counts`` derives
them, ``tests/test_tie_break.py`` checks them). The ``structural`` ranking is therefore *not* an
ordering of candidates at all: it is the mutation-order preference "singles, then doubles, then
triples" with every within-order comparison an exact tie. Whichever tie resolution the sort happens
to inherit chooses the plate.

Before this module the resolution was implicit and inconsistent: ``validate`` sorted the cache in
enumeration order, so all 116 doubles of a B=192 TrpB plate came from the single site pair
(182, 183); ``downstream`` pre-sorted by ``canonical_id``, so the same named method drew those
doubles from three different site pairs. Neither ordering was recorded, and neither was reproducible
from the documented method definition. The as-run TrpB values fell outside the range of 100 seeded
resolutions at B=48 and again at B=192, so a single arbitrary draw was reported as the method.

What this module guarantees. ``seeded_order`` groups candidates into exact-score strata, sorts each
stratum by canonical identity (making it independent of the caller's input order), then applies one
PCG64 permutation per stratum from an explicit ``tie_seed``. The result is:

* identical for the same ``(candidates, score, tie_seed)`` regardless of input enumeration order;
* different, and declared, for a different ``tie_seed``;
* the same algorithm in validation, downstream and the corrective gate paths.

A tie seed is a scientific setting, not an implementation detail: a ranking whose score is tied
across the budget boundary is a *distribution* over plates, and a table that reports one draw is
reporting a sample of size one. ``stratum_crosses_budget`` detects exactly that case so callers can
require the seed distribution instead of a point estimate.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from math import comb

import numpy as np

from epibudget.types import Variant

# Frozen identifier of the tie-resolution algorithm; recorded alongside every seed so a future
# change of algorithm cannot be mistaken for a different draw of the same one.
TIE_BREAK_VERSION = "canonical-strata-pcg64-v1"

# Interactions start at order 2; an order-1 term is an additive effect, not a loop.
_MIN_INTERACTION_ORDER = 2

# The registered default. Any table that reports a single tie draw must report this number with it.
DEFAULT_TIE_SEED = 0

# Number of tie seeds a scientific table must span when a stratum crosses the budget boundary.
REQUIRED_TIE_SEEDS = 100


def canonical_id(variant: Variant) -> str:
    """Order-independent identity string of a variant (sorted mutations, compact JSON).

    Byte-identical to ``downstream.canonical_id`` and ``gate2.canonical_id`` so the three pipelines
    stratify and sort on exactly the same key.
    """
    return json.dumps(
        sorted([list(mutation) for mutation in variant]), separators=(",", ":"), ensure_ascii=True
    )


def analytic_loop_counts(n_sites: int, alphabet_size: int, max_order: int = 3) -> dict[int, int]:
    """Closed-form n(v) per variant order, derived independently of the factor graph.

    A variant v of order k lies in the loop of every interaction S with v subset of S and
    |S| <= max_order. Choosing the remaining j = |S| - k mutated sites from the n_sites - k sites v
    does not touch, each with (alphabet_size - 1) admissible residues, gives

        n(k) = sum_{j=0}^{max_order-k} C(n_sites - k, j) * (alphabet_size - 1)^j

    with the j = 0 term counting S = v itself (only for k >= 2, where v is an interaction). For
    n_sites = 4, alphabet_size = 20, max_order = 3 this yields {1: 1140, 2: 39, 3: 1}: n(v) is
    constant within an order, which is why ``structural`` degenerates to an order preference.
    """
    counts: dict[int, int] = {}
    for order in range(1, max_order + 1):
        total = 0
        for extra in range(0, max_order - order + 1):
            if order + extra < _MIN_INTERACTION_ORDER:
                continue
            total += comb(n_sites - order, extra) * (alphabet_size - 1) ** extra
        counts[order] = total
    return counts


def score_strata[T](
    items: Sequence[T], score: Callable[[T], float], identity: Callable[[T], str]
) -> list[tuple[float, list[T]]]:
    """Group ``items`` into exact-score strata, descending by score, canonically sorted within.

    Sorting each stratum by ``identity`` is what removes the caller's input order from the result:
    two callers holding the same candidate set in different orders produce the same strata.
    """
    grouped: dict[float, list[T]] = {}
    for item in items:
        grouped.setdefault(float(score(item)), []).append(item)
    return [
        (value, sorted(grouped[value], key=identity)) for value in sorted(grouped, reverse=True)
    ]


def seeded_order[T](
    items: Sequence[T],
    score: Callable[[T], float],
    identity: Callable[[T], str],
    tie_seed: int = DEFAULT_TIE_SEED,
) -> list[T]:
    """Rank ``items`` by descending ``score``, resolving exact ties by a seeded permutation.

    Deterministic in ``(set(items), score, tie_seed)`` and invariant to the input ordering.
    """
    rng = np.random.default_rng(tie_seed)
    ordered: list[T] = []
    for _value, stratum in score_strata(items, score, identity):
        permutation = rng.permutation(len(stratum))
        ordered.extend(stratum[int(index)] for index in permutation)
    return ordered


def stratum_crosses_budget[T](
    items: Sequence[T], score: Callable[[T], float], identity: Callable[[T], str], budget: int
) -> bool:
    """True iff an exact-score stratum straddles the budget cut, so the plate is seed-dependent.

    When this holds, which members of that stratum are bought is decided entirely by the tie seed,
    and a single-seed result is one draw from a distribution rather than a property of the method.
    """
    seen = 0
    for _value, stratum in score_strata(items, score, identity):
        if seen < budget < seen + len(stratum):
            return True
        seen += len(stratum)
    return False


def loop_counts_over_universe(
    universe: Sequence[Variant], max_order: int = 3
) -> Mapping[Variant, int]:
    """n(v) for every candidate, counted over the interactions the universe actually contains.

    Equivalent to ``EpistasisFactorGraph.info_gain(frozenset(), v)`` under tau^2 == 1, but computed
    directly from the candidate set so it can serve as an independent cross-check of the graph.
    """
    counts: dict[Variant, int] = dict.fromkeys(universe, 0)
    for variant in universe:
        if not _MIN_INTERACTION_ORDER <= len(variant) <= max_order:
            continue
        mutations = sorted(variant)
        for size in range(1, len(mutations) + 1):
            for member in combinations(mutations, size):
                key = frozenset(member)
                if key in counts:
                    counts[key] += 1
    return counts
