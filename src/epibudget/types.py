"""Core data model. See docs/SPEC.md#2."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

# A point mutation: (0-indexed position into the WT sequence, wt residue, mutant residue).
Mutation = tuple[int, str, str]

# A variant is a set of point mutations. order == len(variant); the empty set is the wild type.
Variant = frozenset[Mutation]


class ScoredVariant(BaseModel):
    """A variant with its conjoint ESM-2 score and model-uncertainty dispersion."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    variant: Variant
    delta_g: float = Field(
        description="Conjoint conditional log-likelihood ratio vs WT (higher=fitter)"
    )
    var_delta_g: float = Field(ge=0.0, description="Dispersion across masking perturbations")


class Interaction(BaseModel):
    """A pairwise or third-order epistatic interaction, its predicted value and uncertainty.

    An interaction is identified by its concrete ``mutations`` (specific residues), not merely by
    its positions: ε depends on the residues, and there are many amino-acid instances per position
    tuple (19² per pair, 19³ per triple over the 20-letter alphabet). ``sites`` is the derived
    position summary. Build via :meth:`of`, which fills ``sites``/``order`` from the mutations.
    """

    model_config = {"frozen": True}

    mutations: tuple[Mutation, ...]
    sites: tuple[int, ...]
    order: int = Field(ge=2, le=3)
    epsilon_hat: float = Field(description="ESM-2-predicted, WT-referenced epistasis coefficient")
    sigma2: float = Field(
        ge=0.0, description="Current uncertainty (variance) about this coefficient"
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> Interaction:
        positions = [m[0] for m in self.mutations]
        if len(set(positions)) != len(positions):
            raise ValueError(f"interaction has repeated positions: {self.mutations!r}")
        if self.order != len(self.mutations):
            raise ValueError(f"order {self.order} != number of mutations {len(self.mutations)}")
        if self.sites != tuple(sorted(positions)):
            raise ValueError(f"sites {self.sites} inconsistent with mutations {self.mutations!r}")
        if tuple(self.mutations) != tuple(sorted(self.mutations)):
            raise ValueError("mutations must be in sorted (canonical) order; use Interaction.of")
        return self

    @classmethod
    def of(cls, mutations: Sequence[Mutation], epsilon_hat: float, sigma2: float) -> Interaction:
        """Build a canonical Interaction from mutations (sorted; ``sites``/``order`` derived)."""
        muts = tuple(sorted(mutations))
        return cls(
            mutations=muts,
            sites=tuple(sorted(m[0] for m in muts)),
            order=len(muts),
            epsilon_hat=epsilon_hat,
            sigma2=sigma2,
        )


class Allocation(BaseModel):
    """The output of budget allocation: B variants ranked by expected information gain."""

    model_config = {"arbitrary_types_allowed": True}

    budget: int = Field(ge=1)
    selected: list[Variant]
    expected_info_gain: list[float]
    epistasis_map: list[Interaction]
    seed: int
    model_id: str


class Config(BaseModel):
    """Runtime configuration; outputs embed the resolved config for reproducibility. See SPEC #9."""

    model_id: str = "facebook/esm2_t33_650M_UR50D"
    device: str = "cpu"
    n_perturbations: int = Field(default=16, ge=1)
    max_order: int = Field(default=3, ge=2, le=3)
    lambda_: float = Field(
        default=0.0, ge=0.0, le=1.0, description="0=info-optimal, 1=fitness-greedy"
    )
    seed: int = 0
    cache_dir: Path = Path("data/cache")
