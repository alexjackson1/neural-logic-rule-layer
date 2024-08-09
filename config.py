from beartype import beartype
from jaxtyping import Float

import torch
from torch import Tensor


@beartype
def l_and(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic conjunction (x ∧ y)."""
    return x * y


@beartype
def l_or(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic disjunction (x ∨ y)."""
    return x + y - x * y


@beartype
def l_not(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic negation (≠g x)."""
    return 1 - x


@beartype
def l_mi(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic material implication (x → y)."""
    return l_or(l_not(x), y)


@beartype
def l_xor(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic exclusive or (x ⊕ y)."""
    return l_and(l_or(x, y), l_not(l_and(x, y)))


@beartype
def l_eq(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    """Algebraic equality (x ≣ y)."""
    return l_not(l_xor(x, y))


ALL_FNS = [
    ("x ∧ y", l_and),
    ("x ∨ y", l_or),
    ("≠g x", lambda x, _: l_not(x)),
    ("≠g y", lambda _, y: l_not(y)),
    ("x ⊕ y", l_xor),
    ("(x + y)/2", lambda x, y: (x + y) / 2),
    ("x·≠g y", lambda x, y: x * l_not(y)),
    ("x", lambda x, _: x),
    ("y", lambda _, y: y),
    ("0.7", lambda x, _: torch.full(x.shape, 0.7)),
]
