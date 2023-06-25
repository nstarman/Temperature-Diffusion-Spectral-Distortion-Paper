"""Typing."""

__all__: list[str] = []

from typing import Any, TypeAlias

from numpy import floating
from numpy.typing import NDArray

NDAf: TypeAlias = NDArray[floating[Any]]
