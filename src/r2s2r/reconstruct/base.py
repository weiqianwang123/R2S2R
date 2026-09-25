"""The reconstruction backend interface and its registry.

A backend turns a :class:`~r2s2r.structs.Capture` into a
:class:`~r2s2r.structs.SceneSpec`. Everything downstream (simulator builders,
deployment) only sees the SceneSpec, so swapping backends touches nothing else.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Callable, ClassVar, TypeVar

from r2s2r.structs import Capture, SceneSpec

_REGISTRY: dict[str, type[ReconstructionBackend]] = {}

_B = TypeVar("_B", bound="type[ReconstructionBackend]")


class ReconstructionBackend(abc.ABC):
    """Capture in, SceneSpec out.

    ``workdir`` holds all backend files.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def reconstruct(self, capture: Capture, workdir: Path) -> SceneSpec:
        """Run the backend end to end."""


def register_backend(name: str) -> Callable[[_B], _B]:
    """Class decorator that makes a backend available to :func:`make_backend`."""

    def _register(cls: _B) -> _B:
        if name in _REGISTRY:
            raise ValueError(f"backend {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return _register


def make_backend(name: str, **kwargs: Any) -> ReconstructionBackend:
    """Instantiate a registered backend by name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def registered_backends() -> list[str]:
    """Names of all registered backends."""
    return sorted(_REGISTRY)
