"""Reconstruction backends: Capture in, SceneSpec out."""

# Importing a backend module registers it.
from r2s2r.reconstruct import simfoundry  # pylint: disable=unused-import
from r2s2r.reconstruct.base import (
    ReconstructionBackend,
    make_backend,
    register_backend,
    registered_backends,
)

__all__ = [
    "ReconstructionBackend",
    "make_backend",
    "register_backend",
    "registered_backends",
]
