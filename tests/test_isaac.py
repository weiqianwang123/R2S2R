"""Tests for sim/isaac.py: where PhysX runs a scene, and which way gravity points."""

import numpy as np
import pytest

from r2s2r.sim.isaac import gravity, physics_device
from r2s2r.structs import ObjectSpec, SceneSpec
from r2s2r.transforms import make_transform


def _scene(objects, T_base_support=np.eye(4)):
    return SceneSpec(
        name="t",
        embodiment="franka_panda",
        objects=objects,
        T_base_support=T_base_support,
        cameras={},
        reference_camera="c",
        reference_step=0,
        joint_positions=np.zeros(7),
    )


def _object(name, **kwargs):
    return ObjectSpec(name, name, f"{name}.urdf", np.eye(4), **kwargs)


def test_a_scene_runs_on_the_cpu_unless_it_has_a_cloth():
    """Rigid and articulated objects on the CPU; a cloth needs the GPU."""
    box = _object("box", joints={"lid_hinge": 0.0})
    towel = _object(
        "towel",
        cloth={"thickness": 0.002, "youngs_modulus": 5e5, "poissons_ratio": 0.3},
    )
    assert physics_device(_scene([_object("cup"), box]), "cuda:0") == "cpu"
    assert physics_device(_scene([box, towel]), "cuda:0") == "cuda:0"


def test_gravity_is_along_the_supports_normal():
    """A support leaning 0.4 degrees: gravity leans with it, 9.81 m/s^2 still."""
    a = np.radians(0.4)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    g = np.array(gravity(_scene([], make_transform(R, [0.4, 0.0, -0.02]))))
    assert np.linalg.norm(g) == pytest.approx(9.81)
    assert g / 9.81 == pytest.approx(-R[:, 2])
