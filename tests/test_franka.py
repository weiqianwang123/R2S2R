"""Tests for robots/franka.py."""

import numpy as np

from r2s2r.robots.franka import FRANKA_HAND_TCP, Q_READY, PandaKinematics


def test_fk_ready_pose():
    """The ready pose points the hand straight down in front of the base."""
    T = PandaKinematics().fk(Q_READY)
    # Flange at 0.107 + TCP offset below the hand; known reference numbers.
    assert np.allclose(T[:3, 3], [0.307, 0.0, 0.487], atol=2e-3)
    assert np.allclose(T[:3, 2], [0.0, 0.0, -1.0], atol=1e-3)
    T_hand = PandaKinematics(tcp_offset=0.0).fk(Q_READY)
    assert np.isclose(T_hand[2, 3] - T[2, 3], FRANKA_HAND_TCP)


def test_jacobian_matches_finite_differences():
    """Linear rows of the geometric Jacobian are the TCP velocity."""
    kin = PandaKinematics()
    q = Q_READY + np.array([0.1, -0.2, 0.3, 0.1, -0.1, 0.2, -0.3])
    J = kin.jacobian(q)
    eps = 1e-6
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = eps
        dp = (kin.fk(q + dq)[:3, 3] - kin.fk(q - dq)[:3, 3]) / (2 * eps)
        assert np.allclose(J[:3, i], dp, atol=1e-6)


def test_ik_round_trip():
    """IK recovers a reachable pose from a nearby seed."""
    kin = PandaKinematics()
    q_true = Q_READY + np.array([0.3, 0.2, -0.2, 0.3, 0.1, -0.2, 0.4])
    target = kin.fk(q_true)
    sol = kin.ik(target, Q_READY)
    assert sol.converged
    assert np.allclose(kin.fk(sol.q), target, atol=1e-3)
    assert np.all(sol.q >= kin.q_min) and np.all(sol.q <= kin.q_max)
