"""Tests for r2s2r.sim with MuJoCo: a scene settled and a recording replayed in it
through the calls every simulator answers, leaving the files Isaac Lab's leave; a
cloth settled afterwards, the bodies within its reach with it, and moving beside a
session (Newton stood in for)."""

import json
import sys
import textwrap

import mujoco
import numpy as np
import pytest
import trimesh
from conftest import hinged_box, rgbd_capture, robot_or_skip, towel_mesh
from scipy.spatial.transform import Rotation

from r2s2r import cli, sim
from r2s2r.assets import urdf_visual_meshes
from r2s2r.sim import cloth as cloth_module
from r2s2r.sim import isaac
from r2s2r.sim.mjscene import Session
from r2s2r.sim.world import cloth_surface
from r2s2r.structs import SceneSpec
from r2s2r.testbed.pick import run_scene_pick
from r2s2r.testbed.policy import CONTROL_DT
from r2s2r.tools.objects import assemble
from r2s2r.transforms import look_at, make_transform, transform_points
from r2s2r.workspace import Workspace

pytest.importorskip("mujoco")

TARGET = (0.5, 0.0, 0.04)
CAMERAS = {
    "c1": ("ext1", look_at((0.95, 0.35, 0.45), TARGET)),
    "c2": ("wrist", look_at((0.45, -0.4, 0.4), TARGET)),
}


def _run(tmp_path, cloths=None, T_base_support=None):
    """A run of the test capture (the FR3 at home), and the hinged box assembled
    on its table (with ``cloths``, name to ``T_base_obj``: a 30 cm towel each; the
    support at ``T_base_support``)."""
    robot_or_skip("fr3_robotiq")
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic", sim="mujoco")
    objects, _ = hinged_box(tmp_path, make_transform(np.eye(3), [0.5, 0.0, 0.0]))
    if T_base_support is not None:
        objects["support"]["T_base_support"] = T_base_support.tolist()
    towel_mesh(0.3).export(tmp_path / "towel.obj")
    for name, T_base_obj in (cloths or {}).items():
        objects["objects"].append(
            {
                "name": name,
                "mesh": "towel.obj",
                "T_base_obj": T_base_obj.tolist(),
                "mass": 0.05,
                "cloth": {"thickness": 0.002, "youngs_modulus": 5e5},
            }
        )
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "hull")
    return ws


def _newton(jobs):
    """A stand-in for the Newton job: every cloth 1 cm lower along the support's
    normal, every dynamic body 1 cm along its x; each job and its input kept in
    ``jobs``."""

    def run_env_job(script, job, workdir, env):  # pylint: disable=unused-argument
        data = np.load(job["input"])
        jobs.append((job, {k: data[k] for k in data.files}))
        out = {
            f"cloth{i}_vertices": data[f"cloth{i}_vertices"] - [0.0, 0.0, 0.01]
            for i in range(len(job["cloths"]))
        }
        dynamic = data["body_dynamic"]
        out["body_pose"] = data["body_pose"] + np.outer(dynamic, [0.01] + [0] * 6)
        np.savez(job["output"], **out)
        speeds = [{"name": c["name"], "speed_m_s": 0.001} for c in job["cloths"]]
        return {"output": job["output"], "cloths": speeds, "device": "cpu"} | {
            "seconds_wall": 0.1
        }

    return run_env_job


def test_a_scene_settles_in_mujoco(tmp_path):
    """The objects at rest (the lid, let go half open, falls shut), the report as
    Isaac Lab's, the scene saved without USDs."""
    ws = _run(tmp_path)
    report = sim.settle(tmp_path / "scene", ws.capture.root, tmp_path / "s5", ws.sim)
    assert report["scene"] == str(tmp_path / "s5" / "scene.json")
    box = report["objects"]["box"]
    assert box["moved_m"] < 0.002 and box["joints_moved"]["hinge"] > 0.7
    assert json.loads((tmp_path / "s5" / "settle.json").read_text()) == {
        k: v for k, v in report.items() if k != "scene"
    }
    (obj,) = SceneSpec.load(tmp_path / "s5").objects
    assert obj.usd is None and obj.joints["hinge"] > -0.1


@pytest.mark.gl
def test_a_recording_replays_in_mujoco(tmp_path):
    """Every frame of the static period rendered and compared: the residuals per
    camera, the renders and the comparison where Isaac Lab's replay leaves them."""
    ws = _run(tmp_path)
    out = tmp_path / "replay"
    summary = sim.replay(tmp_path / "scene", ws.capture.root, out, ws.sim, every=2)
    assert summary["frames"] == len(list((out / "frames").glob("*_sim.png")))
    assert set(summary["median_depth_residual_m"]) == {"ext1", "wrist"}  # by role
    log = json.loads((out / "replay.json").read_text())
    assert max(error for _, error in log["arm_error_rad"]) < 1e-6
    assert (out / "compare" / "compare.json").exists() and summary["sheets"]


def test_a_cloth_settles_after_the_bodies(tmp_path, monkeypatch):
    """MuJoCo holds the towel where it lies; the Newton job gets everything that
    collides (the box's hulls and the robot's, where they came to rest) and the towel,
    in the support's frame, with its material; its settled surface replaces the
    towel's mesh, and the report says how far it moved."""
    ws = _run(tmp_path, {"towel": make_transform(np.eye(3), [0.5, 0.0, 0.17])})
    jobs: list = []
    monkeypatch.setattr(cloth_module, "run_env_job", _newton(jobs))
    report = sim.settle(tmp_path / "scene", ws.capture.root, tmp_path / "s5", ws.sim)
    assert len(jobs) == 1
    job, data = jobs[0]
    material = job["cloths"][0]
    assert material["name"] == "towel"
    assert material["density"] == pytest.approx(0.05 / 0.09)
    assert material["tri_ke"] == pytest.approx(5e5 * 0.002 / 2.6)  # E t / 2 (1 + nu)
    assert material["tri_ka"] == pytest.approx(5e5 * 0.002 * 0.3 / 0.91)
    assert material["bending"] == pytest.approx(5e5 * 0.002**3 / (12 * 0.91))
    assert material["radius"] == pytest.approx(0.001)
    assert len(data["body_pose"]) > 2  # the box's two links, the robot's
    assert not data["body_dynamic"].any()  # the box is articulated: held still
    assert np.allclose(data["cloth0_vertices"][:, 2], 0.17)  # where it lies, held
    assert report["objects"]["box"]["moved_m"] < 0.002  # the towel touched nothing
    assert report["objects"]["towel"] == {
        "moved_m": 0.01,
        "mean_moved_m": 0.01,
        "dropped_m": 0.01,
        "speed_m_s": 0.001,
    }
    assert report["cloth"] == {"device": "cpu", "seconds_wall": 0.1}
    towel = next(o for o in SceneSpec.load(tmp_path / "s5").objects if o.cloth)
    assert towel.asset_path == str(tmp_path / "s5/objects/towel/towel.urdf")
    assert towel.usd is None
    surface = urdf_visual_meshes(towel.asset_path)[0].mesh
    assert np.allclose(surface.vertices[:, 2], -0.01)  # the object frame's


def test_cloths_settle_in_any_frame_and_again_where_they_are(tmp_path, monkeypatch):
    """Two cloths, one turned, on a tilted support: each comes back where the job
    left it, in its own frame; settling the settled scene again in its own directory
    keeps every file; a cloth that starts below the support is refused."""
    tilt = Rotation.from_euler("xz", [3, 20], degrees=True).as_matrix()
    support = make_transform(tilt, [0.02, -0.01, 0.0])
    turned = Rotation.from_euler("zx", [30, 10], degrees=True).as_matrix()
    cloths = {
        "towel": support @ make_transform(np.eye(3), [0.5, 0.0, 0.17]),
        "napkin": support @ make_transform(turned, [0.45, 0.05, 0.25]),
    }
    ws = _run(tmp_path, cloths, support)
    jobs: list = []
    monkeypatch.setattr(cloth_module, "run_env_job", _newton(jobs))
    sim.settle(tmp_path / "scene", ws.capture.root, tmp_path / "s5", ws.sim)
    placed = {  # each surface in the base frame
        name: transform_points(T, towel_mesh(0.3).vertices)
        for name, T in cloths.items()
    }
    lower = make_transform(np.eye(3), -0.01 * support[:3, 2])

    def surfaces():
        scene = SceneSpec.load(tmp_path / "s5")
        return {
            o.name: transform_points(
                o.T_base_obj, urdf_visual_meshes(o.asset_path)[0].mesh.vertices
            )
            for o in scene.objects
            if o.cloth
        }

    for name, points in surfaces().items():
        assert np.allclose(points, transform_points(lower, placed[name]), atol=1e-6)
    cloth_module.settle(tmp_path / "s5", ws.capture.root, 1.0)  # again, in place
    for name, points in surfaces().items():
        expected = transform_points(lower @ lower, placed[name])
        assert np.allclose(points, expected, atol=1e-6)

    below = {"towel": support @ make_transform(np.eye(3), [0.5, 0.0, -0.01])}
    ws = _run(tmp_path / "below", below, support)
    with pytest.raises(ValueError, match="towel starts 10 mm below the support"):
        sim.settle(
            tmp_path / "below/scene", ws.capture.root, tmp_path / "below/s5", ws.sim
        )


def test_the_bodies_within_a_cloths_reach_settle_with_it(tmp_path, monkeypatch):
    """A cup beside the towel moves with it and takes Newton's pose, its hulls meeting
    the other bodies too; a cup out of its reach stays still; the robot's hulls meet
    only the towel."""
    robot_or_skip("fr3_robotiq")
    capture = rgbd_capture(tmp_path / "capture", CAMERAS)
    ws = Workspace.create(capture, tmp_path / "run", "agentic", sim="mujoco")
    objects, _ = hinged_box(tmp_path, make_transform(np.eye(3), [0.5, 0.25, 0.0]))
    trimesh.creation.cylinder(radius=0.03, height=0.08).apply_translation(
        [0, 0, 0.04]
    ).export(tmp_path / "cup.obj")
    towel_mesh(0.3).export(tmp_path / "towel.obj")
    for name, xy in (("near_cup", (0.5, 0.17)), ("far_cup", (0.3, -0.3))):
        T = make_transform(np.eye(3), [*xy, 0.0])
        objects["objects"].append(
            {"name": name, "mesh": "cup.obj", "T_base_obj": T.tolist(), "mass": 0.1}
        )
    objects["objects"].append(
        {
            "name": "towel",
            "mesh": "towel.obj",
            "T_base_obj": make_transform(np.eye(3), [0.5, 0.0, 0.01]).tolist(),
            "mass": 0.05,
            "cloth": {"thickness": 0.002, "youngs_modulus": 5e5},
        }
    )
    (tmp_path / "objects.json").write_text(json.dumps(objects))
    assemble(ws, tmp_path / "objects.json", tmp_path / "scene", "hull")
    jobs: list = []
    monkeypatch.setattr(cloth_module, "run_env_job", _newton(jobs))
    report = sim.settle(tmp_path / "scene", ws.capture.root, tmp_path / "s5", ws.sim)
    assert len(jobs) == 1
    data = jobs[0][1]
    assert data["body_dynamic"].sum() == 1  # the near cup's
    near = int(np.flatnonzero(data["body_dynamic"])[0])
    assert data["hull_bodies"][data["hull_body"] == near].all()  # it meets the box
    assert not data["hull_bodies"].all()  # the robot's hulls meet only the cloths
    assert report["objects"]["near_cup"]["by_cloth"] == {
        "moved_m": 0.01,
        "turned_deg": 0.0,
    }
    assert "by_cloth" not in report["objects"]["far_cup"]
    scene = {o.name: o for o in SceneSpec.load(tmp_path / "s5").objects}
    assert scene["near_cup"].T_base_obj[0, 3] == pytest.approx(0.51, abs=2e-3)
    assert scene["far_cup"].T_base_obj[0, 3] == pytest.approx(0.3, abs=2e-3)


FAKE_NEWTON = """
import base64, json, struct, sys
import numpy as np

def send(message):
    body = json.dumps(message).encode()
    sys.stdout.buffer.write(struct.pack("<I", len(body)) + body)
    sys.stdout.buffer.flush()

vertices, poses = [], None
while True:
    head = sys.stdin.buffer.read(4)
    if len(head) < 4:
        break
    request = json.loads(sys.stdin.buffer.read(struct.unpack("<I", head)[0]))
    if request["op"] == "init":
        data = np.load(request["input"])
        vertices = [data[f"cloth{i}_vertices"] for i in range(len(request["cloths"]))]
        send({"ok": True, "device": "fake"})
    elif request["op"] == "step":  # a centimetre up a second, and the last poses
        a = request["body_pose"]
        poses = np.frombuffer(base64.b64decode(a["array"])).reshape(a["shape"])
        np.save("poses.npy", poses)  # in its working directory, for the test
        vertices = [v + [0.0, 0.0, 0.01 * request["seconds"]] for v in vertices]
        send({"ok": True})
    elif request["op"] == "cloths":
        send({"vertices": [
            {"array": base64.b64encode(v.tobytes()).decode(), "shape": list(v.shape)}
            for v in vertices]})
    elif request["op"] == "close":
        send({"ok": True})
        break
"""


def test_a_cloth_moves_beside_a_session(tmp_path, monkeypatch):
    """The Newton process (stood in for) gets the bodies where the session has them
    and moves the towel; the session draws the towel where it is."""
    _run(tmp_path, {"towel": make_transform(np.eye(3), [0.5, 0.0, 0.17])})
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / cloth_module.JOB).write_text(textwrap.dedent(FAKE_NEWTON))
    monkeypatch.setattr(cloth_module, "JOBS_DIR", jobs)
    monkeypatch.setattr(cloth_module, "env_python", lambda env: sys.executable)
    scene = SceneSpec.load(tmp_path / "scene")
    session = Session(scene, robot_or_skip("fr3_robotiq"))
    cloth = cloth_module.ClothSim(session, tmp_path / "cloth")
    try:
        assert cloth.device == "fake"
        towel = next(o for o in scene.objects if o.cloth)
        start = cloth_surface(towel)
        session.command(session.arm_q() + 0.1, 0.0)  # the arm moves
        session.step(10)
        cloth.step(10 * session.dt)
        moved = cloth.vertices()["towel"]
        assert np.allclose(moved, start + [0.0, 0.0, 0.01 * 10 * session.dt])
        sent = np.load(tmp_path / "cloth" / "poses.npy")
        mujoco.mj_kinematics(session.model, session.data)
        here = cloth_module.body_poses(session.data, cloth.ids, cloth.to_support)
        assert np.allclose(sent, here)  # where the session has them, after its step
        cloth.show()
        m = session.model
        g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body("towel").id)
        mesh = m.geom_dataid[g]
        local = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + len(start)]
        T_mesh = towel.T_base_obj @ make_transform(
            Rotation.from_quat(np.roll(m.geom_quat[g], -1)).as_matrix(), m.geom_pos[g]
        )
        assert np.allclose(transform_points(T_mesh, local), moved, atol=1e-5)
        with pytest.raises(ValueError, match="vertices"):
            session.show_surface("towel", moved[:-1])
    finally:
        cloth.close()
        session.close()


def _fake_newton(tmp_path, monkeypatch):
    """The Newton process stood in for (:data:`FAKE_NEWTON`)."""
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    (jobs / cloth_module.JOB).write_text(textwrap.dedent(FAKE_NEWTON))
    monkeypatch.setattr(cloth_module, "JOBS_DIR", jobs)
    monkeypatch.setattr(cloth_module, "env_python", lambda env: sys.executable)


def test_a_cloth_is_picked_in_the_scene(tmp_path, monkeypatch):
    """The pick program on the scene in MuJoCo, its towel beside it in Newton (stood in
    for: rising a centimetre a second): scored by its highest point; Newton stopped.
    Isaac Lab's pick and the CLI refuse a cloth before anything starts."""
    _run(tmp_path, {"towel": make_transform(np.eye(3), [0.5, 0.0, 0.17])})
    _fake_newton(tmp_path, monkeypatch)
    scene = SceneSpec.load(tmp_path / "scene")
    result = run_scene_pick(scene, "towel", tmp_path / "pick", video_camera=None)
    assert (
        result["deployment"] == "mujoco_scene" and result["policy"]["holding"] is None
    )
    rise = result["object_lift_m"]["towel"]
    seconds = len(json.loads((tmp_path / "pick/commands.json").read_text())["t"])
    assert rise == pytest.approx(0.01 * seconds * CONTROL_DT, rel=0.05)
    assert result["success"] is (rise > 0.05)
    assert (tmp_path / "pick/result.json").exists()
    with pytest.raises(ValueError, match="a cloth"):
        isaac.pick(tmp_path / "scene", "towel", tmp_path / "isaac")
    for argv in (["--target", "towel"], ["--sims", "scene"]):  # isaaclab; no target
        with pytest.raises(SystemExit):
            cli.main(["pick", str(tmp_path / "scene"), "--out", str(tmp_path), *argv])
