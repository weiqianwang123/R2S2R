"""Tests for r2s2r.viewer: what it reads of a run, and what its server serves."""

import json
import struct
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest
from conftest import hinged_box, rgbd_capture

from r2s2r.structs import Capture
from r2s2r.tools.objects import assemble
from r2s2r.transforms import make_transform
from r2s2r.viewer.scenes import scene_glb
from r2s2r.viewer.server import Viewers, make_handler
from r2s2r.viewer.state import run_state
from r2s2r.workspace import Workspace


def _run(tmp_path, name="run", method="agentic"):
    """A new run of ``method`` on a one-camera capture (made once per test)."""
    root = tmp_path / "capture"
    capture = (
        Capture.load(root)
        if (root / "capture.json").exists()
        else rgbd_capture(root, {"c": ("ext1", np.eye(4))}, steps=3)
    )
    return Workspace.create(capture, tmp_path / name, method)


def _events(path, *events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events))


def _stages(ws, **entries):
    run = ws.read_run()
    run["stages"].update(entries)
    ws.write_run(run)


def test_state_reads_stage_status_and_products(tmp_path):
    """Stage 2 done (frames and support to look at); stage 3 running, with the agent's
    last message as its activity; stage 4 not started."""
    ws = _run(tmp_path)
    s2 = ws.root / "s2_frames"
    s2.mkdir()
    (s2 / "support.json").write_text(
        json.dumps(
            {"T_base_support": np.eye(4).tolist(), "extent": [1, 1], "tilt_deg": 0.5}
        )
    )
    (s2 / "support_ext1@0.png").write_bytes(b"")
    (s2 / "output.json").write_text(
        json.dumps(
            {
                "frames": ["ext1@0", "ext1@1", "ext1@2", "ext1@0"],
                "frame_notes": {"ext1@1": "sharp"},
                "support": {"file": "support.json", "description": "a table"},
                "objects": [],
            }
        )
    )
    _events(
        ws.root / "s3_objects" / "codex_0.jsonl",
        {"type": "thread.started", "thread_id": "y"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "Fitting the [mug](/x/y.png) now. Then more.",
            },
        },
    )
    now = round(time.time(), 1)
    _stages(
        ws,
        **{
            "2": {"status": "done", "started": now - 60, "seconds": 50},
            "3": {"status": "running", "started": now},
        },
    )
    state = run_state(ws)
    stages = {s["key"]: s for s in state["stages"]}
    assert state["method"] == "agentic"
    assert stages["2"]["status"] == "done" and stages["2"]["seconds"] == 50
    assert stages["3"]["status"] == "running" and stages["3"]["started"] == now
    assert stages["3"]["activity"] == "Fitting the mug now."
    assert stages["4"]["status"] == "pending"
    assert state["frames"][1] == {"id": "ext1@1", "note": "sharp", "selected": False}
    assert state["support"]["extent"] == [1, 1]
    assert state["support"]["overlays"] == ["s2_frames/support_ext1@0.png"]
    assert not state["objects"] and not state["physics"]
    assert state["scenes"][0]["path"] == "s2_frames/support.json"

    (s2 / "output.json").write_text("{}")  # no longer valid
    assert run_state(ws)["stages"][0]["status"] == "stopped"


def test_state_of_another_methods_run(tmp_path):
    """Stages the method leaves out are skipped; a failed stage says why; objects
    carry their up axis (generated meshes, before objects.json, are y-up)."""
    ws = _run(tmp_path, method="fixed")
    _stages(
        ws,
        **{
            "2": {"status": "failed", "error": "RuntimeError: SimFoundry stage 3"},
            "6": {"status": "skipped"},
        },
    )
    s3 = ws.root / "s3_objects"
    (s3 / "gen" / "mug").mkdir(parents=True)
    (s3 / "gen" / "mug" / "mesh.glb").write_bytes(b"")
    (s3 / "gen" / "mug" / "preview.png").write_bytes(b"")
    state = run_state(ws)
    stages = {s["key"]: s for s in state["stages"]}
    assert stages["2"]["status"] == "failed" and "SimFoundry" in stages["2"]["error"]
    assert stages["6"]["status"] == "skipped"
    assert state["objects"][0]["up"] == "y"
    (s3 / "objects.json").write_text(
        json.dumps(
            {
                "objects": [
                    {"name": "mug", "mesh": "gen/mug/mesh.glb"},
                    {"name": "box", "mesh": "box.obj", "up": "z"},
                ]
            }
        )
    )
    objects = run_state(ws)["objects"]
    assert [o["up"] for o in objects] == ["z", "z"]  # z when not said
    assert objects[0]["glb"] == "s3_objects/gen/mug/mesh.glb" and not objects[1]["glb"]


def test_state_of_a_fixed_run(tmp_path):
    """SimFoundry's stage is a running fixed stage's activity; the frames say which
    was selected; the support's extent is the refined one of the objects file; the
    objects show their previews; the unrefined scene can be looked at."""
    ws = _run(tmp_path, method="fixed")
    s2, s3 = ws.root / "s2_frames", ws.root / "s3_objects"
    s2.mkdir()
    (s2 / "simfoundry.log").write_text(
        "[Stage 3] Segment ground plane\n[Stage 3] completed in 70.1s\n"
        "[Stage 5] Decompose scene\n[Stage 5] cmd: mamba run -n simfoundry python\n"
        "Detecting objects...\n"
    )
    _stages(ws, **{"2": {"status": "running", "started": time.time()}})
    stages = {s["key"]: s for s in run_state(ws)["stages"]}
    assert stages["2"]["activity"] == "SimFoundry Stage 5: Decompose scene"

    (s2 / "support.json").write_text(
        json.dumps({"T_base_support": np.eye(4).tolist(), "tilt_deg": 1.5})
    )
    (s2 / "output.json").write_text(
        json.dumps(
            {
                "frames": ["ext1@0", "ext1@1"],
                "selected": "ext1@1",
                "decided_by": "codex",
                "frame_notes": {"ext1@1": "sees every object"},
                "support": {"file": "support.json", "description": "a table"},
            }
        )
    )
    (s2 / "parsed").mkdir()
    (s2 / "parsed" / "scene.json").write_text("{}")
    (s3 / "mug").mkdir(parents=True)
    (s3 / "mug" / "mesh.obj").write_text("v 0 0 0\n")
    (s3 / "mug" / "preview.png").write_bytes(b"")
    support = {"T_base_support": np.eye(4).tolist(), "extent": [0.4, 0.44]}
    (s3 / "objects.json").write_text(
        json.dumps(
            {
                "support": support,
                "objects": [{"name": "mug", "mesh": "mug/mesh.obj", "up": "z"}],
            }
        )
    )
    state = run_state(ws)
    assert state["frames"] == [
        {"id": "ext1@0", "note": "", "selected": False},
        {"id": "ext1@1", "note": "sees every object", "selected": True},
    ]
    assert state["support"]["extent"] == [0.4, 0.44]
    assert state["support"]["tilt_deg"] == 1.5
    assert state["support"]["description"] == "a table"
    assert len(state["objects"]) == 1
    obj = state["objects"][0]
    assert obj["preview"] == "s3_objects/mug/preview.png" and obj["up"] == "z"
    assert "s2_frames/parsed" in [s["path"] for s in state["scenes"]]


def test_articulated_objects_show_their_joints(tmp_path):
    """An articulated object: its joints counted in the objects, listed (with the
    method's reasons) in the physics, and in its scene GLB's extras, in the base frame,
    with the nodes they move."""
    ws = _run(tmp_path)
    T = make_transform(np.eye(3), [0.5, 0.0, 0.0])
    s3 = ws.root / "s3_objects"
    s3.mkdir()
    hinged_box(s3, T)
    s4 = ws.root / "s4_scene"
    s4.mkdir()
    (s4 / "objects.json").write_text((s3 / "objects.json").read_text())
    for mesh in ("body.obj", "lid.obj"):
        (s4 / mesh).write_bytes((s3 / mesh).read_bytes())
    assemble(ws, s4 / "objects.json", s4 / "scene", "none")
    reason = "a lid hinged at the back"
    (s4 / "output.json").write_text(
        json.dumps({"objects": {"box": {"mass": 0.4, "joints": {"hinge": reason}}}})
    )
    state = run_state(ws)
    assert [(o["joints"], o["model"]) for o in state["objects"]] == [
        (1, "s3_objects/objects.json")
    ]
    (joint,) = state["physics"][0]["joints"]
    assert joint["limits"] == [-1.9, 0.0] and joint["why"] == reason
    assert joint["damping"] is None  # the objects file gives no dynamics: free

    for path in (s3 / "objects.json", s4 / "scene"):
        glb = scene_glb(path, "box")
        size = struct.unpack("<I", glb[12:16])[0]
        gltf = json.loads(glb[20 : 20 + size])
        (joint,) = gltf["scenes"][0]["extras"]["joints"]
        assert joint["nodes"] == ["box__lid__0"]
        assert joint["origin"] == pytest.approx([0.5, 0.05, 0.06], abs=1e-6)
        assert joint["axis"] == pytest.approx([1, 0, 0])
        assert joint["position"] == pytest.approx(-0.8)
        assert "support" not in [n.get("name") for n in gltf["nodes"]]


def test_server_serves_the_run_and_nothing_else(tmp_path):
    """Runs are found under a parent; files outside them are refused."""
    _run(tmp_path, "run_a")
    _run(tmp_path, "run_b")
    (tmp_path / "secret.txt").write_text("no")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Viewers([tmp_path])))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return r.status, r.headers["Content-Type"], r.read()

    try:
        assert json.loads(get("/api/runs")[2]) == ["run_a", "run_b"]
        assert json.loads(get("/api/recording?run=run_b")[2])["name"] == "t"
        state = json.loads(get("/api/state?run=run_a")[2])
        assert [s["status"] for s in state["stages"]] == ["pending"] * 5
        status, kind, body = get("/thumb/inputs/capture/c/1.png?w=160&run=run_a")
        assert status == 200 and kind == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR).shape == (
            120,
            160,
            3,
        )
        assert get("/f/inputs/frames.json?run=run_a")[0] == 200
        assert b"<title>" in get("/")[2]
        for bad in ("/f/..%2Fsecret.txt?run=run_a", "/f/%2Fetc%2Fpasswd?run=run_a"):
            with pytest.raises(urllib.error.HTTPError) as err:
                get(bad)
            assert err.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.gl
def test_robot_poses_follow_the_trajectory(tmp_path):
    """One pose per trajectory step per body; the gripper moves with the joints."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.robots import get_robot
    from r2s2r.viewer.robot import robot_glb, robot_poses

    ws = _run(tmp_path)
    robot = get_robot(ws.capture.embodiment)
    glb, bodies = robot_glb(robot)
    assert glb[:4] == b"glTF" and "gripper/base" in bodies
    ws.capture.trajectory.joint_positions[2, 0] += 0.5  # base joint turned at step 2
    poses = robot_poses(robot, ws.capture, bodies)
    hand = bodies.index("gripper/base")
    assert len(poses["poses"]) == 3 and len(poses["poses"][0]) == len(bodies)
    assert poses["poses"][0][hand] == poses["poses"][1][hand]
    assert poses["poses"][0][hand][:3] != poses["poses"][2][hand][:3]
