"""Tests for r2s2r.viewer: what it reads of a workspace, and what its server serves."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from r2s2r.agentic.workspace import Workspace
from r2s2r.io.rgbd import write_depth
from r2s2r.structs import CameraSpec, Capture, FrameRecord, RobotTrajectory
from r2s2r.transforms import intrinsics_matrix
from r2s2r.viewer.server import Viewers, make_handler
from r2s2r.viewer.state import workspace_state

K = intrinsics_matrix(300.0, 300.0, 159.5, 119.5)
JOINTS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])


def _workspace(tmp_path, name="ws"):
    root = tmp_path / "capture"
    (root / "c").mkdir(parents=True, exist_ok=True)
    frames = []
    for step in range(3):
        cv2.imwrite(
            str(root / f"c/{step}.png"), np.full((240, 320, 3), 40 * step, np.uint8)
        )
        write_depth(root / f"c/{step}_d.png", np.full((240, 320), 0.8))
        frames.append(
            FrameRecord(
                step,
                "c",
                f"c/{step}.png",
                None,
                np.eye(4),
                JOINTS,
                0.0,
                f"c/{step}_d.png",
            )
        )
    cam = CameraSpec("c", "ext1", 320, 240, K, T_base_cam=np.eye(4))
    trajectory = RobotTrajectory(
        np.arange(3), np.arange(3) * 0.1, np.tile(JOINTS, (3, 1)), np.zeros(3)
    )
    capture = Capture(
        "t",
        "test",
        "franka_panda",
        "pick",
        {"c": cam},
        frames,
        (0, 3),
        root,
        trajectory=trajectory,
    )
    capture.save()
    return Workspace.create(capture, tmp_path / name)


def _events(path, *events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events))


def test_state_reads_stage_status_and_products(tmp_path):
    """A finished stage 2 (frames and support to look at); stage 3 still running, with
    the agent's last message as its activity."""
    ws = _workspace(tmp_path)
    s2 = ws.root / "s2_frames"
    _events(
        s2 / "codex_0.jsonl",
        {"type": "thread.started", "thread_id": "x"},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}},
    )
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
    state = workspace_state(ws)
    stages = {s["key"]: s for s in state["stages"]}
    assert stages["2"]["status"] == "done"
    assert stages["3"]["status"] == "running"
    assert stages["3"]["activity"] == "Fitting the mug now."
    assert stages["4"]["status"] == "pending"
    assert state["frames"][1] == {"id": "ext1@1", "note": "sharp"}
    assert state["support"]["extent"] == [1, 1]
    assert state["support"]["overlays"] == ["s2_frames/support_ext1@0.png"]
    assert not state["objects"] and not state["physics"]
    assert state["scenes"][0]["path"] == "s2_frames/support.json"


def test_server_serves_the_workspace_and_nothing_else(tmp_path):
    """Workspaces are found under a parent; files outside them are refused."""
    _workspace(tmp_path, "ws_a")
    _workspace(tmp_path, "ws_b")
    (tmp_path / "secret.txt").write_text("no")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Viewers([tmp_path])))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path) as r:
            return r.status, r.headers["Content-Type"], r.read()

    try:
        assert json.loads(get("/api/workspaces")[2]) == ["ws_a", "ws_b"]
        assert json.loads(get("/api/recording?ws=ws_b")[2])["name"] == "t"
        status, kind, body = get("/thumb/inputs/capture/c/1.png?w=160&ws=ws_a")
        assert status == 200 and kind == "image/jpeg"
        assert cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR).shape == (
            120,
            160,
            3,
        )
        assert get("/f/inputs/frames.json?ws=ws_a")[0] == 200
        assert b"<title>" in get("/")[2]
        for bad in ("/f/..%2Fsecret.txt?ws=ws_a", "/f/%2Fetc%2Fpasswd?ws=ws_a"):
            with pytest.raises(urllib.error.HTTPError) as err:
                get(bad)
            assert err.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.gl
def test_robot_poses_follow_the_trajectory(tmp_path):
    """One pose per trajectory step per body; the hand moves with the joints."""
    # pylint: disable=import-outside-toplevel
    from r2s2r.viewer.robot import robot_glb, robot_poses

    ws = _workspace(tmp_path)
    glb, bodies = robot_glb("franka_panda")
    assert glb[:4] == b"glTF" and "hand" in bodies
    ws.capture.trajectory.joint_positions[2, 0] += 0.5  # base joint turned at step 2
    poses = robot_poses(ws.capture, bodies)
    hand = bodies.index("hand")
    assert len(poses["poses"]) == 3 and len(poses["poses"][0]) == len(bodies)
    assert poses["poses"][0][hand] == poses["poses"][1][hand]
    assert poses["poses"][0][hand][:3] != poses["poses"][2][hand][:3]
