"""Cloths in Newton (its VBD solver), run in the newton env by r2s2r.sim.cloth: settling
on what is under them, or simulated beside a MuJoCo session (``--serve``).

    python cloth_job.py JOB.json RESULT.json      # settle
    python cloth_job.py --serve                   # requests on stdin, answers on stdout

JOB: {"input": NPZ, "output": NPZ, "seconds": 2.0,
      "cloths": [{"name", "density", "tri_ke", "tri_ka", "bending", "radius",
                  "friction"}, ...]}

Each cloth's numbers are r2s2r.sim.cloth.newton_material's: its mass per area, its
membrane's Lame parameters (N/m), its plate's bending stiffness D (N m), its contact
radius and friction. Everything is in the support's frame: the support is the plane
z = 0, gravity -z; a pose is x, y, z and a quaternion x, y, z, w (Warp's order). The
input NPZ holds the scene's bodies: ``body_pose``, ``body_mass``, ``body_com`` and
``body_inertia`` (about the centre of mass, the body's frame) and ``body_dynamic`` per
body; their convex colliders, in their bodies' frames (``hull_vertices`` and
``hull_faces`` of every hull one after another, ``hull_vertex_counts``,
``hull_face_counts``, ``hull_body``, ``hull_friction`` and ``hull_bodies`` per hull:
whether it touches the other bodies too, not only the cloths); ``support_friction``;
and each cloth's surface, ``cloth<i>_vertices`` and ``cloth<i>_faces``. Settling moves
the dynamic bodies with the cloths (each pushes the other) and holds the rest still;
the output NPZ has each cloth's vertices where they came to rest (``cloth<i>_vertices``,
in the same order) and every body's pose (``body_pose``).

Served, every body is held (kinematic) and moved where each request says. Requests
and answers are JSON, each after its length (4 bytes, little-endian); an array is
{"array": base64 of its float64 bytes, "shape": [...]}. Requests: {"op": "init",
"input": NPZ, "cloths": [...]}; {"op": "step", "seconds": dt, "body_pose": array}: the
bodies move there over dt, steadily, and the cloths with them; {"op": "cloths"}:
answered with {"vertices": [array, ...]}; {"op": "close"}. Each is answered (with
{"error": ...} on failure, its traceback in the log).

A cloth bends back toward flat: every edge's rest angle is 0, its stiffness the plate's
D |e| / (A1 + A2) (|e| the edge's length, A1 and A2 its triangles' areas; Newton's
bending energy is k |e| (angle - rest)^2 / 2, and on a mesh of even triangles bent onto
a cylinder this matches the plate's D A / 2 R^2 within 3%). It touches itself and the
other cloths.

Friction: Newton takes a contact's as the geometric mean of its two sides'. A shape the
cloths touch carries max(cloth, its own)^2 / cloth (``soft_contact_mu`` the cloth's),
so a cloth meets it with the higher of the two, as MuJoCo does; the bodies meet each
other with copies of their shapes carrying their own frictions (their geometric mean;
MuJoCo's is the higher); the cloths meet each other with the cloths' (the highest
cloth's, for several).
"""

import base64
import json
import os
import struct
import sys
import time
import traceback
from pathlib import Path

import newton  # pylint: disable=import-error
import numpy as np
import warp as wp  # pylint: disable=import-error

FPS, SUBSTEPS, ITERATIONS = 60, 10, 10  # VBD: 10 substeps a frame, 10 iterations each
DT = 1.0 / FPS / SUBSTEPS
CONTACT_KE, CONTACT_KD = 1.0e4, 1.0e-2  # contacts' stiffness and damping
CONTACT_GAP = 0.005  # m: contacts are looked for this far out
TRI_KD, EDGE_KD = 1.0e-2, 1.0e-4  # stretching's and bending's damping
# Contacts a body may have with other bodies (past the buffer they are dropped and the
# body jumps) and with cloth vertices (a towel over it); cloth contacts per vertex.
BODY_CONTACTS, BODY_CLOTH_CONTACTS, VERTEX_CONTACTS = 1024, 4096, 16


class Scene:
    """The cloths and the bodies of an input NPZ in Newton, and its VBD solver; every
    body held still, if ``held``."""

    def __init__(self, data, cloths, held=False):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        bending = []  # each edge's cloth's D, edge after edge
        self.counts = []  # each cloth's vertices
        for i, cloth in enumerate(cloths):
            edges = builder.edge_count
            vertices = data[f"cloth{i}_vertices"]
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=[wp.vec3(*v) for v in vertices],
                indices=data[f"cloth{i}_faces"].reshape(-1).tolist(),
                density=cloth["density"],
                tri_ke=cloth["tri_ke"],
                tri_ka=cloth["tri_ka"],
                tri_kd=TRI_KD,
                edge_kd=EDGE_KD,
                particle_radius=cloth["radius"],
                label=cloth["name"],
            )
            bending += [cloth["bending"]] * (builder.edge_count - edges)
            self.counts.append(len(vertices))
        cloth_mu = max(c["friction"] for c in cloths)

        def touching(friction, cloths_too):
            """A shape's settings: touching the cloths (else the other bodies)."""
            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.ke, cfg.kd, cfg.gap = CONTACT_KE, CONTACT_KD, CONTACT_GAP
            cfg.mu = max(cloth_mu, friction) ** 2 / cloth_mu if cloths_too else friction
            cfg.density = 0.0  # a body's mass and inertia are given
            cfg.has_particle_collision = cloths_too
            cfg.has_shape_collision = not cloths_too
            return cfg

        self.bodies = []
        dynamic = np.asarray(data["body_dynamic"], bool) & (not held)
        for pose, mass, com, inertia, moves in zip(
            data["body_pose"],
            data["body_mass"],
            data["body_com"],
            data["body_inertia"],
            dynamic,
        ):
            if not moves:  # held: its mass and inertia play no part
                mass, com, inertia = 0.0, np.zeros(3), np.zeros((3, 3))
            self.bodies.append(
                builder.add_body(
                    xform=wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:])),
                    mass=float(mass),
                    com=wp.vec3(*com),
                    inertia=wp.mat33(*np.asarray(inertia).reshape(-1)),
                    lock_inertia=True,
                    is_kinematic=not moves,
                )
            )
        v0 = f0 = 0
        for nv, nf, body, friction, bodies_too in zip(
            data["hull_vertex_counts"],
            data["hull_face_counts"],
            data["hull_body"],
            data["hull_friction"],
            data["hull_bodies"],
        ):
            mesh = newton.Mesh(
                data["hull_vertices"][v0 : v0 + nv].astype(np.float32),
                data["hull_faces"][f0 : f0 + nf].reshape(-1).astype(np.int32),
                compute_inertia=False,
            )
            for cloths_too in (True, False) if bodies_too else (True,):
                builder.add_shape_convex_hull(
                    self.bodies[int(body)],
                    mesh=mesh,
                    cfg=touching(float(friction), cloths_too),
                )
            v0, f0 = v0 + nv, f0 + nf
        for cloths_too in (True, False):  # the support: unbounded (width 0)
            builder.add_shape_plane(
                width=0.0,
                length=0.0,
                cfg=touching(float(data["support_friction"]), cloths_too),
            )
        builder.color(include_bending=True)
        model = builder.finalize(wp.get_preferred_device())
        model.soft_contact_ke, model.soft_contact_kd = CONTACT_KE, CONTACT_KD
        model.soft_contact_mu = cloth_mu
        _bend_toward_flat(model, np.array(bending))
        thickness = 2 * max(c["radius"] for c in cloths)
        self.solver = newton.solvers.SolverVBD(
            model,
            iterations=ITERATIONS,
            rigid_compliant_alm=True,
            rigid_body_contact_buffer_size=BODY_CONTACTS,
            rigid_body_particle_contact_buffer_size=BODY_CLOTH_CONTACTS,
            particle_enable_self_contact=True,
            particle_self_contact_margin=thickness,
            particle_self_contact_gap=thickness,
            collision_pipeline=newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                soft_contact_gap=CONTACT_GAP,
                soft_contact_max=VERTEX_CONTACTS * sum(self.counts),
                # Held bodies (the robot's links, the objects' a body may rest on)
                # need no contacts among themselves.
                include_static_kinematic_pairs=False,
            ),
        )
        self.model = model
        self.state, self.after = model.state(), model.state()
        self.control = model.control()

    def advance(self, seconds, poses=None):
        """``seconds`` of physics; the held bodies move steadily to ``poses`` (each
        body's, in order) as it goes, when given (at once, for no time)."""
        if seconds <= 0:
            if poses is not None:
                self.state.body_q.assign(np.asarray(poses, np.float32))
            return
        n = max(1, int(round(seconds / DT)))
        start = self.state.body_q.numpy()
        for k in range(1, n + 1):
            if poses is not None:
                self.state.body_q.assign(_between(start, poses, k / n))
            self.state.clear_forces()
            self.solver.step(self.state, self.after, self.control, None, seconds / n)
            self.state, self.after = self.after, self.state

    def cloths(self):
        """Each cloth's vertices, in order."""
        q = self.state.particle_q.numpy().astype(float)
        out, first = [], 0
        for n in self.counts:
            out.append(q[first : first + n])
            first += n
        return out

    def cloth_speeds(self):
        """Each cloth's fastest vertex's speed."""
        qd = np.linalg.norm(self.state.particle_qd.numpy(), axis=1)
        out, first = [], 0
        for n in self.counts:
            out.append(float(qd[first : first + n].max()))
            first += n
        return out


def _between(start, end, share):
    """Poses ``share`` of the way from ``start`` to ``end`` (positions straight on,
    rotations normalised linearly: steps are small); VBD takes a held body's velocity
    from where it was."""
    q = np.array(start, float)
    q[:, :3] = start[:, :3] + share * (end[:, :3] - start[:, :3])
    sign = np.where(np.sum(end[:, 3:] * start[:, 3:], axis=1, keepdims=True) < 0, -1, 1)
    rot = start[:, 3:] + share * (sign * end[:, 3:] - start[:, 3:])
    q[:, 3:] = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    return q.astype(np.float32)


def _bend_toward_flat(model, bending):
    """Every edge at rest flat, its stiffness its cloth's plate's (``bending``, D per
    edge) as the module doc says; an edge on the border bends nothing."""
    edges = model.edge_indices.numpy()  # opposite vertex, opposite vertex, v0, v1
    x = model.particle_q.numpy()
    inner = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    o0, o1, a, b = (np.where(inner, edges[:, k], 0) for k in range(4))
    side = x[b] - x[a]
    areas = 0.5 * (
        np.linalg.norm(np.cross(side, x[o0] - x[a]), axis=1)
        + np.linalg.norm(np.cross(side, x[o1] - x[a]), axis=1)
    )
    length = np.linalg.norm(side, axis=1)
    k = np.where(inner, bending * length / np.maximum(areas, 1e-12), 0.0)
    props = model.edge_bending_properties.numpy()
    props[:, 0] = k
    model.edge_bending_properties.assign(props)
    model.edge_rest_angle.zero_()


def settle(job_path, result_path):
    """Run the settling job in ``job_path``; write its result to ``result_path``."""
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    scene = Scene(np.load(job["input"]), job["cloths"])
    start = time.perf_counter()
    scene.advance(job["seconds"])
    out = {f"cloth{i}_vertices": v for i, v in enumerate(scene.cloths())}
    out["body_pose"] = scene.state.body_q.numpy().astype(float)
    np.savez(job["output"], **out)
    result = {
        "output": job["output"],
        "cloths": [
            {"name": c["name"], "speed_m_s": round(s, 4)}
            for c, s in zip(job["cloths"], scene.cloth_speeds())
        ],
        "device": str(scene.model.device),
        "seconds_wall": round(time.perf_counter() - start, 1),
    }
    Path(result_path).write_text(json.dumps(result, indent=1), encoding="utf-8")


def serve():
    """Answer requests on stdin until one says close (the module doc)."""
    answers = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything else printed goes to the log, not the answers
    requests = sys.stdin.buffer
    scene = None
    while True:
        head = requests.read(4)
        if len(head) < 4:
            return
        request = json.loads(requests.read(struct.unpack("<I", head)[0]))
        try:
            answer = {"ok": True}
            if request["op"] == "init":
                scene = Scene(np.load(request["input"]), request["cloths"], held=True)
                answer["device"] = str(scene.model.device)
            elif request["op"] == "step":
                scene.advance(request["seconds"], _array(request["body_pose"]))
            elif request["op"] == "cloths":
                answer["vertices"] = [_encode(v) for v in scene.cloths()]
            elif request["op"] == "close":
                _send(answers, answer)
                return
        except Exception as exc:  # pylint: disable=broad-except
            traceback.print_exc()
            answer = {"error": f"{type(exc).__name__}: {exc}"}
        _send(answers, answer)


def _array(d):
    return np.frombuffer(base64.b64decode(d["array"]), float).reshape(d["shape"])


def _encode(a):
    a = np.ascontiguousarray(a, float)
    return {"array": base64.b64encode(a.tobytes()).decode(), "shape": list(a.shape)}


def _send(stream, message):
    body = json.dumps(message).encode()
    stream.write(struct.pack("<I", len(body)) + body)
    stream.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--serve"]:
        serve()
    else:
        settle(*sys.argv[1:3])
