"""The viewer's HTTP server.

r2s2r viewer PATH [PATH ...] [--port 8765] [--host 0.0.0.0]

Each PATH is a run (any method's: a directory with ``run.json``), or a directory whose
sub-directories are runs (new ones show up as they are made). Routes, each for the run
``?run=<name>`` (default: the first): ``/`` the page; ``/api/runs``, ``/api/recording``,
``/api/state`` (JSON); ``/api/robot.json`` and ``/robot.glb`` (the robot along the
trajectory); ``/f/<path>`` a file of the run; ``/thumb/<path>?w=N`` an image scaled down
(JPEG); ``/glb/<path>`` a mesh as GLB; ``/preview/<path>?up=y`` four views of a mesh
(PNG) turned up-axis up; ``/scene.glb?path=<path>`` a scene directory or objects file,
posed, as GLB. Paths are relative to the run, and nothing outside it can be read through
them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import cv2

from r2s2r.pipeline.workspace import RUN_FILENAME, Workspace
from r2s2r.robots import get_robot
from r2s2r.viewer.robot import robot_glb, robot_poses
from r2s2r.viewer.scenes import mesh_glb, mesh_preview, scene_glb
from r2s2r.viewer.state import recording, run_state

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Viewer:
    """A run, and caches of what is slow to make (on disk, beside the run, so an agent
    never sees them)."""

    def __init__(self, root: Path, cache: Path | None = None) -> None:
        self.ws = Workspace.load(root)
        self.root = self.ws.root
        self.cache = cache or self.root.parent / ".viewer_cache" / self.root.name
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def resolve(self, rel: str) -> Path:
        """A path inside the run (or its cache), else ``PermissionError``."""
        path = (self.root / unquote(rel)).resolve()
        if not (path.is_relative_to(self.root) or path.is_relative_to(self.cache)):
            raise PermissionError(rel)
        return path

    def cached(self, key: str, suffix: str, make: Callable[[], bytes]) -> Path:
        """``make()``'s bytes, made once per ``key``."""
        path = self.cache / f"{hashlib.sha1(key.encode()).hexdigest()}{suffix}"
        if not path.exists():
            with self.lock:
                if not path.exists():
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(make())
                    tmp.replace(path)
        return path

    def robot(self) -> tuple[Path, Path]:
        """The robot's GLB and its poses along the trajectory."""
        robot = get_robot(self.ws.capture.embodiment)
        glb_path = self.cache / f"robot_{robot.name}.glb"
        names_path = self.cache / f"robot_{robot.name}.json"
        poses_path = self.cache / f"robot_{robot.name}_poses.json"
        with self.lock:
            if not glb_path.exists():
                glb, names = robot_glb(robot)
                glb_path.write_bytes(glb)
                names_path.write_text(json.dumps(names))
            if not poses_path.exists():
                names = json.loads(names_path.read_text())
                poses = robot_poses(robot, self.ws.capture, names)
                poses_path.write_text(json.dumps(poses))
        return glb_path, poses_path


class Viewers:
    """The runs served, found afresh on each request."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = [Path(p).resolve() for p in paths]
        self.open: dict[Path, Viewer] = {}
        self.lock = threading.Lock()

    def roots(self) -> list[Path]:
        """Every run under the paths, in order."""
        found = []
        for path in self.paths:
            if (path / RUN_FILENAME).exists():
                found.append(path)
            elif path.is_dir():
                found += sorted(
                    d for d in path.iterdir() if (d / RUN_FILENAME).exists()
                )
        return found

    def get(self, name: str | None) -> Viewer:
        """The run called ``name`` (default: the first)."""
        roots = self.roots()
        if not roots:
            raise FileNotFoundError("no run")
        root = (
            next((r for r in roots if r.name == name), roots[0]) if name else roots[0]
        )
        with self.lock:
            if root not in self.open:
                self.open[root] = Viewer(root)
            return self.open[root]


def make_handler(viewers: Viewers) -> type[BaseHTTPRequestHandler]:
    """The request handler for ``viewers``."""

    class Handler(BaseHTTPRequestHandler):
        """Serves the page, the APIs and the run's files."""

        def log_message(self, *args: Any) -> None:  # pylint: disable=arguments-differ
            logger.debug(*args)

        def do_GET(self) -> None:  # pylint: disable=invalid-name
            """Dispatch on the path."""
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/api/runs":
                    self._json([r.name for r in viewers.roots()])
                    return
                if url.path in ("/", "/index.html"):
                    self._file(STATIC / "index.html")
                    return
                if url.path.startswith("/static/"):
                    self._file((STATIC / url.path[len("/static/") :]).resolve(), STATIC)
                    return
                viewer = viewers.get(query.get("run", [None])[0])
                if url.path == "/api/recording":
                    self._json(recording(viewer.ws))
                elif url.path == "/api/state":
                    self._json(run_state(viewer.ws))
                elif url.path == "/api/robot.json":
                    self._file(viewer.robot()[1])
                elif url.path == "/robot.glb":
                    self._file(viewer.robot()[0])
                elif url.path.startswith("/f/"):
                    self._file(viewer.resolve(url.path[3:]))
                elif url.path.startswith("/thumb/"):
                    self._thumb(
                        viewer,
                        viewer.resolve(url.path[7:]),
                        int(query.get("w", ["480"])[0]),
                    )
                elif url.path.startswith("/glb/"):
                    path = viewer.resolve(url.path[5:])
                    if path.suffix.lower() == ".glb":
                        self._file(path)
                    else:
                        key = f"mesh:{path}:{path.stat().st_mtime}"
                        self._file(viewer.cached(key, ".glb", lambda: mesh_glb(path)))
                elif url.path.startswith("/preview/"):
                    path = viewer.resolve(url.path[9:])
                    up = query.get("up", ["z"])[0]
                    key = f"preview:{path}:{path.stat().st_mtime}:{up}"
                    self._send(
                        viewer.cached(
                            key, ".png", lambda: mesh_preview(path, up)
                        ).read_bytes(),
                        "image/png",
                        True,
                    )
                elif url.path == "/scene.glb":
                    path = viewer.resolve(query["path"][0])
                    marker = path / "scene.json" if path.is_dir() else path
                    key = f"scene:{path}:{marker.stat().st_mtime}"
                    self._file(viewer.cached(key, ".glb", lambda: scene_glb(path)[0]))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
            except PermissionError:
                self.send_error(HTTPStatus.FORBIDDEN)
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("GET %s failed", self.path)
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)[:200])

        def _send(self, body: bytes, content_type: str, cache: bool = False) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any) -> None:
            self._send(json.dumps(payload).encode(), "application/json")

        def _file(self, path: Path, within: Path | None = None) -> None:
            if within is not None and not path.is_relative_to(within):
                raise PermissionError(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix == ".glb":
                kind = "model/gltf-binary"
            self._send(path.read_bytes(), kind)

        def _thumb(self, viewer: Viewer, path: Path, width: int) -> None:
            if not path.is_file():
                raise FileNotFoundError(path)
            key = f"thumb:{path}:{path.stat().st_mtime}:{width}"

            def make() -> bytes:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise FileNotFoundError(path)
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:  # transparent: on a checkerboard-ish grey
                    alpha = image[..., 3:4].astype(float) / 255
                    image = (image[..., :3] * alpha + 60 * (1 - alpha)).astype("uint8")
                if image.shape[1] > width:
                    h = int(round(image.shape[0] * width / image.shape[1]))
                    image = cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
                ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                assert ok
                return bytes(data)

            self._send(
                viewer.cached(key, ".jpg", make).read_bytes(), "image/jpeg", True
            )

    return Handler


def serve(paths: list[Path], host: str = "0.0.0.0", port: int = 8765) -> None:
    """Serve the viewer until interrupted."""
    viewers = Viewers(paths)
    server = ThreadingHTTPServer((host, port), make_handler(viewers))
    names = ", ".join(r.name for r in viewers.roots()) or "none yet"
    print(f"viewer of {names} on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
