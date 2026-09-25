"""Video and contact-sheet output for rollouts."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoRecorder:
    """Collect RGB frames; write an mp4 and a contact sheet of evenly spaced frames."""

    def __init__(self, path: str | Path, fps: float = 25.0) -> None:
        self.path = Path(path)
        self.fps = fps
        self.frames: list[NDArray[np.uint8]] = []

    def add(self, rgb: NDArray[np.uint8]) -> None:
        """Append one RGB frame."""
        self.frames.append(np.ascontiguousarray(rgb[..., :3]))

    def close(self, sheet_frames: int = 6, sheet_width: int = 1920) -> None:
        """Write the video and ``<stem>_sheet.jpg``."""
        if not self.frames:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter.fourcc(*"mp4v"), self.fps, (w, h)
        )
        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        idx = np.linspace(0, len(self.frames) - 1, sheet_frames).round().astype(int)
        cols = 3
        rows = int(np.ceil(len(idx) / cols))
        tile_w = sheet_width // cols
        tile_h = int(round(h * tile_w / w))
        sheet = np.zeros((rows * tile_h, cols * tile_w, 3), np.uint8)
        for k, i in enumerate(idx):
            tile = cv2.resize(
                self.frames[i], (tile_w, tile_h), interpolation=cv2.INTER_AREA
            )
            cv2.putText(
                tile,
                f"t={i / self.fps:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            r, c = divmod(k, cols)
            sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
        cv2.imwrite(
            str(self.path.with_name(f"{self.path.stem}_sheet.jpg")),
            cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        )
