import json
import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


class VideoProbeResult(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int = 0


@lru_cache
def probe_video(video_path: Path) -> VideoProbeResult:
    """Probe video file for dimensions, FPS, and frame count.

    Uses ffprobe if available, falls back to cv2.
    Raises ValueError if neither tool can read the file.
    """
    try:
        return _probe_ffprobe(video_path)
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        pass

    try:
        return _probe_cv2(video_path)
    except ImportError:
        pass

    raise ValueError(f"Could not read video info from {video_path}. Install ffmpeg (ffprobe) or opencv-python (cv2).")


def _parse_frame_count(stream: dict) -> int:
    """Extract a positive frame count from an ffprobe stream dict, or return 0."""
    raw = stream.get("nb_frames")
    if raw is not None:
        value = str(raw).strip()
        if value.isdigit():
            count = int(value)
            if count > 0:
                return count
    return 0


def _count_frames_slow(video_path: Path) -> int:
    """Run ffprobe with ``-count_frames`` to get an accurate frame count.

    This decodes every frame and can be slow on large files.
    Returns 0 if the count cannot be determined.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-print_format",
                "json",
                "-show_entries",
                "stream=nb_read_frames",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0

    if result.returncode != 0:
        return 0

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0
    streams = info.get("streams", [])
    if not streams:
        return 0

    raw = streams[0].get("nb_read_frames")
    if raw is not None:
        value = str(raw).strip()
        if value.isdigit():
            count = int(value)
            if count > 0:
                return count
    return 0


def _probe_ffprobe(video_path: Path) -> VideoProbeResult:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed on {video_path}")

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned invalid JSON for {video_path}") from exc
    streams = info.get("streams", [])
    if not streams:
        raise ValueError(f"No video streams found in {video_path}")

    stream = streams[0]

    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width == 0 or height == 0:
        raise ValueError(f"Could not determine dimensions for {video_path}")

    r_frame_rate = stream.get("r_frame_rate", "")
    # FPS is a fraction like "30000/1001"
    num, den = r_frame_rate.split("/")
    fps = int(num) / int(den)

    frame_count = _parse_frame_count(stream)
    if frame_count == 0:
        frame_count = _count_frames_slow(video_path)

    return VideoProbeResult(width=width, height=height, fps=fps, frame_count=frame_count)


def _probe_cv2(video_path: Path) -> VideoProbeResult:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width == 0 or height == 0:
        raise ValueError(f"Could not determine dimensions for {video_path}")

    return VideoProbeResult(width=width, height=height, fps=fps, frame_count=frame_count)


def find_video_sibling(reference_path: Path) -> Optional[Path]:
    """
    Look for a sibling video file with the same stem as *reference_path*.
    Returns None if there's no video file
    """
    parent = reference_path.parent
    if not parent.is_dir():
        return None
    stem = reference_path.stem
    files_by_name = {entry.name.lower(): entry for entry in parent.iterdir() if entry.is_file()}
    candidates = [f"{stem}{ext}".lower() for ext in VIDEO_EXTENSIONS]
    return next((files_by_name[candidate] for candidate in candidates if candidate in files_by_name), None)
