import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def get_video_dimensions(video_path: Path) -> Tuple[int, int, float]:
    """Read frame width, height, and FPS from a video file.

    Uses ffprobe (zero Python dependencies) if available, then falls back to
    opencv (cv2) if installed.

    Raises ValueError if neither tool can read the file, or if dimensions are zero.
    """
    try:
        return _probe_ffprobe(video_path)
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        pass

    try:
        return _probe_cv2(video_path)
    except ImportError:
        pass

    raise ValueError(
        f"Could not read video dimensions from {video_path}. "
        f"Install ffmpeg (ffprobe) or opencv-python (cv2)."
    )


def _probe_ffprobe(video_path: Path) -> Tuple[int, int, float]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-print_format", "json",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise ValueError(f"ffprobe failed on {video_path}")

    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise ValueError(f"No video streams found in {video_path}")

    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    fps = 0.0
    r_frame_rate = stream.get("r_frame_rate", "")
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        if int(den) != 0:
            fps = int(num) / int(den)

    if width == 0 or height == 0:
        raise ValueError(f"Could not determine dimensions for {video_path}")

    return width, height, fps


def _probe_cv2(video_path: Path) -> Tuple[int, int, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if width == 0 or height == 0:
        raise ValueError(f"Could not determine dimensions for {video_path}")

    return width, height, fps


def find_video_sibling(reference_path: Path, name_stem: Optional[str] = None) -> Optional[Path]:
    """Look for a video file next to *reference_path* whose stem matches *name_stem*.

    If *name_stem* is None, the stem of *reference_path* is used.
    Returns the first match or None.
    """
    parent = reference_path.parent
    if not parent.is_dir():
        return None
    stem = name_stem or reference_path.stem
    for ext in sorted(VIDEO_EXTENSIONS):
        candidate = parent / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None
