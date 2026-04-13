from pathlib import Path
from subprocess import CompletedProcess

import pytest

from dagshub_annotation_converter.util.video import (
    _count_frames_slow,
    _probe_ffprobe,
    find_video_sibling,
)


def test_find_video_sibling_returns_match_with_same_stem(tmp_path: Path):
    reference = tmp_path / "sequence.txt"
    reference.write_text("annotations", encoding="utf-8")
    expected = tmp_path / "sequence.mp4"
    expected.write_bytes(b"video")

    assert find_video_sibling(reference) == expected


def test_find_video_sibling_returns_none_without_video_match(tmp_path: Path):
    reference = tmp_path / "sequence.txt"
    reference.write_text("annotations", encoding="utf-8")
    (tmp_path / "other.mp4").write_bytes(b"video")

    assert find_video_sibling(reference) is None


def test_find_video_sibling_ignores_non_video_files(tmp_path: Path):
    reference = tmp_path / "sequence.txt"
    reference.write_text("annotations", encoding="utf-8")
    (tmp_path / "sequence.csv").write_text("1,2,3", encoding="utf-8")

    assert find_video_sibling(reference) is None


def test_count_frames_slow_returns_zero_on_invalid_json(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(
        "dagshub_annotation_converter.util.video.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args=[], returncode=0, stdout="{", stderr=""),
    )

    assert _count_frames_slow(video_path) == 0


def test_probe_ffprobe_raises_value_error_on_invalid_json(monkeypatch, tmp_path: Path):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(
        "dagshub_annotation_converter.util.video.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args=[], returncode=0, stdout="{", stderr=""),
    )

    with pytest.raises(ValueError, match="ffprobe returned invalid JSON"):
        _probe_ffprobe(video_path)
