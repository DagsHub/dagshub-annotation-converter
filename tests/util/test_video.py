from pathlib import Path

from dagshub_annotation_converter.util.video import find_video_sibling


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
