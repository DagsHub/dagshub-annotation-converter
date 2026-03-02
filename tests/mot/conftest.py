import pytest
from pathlib import Path

from dagshub_annotation_converter.formats.mot import MOTContext


@pytest.fixture
def mot_context() -> MOTContext:
    # CVAT MOT 1.1 format (9 columns):
    # frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    # Categories map class_id -> name (1-indexed as per CVAT)
    context = MOTContext(
        frame_rate=30.0,
        image_width=1920,
        image_height=1080,
        seq_name="test_sequence",
    )
    context.categories = {1: "person", 2: "car"}
    return context


@pytest.fixture
def sample_mot_file() -> Path:
    return Path(__file__).parent / "res" / "gt" / "gt.txt"


@pytest.fixture
def sample_labels_file() -> Path:
    return Path(__file__).parent / "res" / "gt" / "labels.txt"


@pytest.fixture
def sample_mot_dir() -> Path:
    return Path(__file__).parent / "res"


@pytest.fixture
def sample_seqinfo_file() -> Path:
    return Path(__file__).parent / "res" / "seqinfo.ini"
