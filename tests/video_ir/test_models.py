import math

import pytest

from dagshub_annotation_converter.ir.video import (
    CoordinateStyle,
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
    IRVideoSequence,
    track_id_from_identifier,
)


def test_annotation_with_optional_fields():
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=5,
        keyframe=False,
        left=100,
        top=150,
        width=50,
        height=120,
        video_width=1920,
        video_height=1080,
        categories={"car": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        timestamp=0.167,
        visibility=0.8,
    )

    assert ann.timestamp == 0.167
    assert ann.visibility == 0.8
    assert not ann.keyframe


def test_with_filename():
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=100,
        top=150,
        width=50,
        height=120,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )

    ann_with_filename = ann.with_filename("frame_0001.jpg")
    assert ann_with_filename.filename == "frame_0001.jpg"


def test_normalize_denormalize_require_dimensions():
    denorm = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=100,
        top=100,
        width=50,
        height=50,
        video_width=None,
        video_height=None,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )
    with pytest.raises(ValueError, match="Cannot normalize/denormalize video annotation"):
        denorm.normalized()

    norm = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=0.1,
        top=0.1,
        width=0.2,
        height=0.2,
        video_width=None,
        video_height=None,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.NORMALIZED,
    )
    with pytest.raises(ValueError, match="Cannot normalize/denormalize video annotation"):
        norm.denormalized()


def test_denormalize_coordinates(epsilon):
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=10,
        left=0.1,
        top=0.1,
        width=0.2,
        height=0.2,
        video_width=1920,
        video_height=1080,
        categories={"car": 1.0},
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    denormalized = ann.denormalized()

    assert denormalized.coordinate_style == CoordinateStyle.DENORMALIZED
    assert math.isclose(denormalized.left, 192, abs_tol=epsilon)
    assert math.isclose(denormalized.top, 108, abs_tol=epsilon)
    assert math.isclose(denormalized.width, 384, abs_tol=epsilon)
    assert math.isclose(denormalized.height, 216, abs_tol=epsilon)
    assert denormalized.frame_number == 10


def test_normalize_coordinates(epsilon):
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=192,
        top=108,
        width=384,
        height=216,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )

    normalized = ann.normalized()

    assert normalized.coordinate_style == CoordinateStyle.NORMALIZED
    assert math.isclose(normalized.left, 0.1, abs_tol=epsilon)
    assert math.isclose(normalized.top, 0.1, abs_tol=epsilon)
    assert math.isclose(normalized.width, 0.2, abs_tol=epsilon)
    assert math.isclose(normalized.height, 0.2, abs_tol=epsilon)
    assert normalized.frame_number == 0


def test_video_track_uses_track_level_identifier():
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=3,
        left=100,
        top=150,
        width=50,
        height=120,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )

    track = IRVideoAnnotationTrack.from_annotations([ann], id="7")

    assert track.id == "7"
    assert track.track_id == 7
    assert track.annotations[0].imported_id == "7"


def test_video_track_uses_hash_for_non_numeric_identifier():
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=0.1,
        top=0.2,
        width=0.05,
        height=0.1,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    track = IRVideoAnnotationTrack.from_annotations([ann], id="track_person_1")
    materialized = track.to_annotations()

    assert track.track_id == track_id_from_identifier("track_person_1")
    assert materialized[0].imported_id == "track_person_1"


def test_video_sequence_resolves_sequence_metadata():
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=4,
        left=10,
        top=20,
        width=30,
        height=40,
        video_width=None,
        video_height=None,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )
    track = IRVideoAnnotationTrack.from_annotations([ann], id="2")
    sequence = IRVideoSequence(
        tracks=[track],
        filename="video.mp4",
        sequence_length=40,
        video_width=1280,
        video_height=720,
    )

    flattened = sequence.to_annotations()

    assert sequence.resolved_sequence_length() == 40
    assert flattened[0].filename == "video.mp4"
    assert flattened[0].video_width == 1280
    assert flattened[0].video_height == 720


def test_video_sequence_groups_annotations_by_frame():
    ann_a = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=10,
        top=20,
        width=30,
        height=40,
        video_width=1280,
        video_height=720,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )
    ann_b = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=50,
        top=60,
        width=70,
        height=80,
        video_width=1280,
        video_height=720,
        categories={"car": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )

    sequence = IRVideoSequence(
        tracks=[
            IRVideoAnnotationTrack.from_annotations([ann_a], id="1"),
            IRVideoAnnotationTrack.from_annotations([ann_b], id="vehicle"),
        ]
    )

    grouped = sequence.annotations_by_frame()

    assert list(grouped.keys()) == [0]
    assert [track.id for track, _ in grouped[0]] == ["1", "vehicle"]
