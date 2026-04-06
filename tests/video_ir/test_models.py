import math

import pytest

from dagshub_annotation_converter.ir.video import (
    CoordinateStyle,
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
    IRVideoSequence,
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


def test_normalized_copy_does_not_share_mutable_fields():
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
        meta={"source": {"name": "camera-a"}},
    )

    normalized = ann.normalized()
    normalized.categories["car"] = 0.5
    normalized.meta["source"]["name"] = "camera-b"

    assert ann.categories == {"person": 1.0}
    assert ann.meta == {"source": {"name": "camera-a"}}


def test_bbox_interpolate_between_frames(epsilon):
    start = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        keyframe=True,
        left=100,
        top=150,
        width=50,
        height=120,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        rotation=0.0,
        visibility=1.0,
        timestamp=0.0,
    )
    end = IRVideoBBoxFrameAnnotation(
        frame_number=4,
        keyframe=True,
        left=140,
        top=158,
        width=70,
        height=100,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        rotation=20.0,
        visibility=0.5,
        timestamp=0.4,
    )

    interpolated = start.interpolate(end, 0.5)

    assert interpolated.frame_number == 2
    assert interpolated.keyframe is False
    assert math.isclose(interpolated.left, 120, abs_tol=epsilon)
    assert math.isclose(interpolated.top, 154, abs_tol=epsilon)
    assert math.isclose(interpolated.width, 60, abs_tol=epsilon)
    assert math.isclose(interpolated.height, 110, abs_tol=epsilon)
    assert math.isclose(interpolated.rotation, 10.0, abs_tol=epsilon)
    assert math.isclose(interpolated.visibility, 0.75, abs_tol=epsilon)
    assert math.isclose(interpolated.timestamp, 0.2, abs_tol=epsilon)


def test_bbox_interpolate_rotation_uses_shortest_path(epsilon):
    start = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        keyframe=True,
        left=100,
        top=150,
        width=50,
        height=120,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        rotation=350.0,
    )
    end = IRVideoBBoxFrameAnnotation(
        frame_number=4,
        keyframe=True,
        left=140,
        top=158,
        width=70,
        height=100,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        rotation=10.0,
    )

    interpolated = start.interpolate(end, 0.5)

    assert math.isclose(interpolated.rotation % 360.0, 0.0, abs_tol=epsilon)


def test_track_normalize_coordinates_uses_shared_dimensions(epsilon):
    ann_a = IRVideoBBoxFrameAnnotation(
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
    ann_b = IRVideoBBoxFrameAnnotation(
        frame_number=1,
        left=384,
        top=216,
        width=192,
        height=108,
        video_width=None,
        video_height=None,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )

    track = IRVideoAnnotationTrack.from_annotations([ann_a, ann_b], object_id="1")
    normalized_track = track.normalized()

    first = normalized_track.annotations[0]
    second = normalized_track.annotations[1]

    assert first.coordinate_style == CoordinateStyle.NORMALIZED
    assert second.coordinate_style == CoordinateStyle.NORMALIZED
    assert second.video_width == 1920
    assert second.video_height == 1080
    assert math.isclose(second.left, 0.2, abs_tol=epsilon)
    assert math.isclose(second.top, 0.2, abs_tol=epsilon)
    assert math.isclose(second.width, 0.1, abs_tol=epsilon)
    assert math.isclose(second.height, 0.1, abs_tol=epsilon)


def test_track_normalized_does_not_mutate_source_track():
    ann_a = IRVideoBBoxFrameAnnotation(
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
    ann_b = IRVideoBBoxFrameAnnotation(
        frame_number=1,
        left=0.2,
        top=0.2,
        width=0.1,
        height=0.1,
        video_width=None,
        video_height=None,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.NORMALIZED,
        imported_id="original-id",
    )

    track = IRVideoAnnotationTrack.from_annotations([ann_a, ann_b], object_id="track-1")
    track.annotations[1].imported_id = "original-id"

    normalized_track = track.normalized()

    assert track.annotations[0].coordinate_style == CoordinateStyle.DENORMALIZED
    assert track.annotations[1].coordinate_style == CoordinateStyle.NORMALIZED
    assert track.annotations[1].video_width is None
    assert track.annotations[1].video_height is None
    assert track.annotations[1].imported_id == "original-id"

    assert normalized_track.annotations[0].coordinate_style == CoordinateStyle.NORMALIZED
    assert normalized_track.annotations[1].video_width == 1920
    assert normalized_track.annotations[1].video_height == 1080
    assert normalized_track.annotations[1].imported_id == "track-1"


def test_track_denormalize_coordinates_accepts_explicit_dimensions(epsilon):
    ann = IRVideoBBoxFrameAnnotation(
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

    track = IRVideoAnnotationTrack.from_annotations([ann], object_id="1")
    denormalized_track = track.denormalized(video_width=1920, video_height=1080)
    first = denormalized_track.annotations[0]

    assert first.coordinate_style == CoordinateStyle.DENORMALIZED
    assert first.video_width == 1920
    assert first.video_height == 1080
    assert math.isclose(first.left, 192, abs_tol=epsilon)
    assert math.isclose(first.top, 108, abs_tol=epsilon)
    assert math.isclose(first.width, 384, abs_tol=epsilon)
    assert math.isclose(first.height, 216, abs_tol=epsilon)


def test_track_denormalize_ignores_non_positive_explicit_dimensions(epsilon):
    ann = IRVideoBBoxFrameAnnotation(
        frame_number=0,
        left=0.1,
        top=0.1,
        width=0.2,
        height=0.2,
        video_width=1920,
        video_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    track = IRVideoAnnotationTrack.from_annotations([ann], object_id="1")
    denormalized_track = track.denormalized(video_width=0, video_height=0)
    first = denormalized_track.annotations[0]

    assert first.coordinate_style == CoordinateStyle.DENORMALIZED
    assert first.video_width == 1920
    assert first.video_height == 1080
    assert math.isclose(first.left, 192, abs_tol=epsilon)
    assert math.isclose(first.top, 108, abs_tol=epsilon)


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

    track = IRVideoAnnotationTrack.from_annotations([ann], object_id="7")

    assert track.object_id == "7"
    assert track.annotations[0].imported_id == "7"


def test_video_track_preserves_non_numeric_identifier():
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

    track = IRVideoAnnotationTrack.from_annotations([ann], object_id="track_person_1")
    materialized = track.to_annotations()

    assert track.object_id == "track_person_1"
    assert materialized[0].imported_id == "track_person_1"


def test_video_sequence_from_annotations_roundtrip():
    track_a = IRVideoAnnotationTrack.from_annotations(
        [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,
                left=10,
                top=20,
                width=30,
                height=40,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ],
        object_id="track_a",
    )
    track_b = IRVideoAnnotationTrack.from_annotations(
        [
            IRVideoBBoxFrameAnnotation(
                frame_number=4,
                left=50,
                top=60,
                width=70,
                height=80,
                video_width=1920,
                video_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ],
        object_id="track_b",
    )

    sequence = IRVideoSequence.from_annotations([track_a, track_b], filename="video.mp4")

    assert sequence.filename == "video.mp4"
    assert [track.object_id for track in sequence.tracks] == ["track_a", "track_b"]
    assert [ann.frame_number for ann in sequence.tracks[0].annotations] == [0]
    assert [ann.frame_number for ann in sequence.tracks[1].annotations] == [4]


def test_video_sequence_from_annotations_prefers_explicit_filename():
    track = IRVideoAnnotationTrack.from_annotations(
        [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,
                filename="source.mp4",
                left=10,
                top=20,
                width=30,
                height=40,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ],
        object_id="track_a",
    )

    sequence = IRVideoSequence.from_annotations([track], filename="override.mp4")

    assert sequence.filename == "override.mp4"


def test_video_sequence_from_annotations_requires_annotations():
    with pytest.raises(ValueError, match="Cannot create IRVideoSequence from empty tracks"):
        IRVideoSequence.from_annotations([])


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
    track = IRVideoAnnotationTrack.from_annotations([ann], object_id="2")
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
    assert flattened[0].sequence_length == 40


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
            IRVideoAnnotationTrack.from_annotations([ann_a], object_id="1"),
            IRVideoAnnotationTrack.from_annotations([ann_b], object_id="vehicle"),
        ]
    )

    grouped = sequence.annotations_by_frame()

    assert list(grouped.keys()) == [0]
    assert [track.object_id for track, _ in grouped[0]] == ["1", "vehicle"]
