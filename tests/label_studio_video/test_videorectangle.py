import math

from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
    VideoRectangleSequenceItem,
    VideoRectangleValue,
)
from dagshub_annotation_converter.ir.video import CoordinateStyle, IRVideoAnnotationTrack, IRVideoBBoxFrameAnnotation


def _track_from_annotations(
    annotations: list[IRVideoBBoxFrameAnnotation],
    object_id: str | None = None,
) -> IRVideoAnnotationTrack:
    if not annotations:
        raise ValueError("Cannot create a track from empty annotations")
    resolved_object_id = object_id or annotations[0].imported_id or "0"
    return IRVideoAnnotationTrack.from_annotations(annotations, object_id=resolved_object_id)


def _single_track(annotation: VideoRectangleAnnotation) -> IRVideoAnnotationTrack:
    tracks = annotation.to_ir_annotation()
    assert len(tracks) == 1
    return tracks[0]


class TestVideoRectangleAnnotation:
    def test_parse_from_dict(self, sample_ls_video_task_data):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]

        ann = VideoRectangleAnnotation.model_validate(result)

        assert ann.id == "track_person_1"
        assert ann.type == "videorectangle"
        assert ann.original_width == 1920
        assert ann.original_height == 1080
        assert len(ann.value.sequence) == 5
        assert ann.value.labels == ["person"]

    def test_to_ir_annotation_returns_track(self, sample_ls_video_task_data):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)

        track = _single_track(ann)

        assert track.object_id == "track_person_1"
        assert len(track.annotations) == 5

        assert {a.imported_id for a in track.annotations} == {"track_person_1"}

        for ir_ann in track.annotations:
            assert "person" in ir_ann.categories
            assert ir_ann.keyframe
            assert ir_ann.visibility == 1.0

    def test_to_ir_annotation_enabled_controls_interpolation_state(self):
        ann = VideoRectangleAnnotation(
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(
                        frame=1,
                        x=10.0,
                        y=20.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                        visibility=0.75,
                    ),
                ],
                labels=["person"],
            ),
        )

        track = _single_track(ann)
        assert len(track.annotations) == 1
        assert track.annotations[0].keyframe is False
        assert "ls_enabled" not in track.annotations[0].meta
        assert "outside" not in track.annotations[0].meta
        assert track.annotations[0].visibility == 0.75

    def test_to_ir_annotation_coordinate_conversion(self, sample_ls_video_task_data, epsilon):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)

        track = _single_track(ann)

        first = track.annotations[0]

        assert first.coordinate_style == CoordinateStyle.NORMALIZED
        assert math.isclose(first.left, 0.05208333, abs_tol=epsilon)
        assert math.isclose(first.top, 0.13888889, abs_tol=epsilon)

    def test_to_ir_annotation_frame_numbers(self, sample_ls_video_task_data):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)

        track = _single_track(ann)

        frame_numbers = sorted([a.frame_number for a in track.annotations])
        assert frame_numbers == [0, 1, 2, 3, 4]

    def test_to_ir_annotation_timestamps(self, sample_ls_video_task_data, epsilon):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)

        track = _single_track(ann)

        first = [a for a in track.annotations if a.frame_number == 0][0]
        assert math.isclose(first.timestamp, 0.033, abs_tol=epsilon)

    def test_from_ir_annotation(self):
        ir_annotations = [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,  # 0-based (IR format)
                keyframe=False,
                left=0.05208333,
                top=0.13888889,
                width=0.02604167,
                height=0.11111111,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                timestamp=0.033,
            ),
            IRVideoBBoxFrameAnnotation(
                frame_number=1,  # 0-based (IR format)
                left=0.05729167,
                top=0.14074074,
                width=0.02604167,
                height=0.11111111,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                timestamp=0.067,
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]

        assert ls_ann.type == "videorectangle"
        assert ls_ann.original_width == 1920
        assert ls_ann.original_height == 1080
        assert len(ls_ann.value.sequence) == 2
        assert ls_ann.value.labels == ["person"]
        # Frame numbers should be 1-based in LS format
        assert ls_ann.value.sequence[0].frame == 1
        assert ls_ann.value.sequence[1].frame == 2
        # First has keyframe=False => no interpolation from this frame.
        # Second is keyframe=True and has no explicit stop boundary.
        assert not ls_ann.value.sequence[0].enabled
        assert ls_ann.value.sequence[1].enabled

    def test_from_ir_annotation_sparse_keyframes_keep_interpolation(self):
        ir_annotations = [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,
                keyframe=True,
                left=0.1,
                top=0.2,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                visibility=1.0,
            ),
            IRVideoBBoxFrameAnnotation(
                frame_number=9,
                keyframe=True,
                left=0.2,
                top=0.2,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                visibility=1.0,
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]
        assert ls_ann.value.sequence[0].enabled
        assert ls_ann.value.sequence[1].enabled

    def test_from_ir_annotation_outside_row_disables_previous_visible_keyframe(self):
        ir_annotations = [
            IRVideoBBoxFrameAnnotation(
                frame_number=2,
                keyframe=True,
                left=0.05,
                top=0.13,
                width=0.03,
                height=0.11,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                visibility=1.0,
            ),
            IRVideoBBoxFrameAnnotation(
                frame_number=3,
                keyframe=True,
                left=0.05,
                top=0.13,
                width=0.03,
                height=0.11,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                visibility=0.0,
                meta={},
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]
        assert len(ls_ann.value.sequence) == 1
        assert ls_ann.value.sequence[0].frame == 3
        assert not ls_ann.value.sequence[0].enabled

    def test_from_ir_annotation_coordinate_conversion(self, epsilon):
        ir_annotations = [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,  # 0-based (IR format)
                left=0.1,  # Should become 10%
                top=0.2,  # Should become 20%
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]

        seq_item = ls_ann.value.sequence[0]
        assert math.isclose(seq_item.x, 10.0, abs_tol=epsilon)
        assert math.isclose(seq_item.y, 20.0, abs_tol=epsilon)
        assert seq_item.frame == 1  # Should be 1-based in LS

    def test_from_ir_annotation_denormalized(self, epsilon):
        ir_annotations = [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,  # 0-based (IR format)
                left=192,  # 192/1920 = 0.1 = 10%
                top=216,  # 216/1080 = 0.2 = 20%
                width=96,
                height=108,
                video_width=1920,
                video_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]

        seq_item = ls_ann.value.sequence[0]
        assert math.isclose(seq_item.x, 10.0, abs_tol=epsilon)
        assert math.isclose(seq_item.y, 20.0, abs_tol=epsilon)
        assert seq_item.frame == 1  # Should be 1-based in LS

    def test_from_ir_annotation_uses_ls_standard_sequence_keys(self):
        ir_annotations = [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,
                keyframe=True,
                left=0.1,
                top=0.2,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                visibility=0.25,
                meta={},
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]
        seq_dump = ls_ann.model_dump()["value"]["sequence"][0]
        assert set(seq_dump.keys()) == {"frame", "x", "y", "width", "height", "enabled", "time", "rotation"}


class TestVideoRectangleFrameNumberConversion:
    def test_ls_to_ir_frame_conversion(self):
        ls_ann = VideoRectangleAnnotation(
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(frame=1, x=10.0, y=20.0, width=5.0, height=10.0),
                    VideoRectangleSequenceItem(frame=2, x=11.0, y=21.0, width=5.0, height=10.0),
                    VideoRectangleSequenceItem(frame=10, x=15.0, y=25.0, width=5.0, height=10.0),
                ],
                labels=["object"],
            ),
        )

        track = _single_track(ls_ann)

        frame_numbers = sorted([a.frame_number for a in track.annotations])
        assert frame_numbers == [0, 1, 9]  # LS 1,2,10 -> IR 0,1,9

    def test_ir_to_ls_frame_conversion(self):
        ir_annotations = [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,  # 0-based
                left=0.1,
                top=0.2,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxFrameAnnotation(
                frame_number=1,  # 0-based
                left=0.11,
                top=0.21,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxFrameAnnotation(
                frame_number=9,  # 0-based
                left=0.15,
                top=0.25,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]

        # LS should have 1-based frames
        ls_frames = [item.frame for item in ls_ann.value.sequence]
        assert ls_frames == [1, 2, 10]

    def test_cvat_to_ls_frame_conversion_roundtrip(self):
        original_ir = [
            IRVideoBBoxFrameAnnotation(
                frame_number=0,  # Frame 0 - this is the critical one!
                left=0.1,
                top=0.2,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxFrameAnnotation(
                frame_number=5,
                left=0.15,
                top=0.25,
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(original_ir))[0]

        assert ls_ann.value.sequence[0].frame == 1
        assert ls_ann.value.sequence[1].frame == 6

        recovered_ir = _single_track(ls_ann).annotations

        recovered_frames = sorted([a.frame_number for a in recovered_ir])
        assert recovered_frames == [0, 5]  # Frame 0 is preserved!


class TestVideoRectangleRoundtrip:
    def test_roundtrip_single_track(self, sample_ls_video_task_data, epsilon):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        original = VideoRectangleAnnotation.model_validate(result)

        ir_annotations = _single_track(original).annotations

        reconstructed = VideoRectangleAnnotation.from_ir_annotation(_track_from_annotations(ir_annotations))[0]

        assert len(reconstructed.value.sequence) == len(original.value.sequence)

        assert reconstructed.value.labels == original.value.labels

        for orig_item, recon_item in zip(original.value.sequence, reconstructed.value.sequence):
            assert orig_item.frame == recon_item.frame
            assert math.isclose(orig_item.x, recon_item.x, abs_tol=epsilon)
            assert math.isclose(orig_item.y, recon_item.y, abs_tol=epsilon)
            assert math.isclose(orig_item.width, recon_item.width, abs_tol=epsilon)
            assert math.isclose(orig_item.height, recon_item.height, abs_tol=epsilon)
