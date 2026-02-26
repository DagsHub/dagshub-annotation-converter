import math

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
    VideoRectangleSequenceItem,
)


class TestVideoRectangleSequenceItem:
    def test_create_sequence_item(self):
        item = VideoRectangleSequenceItem(
            frame=1,
            x=5.208333,
            y=13.888889,
            width=2.604167,
            height=11.111111,
            enabled=True,
            time=0.033,
        )
        
        assert item.frame == 1
        assert item.x == 5.208333
        assert item.y == 13.888889
        assert item.width == 2.604167
        assert item.height == 11.111111
        assert item.enabled
        assert item.time == 0.033


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

    def test_to_ir_annotations(self, sample_ls_video_task_data):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        assert len(ir_annotations) == 5
        
        # All should have same track_id (derived from annotation id)
        track_ids = {a.track_id for a in ir_annotations}
        assert len(track_ids) == 1
        
        # All should have "person" category
        for ir_ann in ir_annotations:
            assert "person" in ir_ann.categories
            assert ir_ann.keyframe
            assert ir_ann.visibility == 1.0

    def test_to_ir_annotations_enabled_controls_interpolation_state(self):
        from dagshub_annotation_converter.formats.label_studio.videorectangle import (
            VideoRectangleValue,
            VideoRectangleSequenceItem,
        )

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
            meta={"original_track_id": 1},
        )

        ir_annotations = ann.to_ir_annotations()
        assert len(ir_annotations) == 1
        assert ir_annotations[0].keyframe
        assert ir_annotations[0].meta.get("ls_enabled") is False
        assert not ir_annotations[0].meta.get("outside")
        assert ir_annotations[0].visibility == 0.75

    def test_to_ir_annotations_disabled_without_visibility_stays_visible(self):
        from dagshub_annotation_converter.formats.label_studio.videorectangle import (
            VideoRectangleValue,
            VideoRectangleSequenceItem,
        )

        ann = VideoRectangleAnnotation(
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(
                        frame=11,
                        x=10.0,
                        y=20.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                    ),
                ],
                labels=["person"],
            ),
            meta={"original_track_id": 1},
        )

        ir_annotations = ann.to_ir_annotations()
        assert len(ir_annotations) == 1
        assert ir_annotations[0].meta.get("ls_enabled") is False
        assert not ir_annotations[0].meta.get("outside")
        assert ir_annotations[0].visibility == 1.0

    def test_to_ir_annotations_preserves_outside_flag(self):
        from dagshub_annotation_converter.formats.label_studio.videorectangle import (
            VideoRectangleValue,
            VideoRectangleSequenceItem,
        )

        ann = VideoRectangleAnnotation(
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(
                        frame=4,
                        x=10.0,
                        y=20.0,
                        width=5.0,
                        height=10.0,
                        enabled=True,
                        outside=True,
                        visibility=0.0,
                    ),
                ],
                labels=["person"],
            ),
            meta={"original_track_id": 1},
        )

        ir_annotations = ann.to_ir_annotations()
        assert len(ir_annotations) == 1
        assert ir_annotations[0].meta.get("outside") is True

    def test_to_ir_annotations_parses_string_outside_false_as_visible(self):
        from dagshub_annotation_converter.formats.label_studio.videorectangle import (
            VideoRectangleValue,
            VideoRectangleSequenceItem,
        )

        ann = VideoRectangleAnnotation(
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(
                        frame=9,
                        x=10.0,
                        y=20.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                        outside="false",
                        visibility="1",
                    ),
                ],
                labels=["person"],
            ),
            meta={"original_track_id": 1},
        )

        ir_annotations = ann.to_ir_annotations()
        assert len(ir_annotations) == 1
        assert ir_annotations[0].meta.get("outside") is False
        assert ir_annotations[0].visibility == 1.0

    def test_to_ir_annotations_coordinate_conversion(self, sample_ls_video_task_data, epsilon):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        # First annotation: x=5.208333%, y=13.888889%
        # Should be normalized to 0.05208333, 0.13888889
        first = ir_annotations[0]
        
        assert first.coordinate_style == CoordinateStyle.NORMALIZED
        assert math.isclose(first.left, 0.05208333, abs_tol=epsilon)
        assert math.isclose(first.top, 0.13888889, abs_tol=epsilon)

    def test_to_ir_annotations_frame_numbers(self, sample_ls_video_task_data):
        """Test that frame numbers are correctly converted from 1-based (LS) to 0-based (IR)."""
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        frame_numbers = sorted([a.frame_number for a in ir_annotations])
        assert frame_numbers == [0, 1, 2, 3, 4]

    def test_to_ir_annotations_timestamps(self, sample_ls_video_task_data, epsilon):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        # First frame (LS frame 1 -> IR frame 0) should have timestamp 0.033
        first = [a for a in ir_annotations if a.frame_number == 0][0]
        assert math.isclose(first.timestamp, 0.033, abs_tol=epsilon)

    def test_from_ir_annotations(self):
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
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
            IRVideoBBoxAnnotation(
                track_id=1,
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
        
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        assert ls_ann.type == "videorectangle"
        assert ls_ann.original_width == 1920
        assert ls_ann.original_height == 1080
        assert len(ls_ann.value.sequence) == 2
        assert ls_ann.value.labels == ["person"]
        # Frame numbers should be 1-based in LS format
        assert ls_ann.value.sequence[0].frame == 1
        assert ls_ann.value.sequence[1].frame == 2
        assert ls_ann.value.sequence[0].enabled
        assert not ls_ann.value.sequence[1].enabled

    def test_from_ir_annotations_sparse_keyframes_keep_interpolation(self):
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
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
            IRVideoBBoxAnnotation(
                track_id=1,
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

        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        assert ls_ann.value.sequence[0].enabled
        assert not ls_ann.value.sequence[1].enabled

    def test_from_ir_annotations_outside_row_disables_previous_visible_keyframe(self):
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
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
            IRVideoBBoxAnnotation(
                track_id=1,
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
                meta={"outside": True},
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        assert len(ls_ann.value.sequence) == 2
        assert ls_ann.value.sequence[0].frame == 3
        assert not ls_ann.value.sequence[0].enabled
        assert ls_ann.value.sequence[1].frame == 4
        assert not ls_ann.value.sequence[1].enabled

    def test_from_ir_annotations_coordinate_conversion(self, epsilon):
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based (IR format)
                left=0.1,  # Should become 10%
                top=0.2,   # Should become 20%
                width=0.05,
                height=0.1,
                video_width=1920,
                video_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]
        
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        seq_item = ls_ann.value.sequence[0]
        assert math.isclose(seq_item.x, 10.0, abs_tol=epsilon)
        assert math.isclose(seq_item.y, 20.0, abs_tol=epsilon)
        assert seq_item.frame == 1  # Should be 1-based in LS

    def test_from_ir_annotations_denormalized(self, epsilon):
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based (IR format)
                left=192,   # 192/1920 = 0.1 = 10%
                top=216,    # 216/1080 = 0.2 = 20%
                width=96,
                height=108,
                video_width=1920,
                video_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
        ]
        
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        seq_item = ls_ann.value.sequence[0]
        assert math.isclose(seq_item.x, 10.0, abs_tol=epsilon)
        assert math.isclose(seq_item.y, 20.0, abs_tol=epsilon)
        assert seq_item.frame == 1  # Should be 1-based in LS

    def test_from_ir_annotations_uses_ls_standard_sequence_keys(self):
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
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
                meta={"outside": False},
            ),
        ]

        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        seq_dump = ls_ann.model_dump()["value"]["sequence"][0]
        assert set(seq_dump.keys()) == {"frame", "x", "y", "width", "height", "enabled", "time", "rotation"}


class TestVideoRectangleFrameNumberConversion:
    def test_ls_to_ir_frame_conversion(self):
        from dagshub_annotation_converter.formats.label_studio.videorectangle import (
            VideoRectangleAnnotation,
            VideoRectangleValue,
            VideoRectangleSequenceItem,
        )
        
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
            meta={"original_track_id": 1},
        )
        
        ir_annotations = ls_ann.to_ir_annotations()
        
        frame_numbers = sorted([a.frame_number for a in ir_annotations])
        assert frame_numbers == [0, 1, 9]  # LS 1,2,10 -> IR 0,1,9

    def test_ls_to_ir_frame_conversion_supports_zero_based_sequences(self):
        from dagshub_annotation_converter.formats.label_studio.videorectangle import (
            VideoRectangleAnnotation,
            VideoRectangleValue,
            VideoRectangleSequenceItem,
        )

        ls_ann = VideoRectangleAnnotation(
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(frame=0, x=10.0, y=20.0, width=5.0, height=10.0),
                    VideoRectangleSequenceItem(frame=1, x=11.0, y=21.0, width=5.0, height=10.0),
                    VideoRectangleSequenceItem(frame=9, x=15.0, y=25.0, width=5.0, height=10.0),
                ],
                labels=["object"],
            ),
            meta={"original_track_id": 1},
        )

        ir_annotations = ls_ann.to_ir_annotations()

        frame_numbers = sorted([a.frame_number for a in ir_annotations])
        assert frame_numbers == [0, 1, 9]

    def test_ir_to_ls_frame_conversion(self):
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based
                left=0.1, top=0.2, width=0.05, height=0.1,
                video_width=1920, video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=1,  # 0-based
                left=0.11, top=0.21, width=0.05, height=0.1,
                video_width=1920, video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=9,  # 0-based
                left=0.15, top=0.25, width=0.05, height=0.1,
                video_width=1920, video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]
        
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        # LS should have 1-based frames
        ls_frames = [item.frame for item in ls_ann.value.sequence]
        assert ls_frames == [1, 2, 10]

    def test_cvat_to_ls_frame_conversion_roundtrip(self):
        original_ir = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # Frame 0 - this is the critical one!
                left=0.1, top=0.2, width=0.05, height=0.1,
                video_width=1920, video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=5,
                left=0.15, top=0.25, width=0.05, height=0.1,
                video_width=1920, video_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]
        

        ls_ann = VideoRectangleAnnotation.from_ir_annotations(original_ir)
        
        assert ls_ann.value.sequence[0].frame == 1
        assert ls_ann.value.sequence[1].frame == 6
        
        recovered_ir = ls_ann.to_ir_annotations()
        
        recovered_frames = sorted([a.frame_number for a in recovered_ir])
        assert recovered_frames == [0, 5]  # Frame 0 is preserved!


class TestVideoRectangleRoundtrip:
    def test_roundtrip_single_track(self, sample_ls_video_task_data, epsilon):
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        original = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = original.to_ir_annotations()
        
        reconstructed = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        assert len(reconstructed.value.sequence) == len(original.value.sequence)
        
        assert reconstructed.value.labels == original.value.labels
        
        for orig_item, recon_item in zip(original.value.sequence, reconstructed.value.sequence):
            assert orig_item.frame == recon_item.frame
            assert math.isclose(orig_item.x, recon_item.x, abs_tol=epsilon)
            assert math.isclose(orig_item.y, recon_item.y, abs_tol=epsilon)
            assert math.isclose(orig_item.width, recon_item.width, abs_tol=epsilon)
            assert math.isclose(orig_item.height, recon_item.height, abs_tol=epsilon)
