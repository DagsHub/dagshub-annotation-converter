"""Tests for Label Studio Video rectangle format."""

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
    VideoRectangleSequenceItem,
)


class TestVideoRectangleSequenceItem:
    """Tests for individual sequence items."""

    def test_create_sequence_item(self):
        """Test creating a sequence item."""
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
    """Tests for VideoRectangleAnnotation parsing and conversion."""

    def test_parse_from_dict(self, sample_ls_video_task_data):
        """Test parsing VideoRectangleAnnotation from JSON dict."""
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        
        ann = VideoRectangleAnnotation.model_validate(result)
        
        assert ann.id == "track_person_1"
        assert ann.type == "videorectangle"
        assert ann.original_width == 1920
        assert ann.original_height == 1080
        assert len(ann.value.sequence) == 5
        assert ann.value.labels == ["person"]

    def test_to_ir_annotations(self, sample_ls_video_task_data):
        """Test converting to IR video annotations."""
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        # Should produce one annotation per frame
        assert len(ir_annotations) == 5
        
        # All should have same track_id (derived from annotation id)
        track_ids = {a.track_id for a in ir_annotations}
        assert len(track_ids) == 1
        
        # All should have "person" category
        for ir_ann in ir_annotations:
            assert "person" in ir_ann.categories

    def test_to_ir_annotations_coordinate_conversion(self, sample_ls_video_task_data):
        """Test that LS percentage coords are converted to normalized."""
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        # First annotation: x=5.208333%, y=13.888889%
        # Should be normalized to 0.05208333, 0.13888889
        first = ir_annotations[0]
        
        assert first.coordinate_style == CoordinateStyle.NORMALIZED
        assert abs(first.left - 0.05208333) < 1e-6
        assert abs(first.top - 0.13888889) < 1e-6

    def test_to_ir_annotations_frame_numbers(self, sample_ls_video_task_data):
        """Test that frame numbers are correctly converted from 1-based (LS) to 0-based (IR)."""
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        # Label Studio uses 1-based frames (1,2,3,4,5), IR should be 0-based (0,1,2,3,4)
        frame_numbers = sorted([a.frame_number for a in ir_annotations])
        assert frame_numbers == [0, 1, 2, 3, 4]

    def test_to_ir_annotations_timestamps(self, sample_ls_video_task_data):
        """Test that timestamps are preserved."""
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        ann = VideoRectangleAnnotation.model_validate(result)
        
        ir_annotations = ann.to_ir_annotations()
        
        # First frame (LS frame 1 -> IR frame 0) should have timestamp 0.033
        first = [a for a in ir_annotations if a.frame_number == 0][0]
        assert abs(first.timestamp - 0.033) < 1e-6

    def test_from_ir_annotations(self):
        """Test creating VideoRectangleAnnotation from IR annotations."""
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based (IR format)
                left=0.05208333,
                top=0.13888889,
                width=0.02604167,
                height=0.11111111,
                image_width=1920,
                image_height=1080,
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
                image_width=1920,
                image_height=1080,
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

    def test_from_ir_annotations_coordinate_conversion(self):
        """Test that normalized coords are converted to LS percentage."""
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based (IR format)
                left=0.1,  # Should become 10%
                top=0.2,   # Should become 20%
                width=0.05,
                height=0.1,
                image_width=1920,
                image_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]
        
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        seq_item = ls_ann.value.sequence[0]
        assert abs(seq_item.x - 10.0) < 1e-6
        assert abs(seq_item.y - 20.0) < 1e-6
        assert seq_item.frame == 1  # Should be 1-based in LS

    def test_from_ir_annotations_denormalized(self):
        """Test conversion from denormalized IR coordinates."""
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based (IR format)
                left=192,   # 192/1920 = 0.1 = 10%
                top=216,    # 216/1080 = 0.2 = 20%
                width=96,
                height=108,
                image_width=1920,
                image_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
        ]
        
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        seq_item = ls_ann.value.sequence[0]
        assert abs(seq_item.x - 10.0) < 1e-6
        assert abs(seq_item.y - 20.0) < 1e-6
        assert seq_item.frame == 1  # Should be 1-based in LS


class TestVideoRectangleFrameNumberConversion:
    """Tests for frame number conversion between Label Studio (1-based) and IR (0-based)."""

    def test_ls_to_ir_frame_conversion(self):
        """Test that Label Studio 1-based frames are converted to 0-based IR frames."""
        from dagshub_annotation_converter.formats.label_studio.videorectangle import (
            VideoRectangleAnnotation,
            VideoRectangleValue,
            VideoRectangleSequenceItem,
        )
        
        # Create LS annotation with 1-based frames
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
        
        # IR should have 0-based frames
        frame_numbers = sorted([a.frame_number for a in ir_annotations])
        assert frame_numbers == [0, 1, 9]  # LS 1,2,10 -> IR 0,1,9

    def test_ir_to_ls_frame_conversion(self):
        """Test that 0-based IR frames are converted to 1-based Label Studio frames."""
        # Create IR annotations with 0-based frames
        ir_annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based
                left=0.1, top=0.2, width=0.05, height=0.1,
                image_width=1920, image_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=1,  # 0-based
                left=0.11, top=0.21, width=0.05, height=0.1,
                image_width=1920, image_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=9,  # 0-based
                left=0.15, top=0.25, width=0.05, height=0.1,
                image_width=1920, image_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]
        
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        # LS should have 1-based frames
        ls_frames = [item.frame for item in ls_ann.value.sequence]
        assert ls_frames == [1, 2, 10]  # IR 0,1,9 -> LS 1,2,10

    def test_cvat_to_ls_frame_conversion_roundtrip(self):
        """Test CVAT (0-based) -> IR -> LS (1-based) -> IR -> CVAT (0-based) preserves frame 0."""
        # Start with 0-based IR (simulating CVAT import)
        original_ir = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # Frame 0 - this is the critical one!
                left=0.1, top=0.2, width=0.05, height=0.1,
                image_width=1920, image_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=5,
                left=0.15, top=0.25, width=0.05, height=0.1,
                image_width=1920, image_height=1080,
                categories={"object": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]
        
        # Convert to Label Studio
        ls_ann = VideoRectangleAnnotation.from_ir_annotations(original_ir)
        
        # LS frames should be 1-based
        assert ls_ann.value.sequence[0].frame == 1
        assert ls_ann.value.sequence[1].frame == 6
        
        # Convert back to IR
        recovered_ir = ls_ann.to_ir_annotations()
        
        # IR frames should be 0-based again
        recovered_frames = sorted([a.frame_number for a in recovered_ir])
        assert recovered_frames == [0, 5]  # Frame 0 is preserved!


class TestVideoRectangleRoundtrip:
    """Tests for roundtrip conversion LS -> IR -> LS."""

    def test_roundtrip_single_track(self, sample_ls_video_task_data):
        """Test roundtrip conversion preserves data."""
        result = sample_ls_video_task_data["annotations"][0]["result"][0]
        original = VideoRectangleAnnotation.model_validate(result)
        
        # Convert to IR
        ir_annotations = original.to_ir_annotations()
        
        # Convert back to LS
        reconstructed = VideoRectangleAnnotation.from_ir_annotations(ir_annotations)
        
        # Same number of sequence items
        assert len(reconstructed.value.sequence) == len(original.value.sequence)
        
        # Same labels
        assert reconstructed.value.labels == original.value.labels
        
        # Coordinates and frame numbers should match within tolerance
        for orig_item, recon_item in zip(original.value.sequence, reconstructed.value.sequence):
            assert orig_item.frame == recon_item.frame  # Frame numbers preserved in roundtrip
            assert abs(orig_item.x - recon_item.x) < 0.001
            assert abs(orig_item.y - recon_item.y) < 0.001
            assert abs(orig_item.width - recon_item.width) < 0.001
            assert abs(orig_item.height - recon_item.height) < 0.001
