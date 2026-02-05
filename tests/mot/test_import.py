"""Tests for MOT format import."""
import pytest
from pathlib import Path

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.mot import MOTContext
from dagshub_annotation_converter.formats.mot.bbox import import_bbox_from_line
from dagshub_annotation_converter.converters.mot import load_mot_from_file, load_mot_from_dir


class TestMOTLineImport:
    """Tests for parsing individual MOT lines.
    
    CVAT MOT 1.1 format (9 columns):
    frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    
    Example: 1,1,1363,569,103,241,1,1,0.86014
    """

    def test_parse_basic_line(self, mot_context):
        """Test parsing a basic CVAT MOT line (9 columns)."""
        # CVAT format: frame,track,x,y,w,h,not_ignored,class_id,visibility
        # MOT frame 1 becomes IR frame 0 (1-based to 0-based conversion)
        line = "1,1,100,150,50,120,1,1,1.0"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.frame_number == 0  # MOT frame 1 -> IR frame 0
        assert ann.track_id == 1
        assert ann.left == 100
        assert ann.top == 150
        assert ann.width == 50
        assert ann.height == 120
        assert ann.visibility == 1.0
        assert "person" in ann.categories  # class_id=1 maps to "person"
        assert ann.coordinate_style == CoordinateStyle.DENORMALIZED

    def test_parse_line_with_float_coords(self, mot_context):
        """Test parsing MOT line with floating point coordinates."""
        line = "1,3,794.27,247.59,71.245,174.88,1,1,0.86014"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.frame_number == 0  # MOT frame 1 -> IR frame 0
        assert ann.track_id == 3
        assert abs(ann.left - 794.27) < 0.01
        assert abs(ann.top - 247.59) < 0.01
        assert abs(ann.width - 71.245) < 0.01
        assert abs(ann.height - 174.88) < 0.01
        assert abs(ann.visibility - 0.86014) < 0.0001

    def test_parse_line_with_class_id(self, mot_context):
        """Test parsing MOT line with class_id mapping to category."""
        line = "1,2,500,200,150,100,1,2,1.0"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.track_id == 2
        assert "car" in ann.categories  # class_id=2 maps to "car"

    def test_parse_line_ignored_entry(self, mot_context):
        """Test parsing MOT line with not_ignored=0 (should be ignored)."""
        # not_ignored=0 means the entry should be ignored in evaluation
        line = "1,1,100,150,50,120,0,1,1.0"
        
        ann = import_bbox_from_line(line, mot_context)
        
        # Should still parse, but mark as ignored in metadata
        assert ann.meta.get("ignored") == True

    def test_parse_line_with_partial_visibility(self, mot_context):
        """Test parsing MOT line with partial occlusion."""
        line = "3,1,120,154,50,120,1,1,0.5"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.frame_number == 2  # MOT frame 3 -> IR frame 2
        assert ann.visibility == 0.5


class TestMOTFileImport:
    """Tests for importing full MOT files."""

    def test_load_from_file(self, sample_mot_file, mot_context):
        """Test loading annotations from a MOT file."""
        annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        # Should have annotations grouped by frame (0-based)
        assert len(annotations) > 0
        
        # IR Frame 0 (MOT Frame 1) should have 2 annotations (track 1 and track 2)
        frame_0_anns = annotations.get(0, [])
        assert len(frame_0_anns) == 2
        
        # Verify track IDs
        track_ids = {ann.track_id for ann in frame_0_anns}
        assert track_ids == {1, 2}

    def test_load_from_file_track_consistency(self, sample_mot_file, mot_context):
        """Test that track IDs are consistent across frames."""
        annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        # Track 1 should appear in all 5 frames (IR frames 0-4)
        track_1_frames = []
        for frame_num, anns in annotations.items():
            for ann in anns:
                if ann.track_id == 1:
                    track_1_frames.append(frame_num)
        
        assert len(track_1_frames) == 5
        assert sorted(track_1_frames) == [0, 1, 2, 3, 4]  # 0-based IR frames

    def test_load_skips_comments(self, sample_mot_file, mot_context):
        """Test that comment lines are skipped."""
        annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        # Total should be 10 annotations (5 frames x 2 tracks)
        total_anns = sum(len(anns) for anns in annotations.values())
        assert total_anns == 10


class TestMOTDirectoryImport:
    """Tests for importing MOT from directory with seqinfo.ini."""

    def test_load_from_dir(self, sample_mot_dir):
        """Test loading from directory reads seqinfo.ini and labels.txt."""
        annotations, context = load_mot_from_dir(sample_mot_dir)
        
        # Context should be populated from seqinfo.ini
        assert context.frame_rate == 30.0
        assert context.image_width == 1920
        assert context.image_height == 1080
        assert context.seq_name == "test_sequence"
        
        # Categories should be loaded from gt/labels.txt
        assert context.categories == {1: "person", 2: "car"}


class TestMOTFrameNumberConversion:
    """Tests for frame number conversion between MOT (1-based) and IR (0-based)."""

    def test_mot_import_converts_frame_to_0_based(self, mot_context):
        """Test that MOT 1-based frames are converted to 0-based IR frames on import."""
        from dagshub_annotation_converter.formats.mot.bbox import import_bbox_from_line
        
        # MOT frame 1 should become IR frame 0
        line1 = "1,1,100,150,50,120,1,1,1.0"
        ann1 = import_bbox_from_line(line1, mot_context)
        assert ann1.frame_number == 0
        
        # MOT frame 10 should become IR frame 9
        line10 = "10,1,100,150,50,120,1,1,1.0"
        ann10 = import_bbox_from_line(line10, mot_context)
        assert ann10.frame_number == 9

    def test_mot_export_converts_frame_to_1_based(self, mot_context):
        """Test that 0-based IR frames are converted to 1-based MOT frames on export."""
        from dagshub_annotation_converter.formats.mot.bbox import export_bbox_to_line
        
        # Create IR annotation with frame 0
        ann0 = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=100, top=150, width=50, height=120,
            image_width=1920, image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=1.0,
        )
        line0 = export_bbox_to_line(ann0, mot_context)
        assert line0.startswith("1,")  # MOT frame should be 1
        
        # Create IR annotation with frame 9
        ann9 = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=9,
            left=100, top=150, width=50, height=120,
            image_width=1920, image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=1.0,
        )
        line9 = export_bbox_to_line(ann9, mot_context)
        assert line9.startswith("10,")  # MOT frame should be 10

    def test_mot_roundtrip_preserves_frame_numbers(self, mot_context):
        """Test that frame numbers are preserved through MOT -> IR -> MOT roundtrip."""
        from dagshub_annotation_converter.formats.mot.bbox import import_bbox_from_line, export_bbox_to_line
        
        # Start with MOT frame 1
        original_line = "1,1,100,150,50,120,1,1,1.0"
        
        # Import (MOT 1 -> IR 0)
        ann = import_bbox_from_line(original_line, mot_context)
        assert ann.frame_number == 0
        
        # Export (IR 0 -> MOT 1)
        exported_line = export_bbox_to_line(ann, mot_context)
        
        # Frame number should be back to 1
        parts = exported_line.split(",")
        assert parts[0] == "1"
