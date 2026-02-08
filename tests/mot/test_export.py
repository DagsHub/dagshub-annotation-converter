"""Tests for MOT format export."""
from pathlib import Path
import tempfile

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.mot.bbox import export_bbox_to_line
from dagshub_annotation_converter.converters.mot import export_to_mot


class TestMOTLineExport:
    """Tests for exporting individual annotations to MOT lines.
    
    CVAT MOT 1.1 format (9 columns):
    frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    """

    def test_export_basic_annotation(self, mot_context):
        """Test exporting a basic annotation to CVAT MOT line (9 columns)."""
        # IR uses 0-based frames, MOT uses 1-based frames
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,  # 0-based IR frame
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=1.0,
        )
        
        line = export_bbox_to_line(ann, mot_context)
        
        # Expected format: frame,track,x,y,w,h,not_ignored,class_id,visibility
        parts = line.split(",")
        assert len(parts) == 9  # CVAT MOT format has 9 columns
        assert parts[0] == "1"  # MOT frame (IR 0 -> MOT 1)
        assert parts[1] == "1"  # track_id
        assert parts[2] == "100"  # x (left)
        assert parts[3] == "150"  # y (top)
        assert parts[4] == "50"  # width
        assert parts[5] == "120"  # height
        assert parts[6] == "1"  # not_ignored (1=active)
        assert parts[7] == "1"  # class_id (person=1)
        assert float(parts[8]) == 1.0  # visibility

    def test_export_normalized_annotation(self, mot_context):
        """Test exporting normalized coordinates (should denormalize)."""
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,  # 0-based IR frame
            left=0.1,  # 192 pixels
            top=0.1,   # 108 pixels
            width=0.2,  # 384 pixels
            height=0.2,  # 216 pixels
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.NORMALIZED,
            visibility=1.0,
        )
        
        line = export_bbox_to_line(ann, mot_context)
        parts = line.split(",")
        
        # Should be denormalized to pixel values
        assert abs(float(parts[2]) - 192) < 1  # x (left)
        assert abs(float(parts[3]) - 108) < 1  # y (top)

    def test_export_with_partial_visibility(self, mot_context):
        """Test exporting annotation with partial occlusion."""
        ann = IRVideoBBoxAnnotation(
            track_id=2,
            frame_number=3,
            left=120,
            top=154,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"car": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=0.5,
        )
        
        line = export_bbox_to_line(ann, mot_context)
        parts = line.split(",")
        
        assert parts[7] == "2"  # class_id (car=2)
        assert float(parts[8]) == 0.5  # visibility

    def test_export_ignored_annotation(self, mot_context):
        """Test exporting annotation marked as ignored."""
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=1,
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=1.0,
            meta={"ignored": True},
        )
        
        line = export_bbox_to_line(ann, mot_context)
        parts = line.split(",")
        
        assert parts[6] == "0"  # not_ignored=0 means ignored


class TestMOTFileExport:
    """Tests for exporting to full MOT files."""

    def test_export_to_file(self, mot_context):
        """Test exporting multiple annotations to a MOT file."""
        # IR uses 0-based frames, MOT uses 1-based frames
        annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,  # 0-based IR frame
                left=100,
                top=150,
                width=50,
                height=120,
                image_width=1920,
                image_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=2,
                frame_number=0,  # 0-based IR frame
                left=500,
                top=200,
                width=150,
                height=100,
                image_width=1920,
                image_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=1,  # 0-based IR frame
                left=110,
                top=152,
                width=50,
                height=120,
                image_width=1920,
                image_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(annotations, mot_context, output_path)
            
            # Read and verify output
            content = output_path.read_text()
            lines = [line for line in content.strip().split("\n") if line and not line.startswith("#")]
            
            assert len(lines) == 3
            
            # Lines should be sorted by MOT frame (1-based), then by track_id
            first_line_parts = lines[0].split(",")
            assert first_line_parts[0] == "1"  # MOT frame 1 (from IR frame 0)
            
        finally:
            output_path.unlink()

    def test_export_roundtrip(self, mot_context, sample_mot_file):
        """Test that import -> export produces equivalent output."""
        from dagshub_annotation_converter.converters.mot import load_mot_from_file
        
        # Import
        annotations_by_frame = load_mot_from_file(sample_mot_file, mot_context)
        
        # Flatten annotations
        all_annotations = []
        for frame_anns in annotations_by_frame.values():
            all_annotations.extend(frame_anns)
        
        # Export to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(all_annotations, mot_context, output_path)
            
            # Re-import
            reimported = load_mot_from_file(output_path, mot_context)
            
            # Same number of frames
            assert len(reimported) == len(annotations_by_frame)
            
            # Same number of annotations per frame
            for frame_num in annotations_by_frame:
                assert len(reimported[frame_num]) == len(annotations_by_frame[frame_num])
                
        finally:
            output_path.unlink()
