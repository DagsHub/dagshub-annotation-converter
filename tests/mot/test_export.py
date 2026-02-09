from pathlib import Path
import tempfile

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.mot.bbox import export_bbox_to_line
from dagshub_annotation_converter.converters.mot import export_to_mot


class TestMOTLineExport:
    def test_export_basic_annotation(self, mot_context):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
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
        
        # CVAT MOT 1.1: frame,track,x,y,w,h,not_ignored,class_id,visibility
        parts = line.split(",")
        assert len(parts) == 9
        assert parts[0] == "1"
        assert parts[1] == "1"
        assert parts[2] == "100"
        assert parts[3] == "150"
        assert parts[4] == "50"
        assert parts[5] == "120"
        assert parts[6] == "1"
        assert parts[7] == "1"
        assert float(parts[8]) == 1.0

    def test_export_normalized_annotation(self, mot_context):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=0.1,
            top=0.1,
            width=0.2,
            height=0.2,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.NORMALIZED,
            visibility=1.0,
        )
        
        line = export_bbox_to_line(ann, mot_context)
        parts = line.split(",")
        
        assert abs(float(parts[2]) - 192) < 1
        assert abs(float(parts[3]) - 108) < 1

    def test_export_with_partial_visibility(self, mot_context):
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
        
        assert parts[7] == "2"
        assert float(parts[8]) == 0.5

    def test_export_ignored_annotation(self, mot_context):
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
        
        assert parts[6] == "0"


class TestMOTFileExport:
    def test_export_to_file(self, mot_context):
        annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,
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
                frame_number=0,
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
                frame_number=1,
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
            
            content = output_path.read_text()
            lines = [line for line in content.strip().split("\n") if line and not line.startswith("#")]
            
            assert len(lines) == 3
            
            first_line_parts = lines[0].split(",")
            assert first_line_parts[0] == "1"
            
        finally:
            output_path.unlink()

    def test_export_roundtrip(self, mot_context, sample_mot_file):
        from dagshub_annotation_converter.converters.mot import load_mot_from_file
        
        annotations_by_frame = load_mot_from_file(sample_mot_file, mot_context)
        
        all_annotations = []
        for frame_anns in annotations_by_frame.values():
            all_annotations.extend(frame_anns)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(all_annotations, mot_context, output_path)
            
            reimported = load_mot_from_file(output_path, mot_context)
            
            assert len(reimported) == len(annotations_by_frame)
            
            for frame_num in annotations_by_frame:
                assert len(reimported[frame_num]) == len(annotations_by_frame[frame_num])
                
        finally:
            output_path.unlink()
