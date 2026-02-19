from pathlib import Path
import tempfile
import math
import configparser
from zipfile import ZipFile
import pytest

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.mot.bbox import export_bbox_to_line
from dagshub_annotation_converter.converters.mot import export_to_mot, export_mot_to_dir, export_mot_sequences_to_dirs
from dagshub_annotation_converter.formats.mot.context import MOTContext


class TestMOTLineExport:
    def test_export_basic_annotation(self, mot_context):
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
        
        assert math.isclose(float(parts[2]), 192, abs_tol=1)
        assert math.isclose(float(parts[3]), 108, abs_tol=1)

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

    def test_export_to_dir_uses_probed_dimensions_for_seqinfo(self, tmp_path, monkeypatch):
        context = MOTContext(frame_rate=30.0, image_width=None, image_height=None, seq_name="test_sequence")
        context.categories = {1: "person"}
        annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,
                left=100,
                top=150,
                width=50,
                height=120,
                image_width=0,
                image_height=0,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ]

        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.mot.get_video_dimensions",
            lambda _: (1280, 720, 25.0),
        )

        out_dir = tmp_path / "mot_out"
        export_mot_to_dir(annotations, context, out_dir, video_file="local_video.mp4")

        seqinfo = configparser.ConfigParser()
        seqinfo.read(out_dir / "seqinfo.ini")
        seq = seqinfo["Sequence"]
        assert seq["imWidth"] == "1280"
        assert seq["imHeight"] == "720"
        assert seq["frameRate"] == "25.0"

    def test_export_raises_without_dimensions_or_video_file(self, tmp_path):
        context = MOTContext(frame_rate=30.0, image_width=None, image_height=None)
        context.categories = {1: "person"}
        annotations = [
            IRVideoBBoxAnnotation(
                track_id=1,
                frame_number=0,
                left=100,
                top=150,
                width=50,
                height=120,
                image_width=0,
                image_height=0,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ]

        output_path = tmp_path / "gt.txt"
        with pytest.raises(ValueError, match="Cannot determine frame dimensions for MOT export"):
            export_to_mot(annotations, context, output_path)

    def test_export_sequences_to_dirs_groups_by_filename(self, tmp_path):
        context = MOTContext(frame_rate=30.0, image_width=1920, image_height=1080, seq_name="default")
        context.categories = {1: "person"}
        ann_a = IRVideoBBoxAnnotation(
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
            filename="earth-space-small.mp4",
        )
        ann_b = IRVideoBBoxAnnotation(
            track_id=2,
            frame_number=0,
            left=500,
            top=200,
            width=150,
            height=100,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="jelly.mp4",
        )

        outputs = export_mot_sequences_to_dirs([ann_a, ann_b], context, tmp_path)
        assert set(outputs.keys()) == {"earth-space-small.mp4", "jelly.mp4"}
        assert (tmp_path / "earth-space-small.mp4.zip").exists()
        assert (tmp_path / "jelly.mp4.zip").exists()
        with ZipFile(tmp_path / "earth-space-small.mp4.zip") as z:
            assert "gt/gt.txt" in z.namelist()
            assert "gt/labels.txt" in z.namelist()
            assert "seqinfo.ini" in z.namelist()
        with ZipFile(tmp_path / "jelly.mp4.zip") as z:
            assert "gt/gt.txt" in z.namelist()
            assert "gt/labels.txt" in z.namelist()
            assert "seqinfo.ini" in z.namelist()

    def test_export_sequences_to_dirs_accepts_stem_video_file_keys(self, tmp_path, monkeypatch):
        context = MOTContext(frame_rate=30.0, image_width=None, image_height=None, seq_name="default")
        context.categories = {1: "person"}
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=None,
            image_height=None,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="earth-space-small.mp4",
        )

        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.mot.get_video_dimensions",
            lambda _: (1280, 720, 25.0),
        )

        outputs = export_mot_sequences_to_dirs(
            [ann],
            context,
            tmp_path,
            video_files={"earth-space-small": "local_video.mp4"},
        )
        assert "earth-space-small.mp4" in outputs
        with ZipFile(tmp_path / "earth-space-small.mp4.zip") as z:
            seqinfo_text = z.read("seqinfo.ini").decode("utf-8")
        seqinfo = configparser.ConfigParser()
        seqinfo.read_string(seqinfo_text)
        seq = seqinfo["Sequence"]
        assert seq["imWidth"] == "1280"
        assert seq["imHeight"] == "720"
