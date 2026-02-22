import tempfile
from pathlib import Path
from zipfile import ZipFile
import math
import shutil

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.mot.bbox import import_bbox_from_line
from dagshub_annotation_converter.converters.mot import (
    load_mot_from_file,
    load_mot_from_dir,
    load_mot_from_zip,
    load_mot_from_fs,
)


class TestMOTLineImport:
    # CVAT MOT 1.1 format (9 columns):
    # frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility

    def test_parse_basic_line(self, mot_context):
        line = "1,1,100,150,50,120,1,1,1.0"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.frame_number == 0
        assert ann.track_id == 1
        assert ann.left == 100
        assert ann.top == 150
        assert ann.width == 50
        assert ann.height == 120
        assert ann.keyframe
        assert ann.visibility == 1.0
        assert "person" in ann.categories
        assert ann.coordinate_style == CoordinateStyle.DENORMALIZED

    def test_parse_line_with_float_coords(self, mot_context, epsilon):
        line = "1,3,794.27,247.59,71.245,174.88,1,1,0.86014"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.frame_number == 0  # MOT frame 1 -> IR frame 0
        assert ann.track_id == 3
        assert math.isclose(ann.left, 794.27, abs_tol=epsilon)
        assert math.isclose(ann.top, 247.59, abs_tol=epsilon)
        assert math.isclose(ann.width, 71.245, abs_tol=epsilon)
        assert math.isclose(ann.height, 174.88, abs_tol=epsilon)
        assert math.isclose(ann.visibility, 0.86014, abs_tol=epsilon)

    def test_parse_line_with_class_id(self, mot_context):
        line = "1,2,500,200,150,100,1,2,1.0"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.track_id == 2
        assert "car" in ann.categories

    def test_parse_line_ignored_entry(self, mot_context):
        line = "1,1,100,150,50,120,0,1,1.0"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.meta.get("ignored")

    def test_parse_line_with_partial_visibility(self, mot_context):
        line = "3,1,120,154,50,120,1,1,0.5"
        
        ann = import_bbox_from_line(line, mot_context)
        
        assert ann.frame_number == 2  # MOT frame 3 -> IR frame 2
        assert ann.visibility == 0.5

    def test_parse_line_with_zero_visibility_marks_outside(self, mot_context):
        line = "11,1,120,154,50,120,1,1,0.0"

        ann = import_bbox_from_line(line, mot_context)

        assert ann.frame_number == 10
        assert ann.visibility == 0.0
        assert ann.meta.get("outside") is True


class TestMOTFileImport:
    def test_load_from_file(self, sample_mot_file, mot_context):
        annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        assert len(annotations) > 0
        
        # IR Frame 0 (MOT Frame 1) should have 2 annotations (track 1 and track 2)
        frame_0_anns = annotations.get(0, [])
        assert len(frame_0_anns) == 2
        
        track_ids = {ann.track_id for ann in frame_0_anns}
        assert track_ids == {1, 2}

    def test_load_from_file_track_consistency(self, sample_mot_file, mot_context):
        annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        # Track 1 should appear in all 5 frames (IR frames 0-4)
        track_1_frames = []
        for frame_num, anns in annotations.items():
            for ann in anns:
                if ann.track_id == 1:
                    track_1_frames.append(frame_num)
        
        assert len(track_1_frames) == 5
        assert sorted(track_1_frames) == [0, 1, 2, 3, 4]

    def test_load_skips_comments(self, sample_mot_file, mot_context):
        annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        total_anns = sum(len(anns) for anns in annotations.values())
        assert total_anns == 10


class TestMOTZipImport:
    def test_load_from_zip_same_as_dir(self, sample_mot_dir):
        res_dir = Path(__file__).parent / "res"
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)
        try:
            with ZipFile(zip_path, "w") as z:
                for f in (res_dir / "gt").iterdir():
                    z.write(f, f"gt/{f.name}")
                z.write(res_dir / "seqinfo.ini", "seqinfo.ini")
            dir_anns, dir_ctx = load_mot_from_dir(sample_mot_dir)
            zip_anns, zip_ctx = load_mot_from_zip(zip_path)
            assert len(dir_anns) == len(zip_anns)
            for frame, anns in dir_anns.items():
                assert frame in zip_anns
                assert len(anns) == len(zip_anns[frame])
            assert dir_ctx.frame_rate == zip_ctx.frame_rate
            assert dir_ctx.image_width == zip_ctx.image_width
            assert dir_ctx.categories == zip_ctx.categories
        finally:
            zip_path.unlink(missing_ok=True)

    def test_load_from_zip_nested_structure(self, sample_mot_dir):
        res_dir = Path(__file__).parent / "res"
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)
        try:
            with ZipFile(zip_path, "w") as z:
                z.write(res_dir / "gt" / "gt.txt", "myseq/gt/gt.txt")
                z.write(res_dir / "gt" / "labels.txt", "myseq/gt/labels.txt")
                z.write(res_dir / "seqinfo.ini", "myseq/seqinfo.ini")
            zip_anns, zip_ctx = load_mot_from_zip(zip_path)
            assert len(zip_anns) > 0
            total = sum(len(a) for a in zip_anns.values())
            assert total == 10
            assert zip_ctx.categories == {1: "person", 2: "car"}
        finally:
            zip_path.unlink(missing_ok=True)


class TestMOTDirectoryImport:
    def test_load_from_dir(self, sample_mot_dir):
        annotations, context = load_mot_from_dir(sample_mot_dir)
        
        assert context.frame_rate == 30.0
        assert context.image_width == 1920
        assert context.image_height == 1080
        assert context.seq_name == "test_sequence"
        
        assert context.categories == {1: "person", 2: "car"}

    def test_load_from_fs_multiple_sequences(self, sample_mot_dir, tmp_path):
        shutil.copytree(sample_mot_dir, tmp_path / "seq_a")
        shutil.copytree(sample_mot_dir, tmp_path / "seq_b")

        loaded = load_mot_from_fs(tmp_path)
        assert set(loaded.keys()) == {"seq_a", "seq_b"}
        assert all(len(anns) > 0 for anns, _ in loaded.values())

    def test_load_from_fs_uses_datasource_path_layout(self, tmp_path, monkeypatch):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        video_path = tmp_path / "data" / "data" / "videos" / "earth-space-small.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake-video")

        zip_path = labels_dir / "earth-space-small.mp4.zip"
        with ZipFile(zip_path, "w") as z:
            z.writestr("gt/gt.txt", "1,1,100,150,50,120,1,1,1.0\n")
            z.writestr("gt/labels.txt", "person\n")

        probed_paths = []

        def fake_dimensions(path):
            probed_paths.append(Path(path))
            return 720, 480, 24.0

        monkeypatch.setattr("dagshub_annotation_converter.converters.mot.get_video_dimensions", fake_dimensions)
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.mot.get_video_frame_count",
            lambda _: 400,
        )

        loaded = load_mot_from_fs(labels_dir, datasource_path="data/videos")
        anns, context = loaded["earth-space-small.mp4.zip"]

        assert context.image_width == 720
        assert context.image_height == 480
        assert context.seq_length == 400
        assert 0 in anns
        assert probed_paths == [video_path]


class TestMOTFrameNumberConversion:
    def test_mot_import_converts_frame_to_0_based(self, mot_context):
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
        from dagshub_annotation_converter.formats.mot.bbox import export_bbox_to_line
        
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
        from dagshub_annotation_converter.formats.mot.bbox import import_bbox_from_line, export_bbox_to_line
        
        original_line = "1,1,100,150,50,120,1,1,1.0"
        
        # Import (MOT 1 -> IR 0)
        ann = import_bbox_from_line(original_line, mot_context)
        assert ann.frame_number == 0
        
        # Export (IR 0 -> MOT 1)
        exported_line = export_bbox_to_line(ann, mot_context)
        
        # Frame number should be back to 1
        parts = exported_line.split(",")
        assert parts[0] == "1"
