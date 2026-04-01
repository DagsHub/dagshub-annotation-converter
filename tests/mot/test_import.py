import math
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from dagshub_annotation_converter.converters.mot import (
    load_mot_from_dir,
    load_mot_from_file,
    load_mot_from_fs,
    load_mot_from_zip,
)
from dagshub_annotation_converter.formats.mot.bbox import _export_bbox_to_line, import_bbox_from_line
from dagshub_annotation_converter.formats.mot.context import MOTContext
from dagshub_annotation_converter.ir.video import CoordinateStyle, IRVideoBBoxFrameAnnotation


class TestMOTLineImport:
    # CVAT MOT 1.1 format (9 columns):
    # frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility

    def test_parse_basic_line(self, mot_context):
        line = "1,1,100,150,50,120,1,1,1.0"

        track_id, ann = import_bbox_from_line(line, mot_context)

        assert ann.frame_number == 0
        assert track_id == 1
        assert ann.left == 100
        assert ann.top == 150
        assert ann.width == 50
        assert ann.height == 120
        assert ann.keyframe is False
        assert ann.visibility == 1.0
        assert "person" in ann.categories
        assert ann.coordinate_style == CoordinateStyle.DENORMALIZED

    def test_parse_line_with_float_coords(self, mot_context, epsilon):
        line = "1,3,794.27,247.59,71.245,174.88,1,1,0.86014"

        track_id, ann = import_bbox_from_line(line, mot_context)

        assert ann.frame_number == 0
        assert track_id == 3
        assert math.isclose(ann.left, 794.27, abs_tol=epsilon)
        assert math.isclose(ann.top, 247.59, abs_tol=epsilon)
        assert math.isclose(ann.width, 71.245, abs_tol=epsilon)
        assert math.isclose(ann.height, 174.88, abs_tol=epsilon)
        assert math.isclose(ann.visibility, 0.86014, abs_tol=epsilon)

    def test_parse_line_with_class_id(self, mot_context):
        line = "1,2,500,200,150,100,1,2,1.0"

        track_id, ann = import_bbox_from_line(line, mot_context)

        assert track_id == 2
        assert "car" in ann.categories

    def test_parse_line_ignored_entry(self, mot_context):
        line = "1,1,100,150,50,120,0,1,1.0"

        ann = import_bbox_from_line(line, mot_context)

        assert ann is None

    def test_parse_line_with_partial_visibility(self, mot_context):
        line = "3,1,120,154,50,120,1,1,0.5"

        track_id, ann = import_bbox_from_line(line, mot_context)

        assert track_id == 1
        assert ann.frame_number == 2
        assert ann.visibility == 0.5

    def test_parse_line_with_zero_visibility(self, mot_context):
        line = "11,1,120,154,50,120,1,1,0.0"

        track_id, ann = import_bbox_from_line(line, mot_context)

        assert track_id == 1
        assert ann.frame_number == 10
        assert ann.visibility == 0.0
        assert "outside" not in ann.meta

    def test_parse_line_rejects_non_positive_frame_id(self, mot_context):
        with pytest.raises(ValueError, match="Invalid MOT frame_id 0"):
            import_bbox_from_line("0,1,100,150,50,120,1,1,1.0", mot_context)

    def test_parse_line_rejects_unknown_class_id_with_context(self, mot_context):
        with pytest.raises(ValueError, match="Unknown MOT class_id 999 in frame 1 track 2"):
            import_bbox_from_line("1,2,100,150,50,120,1,999,1.0", mot_context)


class TestMOTContext:
    def test_load_labels_preserves_names_with_spaces(self):
        categories = MOTContext.load_labels_from_string("traffic light\nfire hydrant\n")

        assert categories[1].name == "traffic light"
        assert categories[2].name == "fire hydrant"


class TestMOTFileImport:
    def test_load_from_file(self, sample_mot_file, mot_context):
        sequence = load_mot_from_file(sample_mot_file, mot_context)

        assert len(sequence.tracks) > 0

        frame_0 = sequence.annotations_by_frame().get(0, [])
        assert len(frame_0) == 2

        track_ids = {track.track_id for track, _ in frame_0}
        assert track_ids == {"1", "2"}

    def test_load_from_file_track_consistency(self, sample_mot_file, mot_context):
        sequence = load_mot_from_file(sample_mot_file, mot_context)

        track_1 = next(track for track in sequence.tracks if track.track_id == "1")
        track_1_frames = [ann.frame_number for ann in track_1.annotations]

        assert len(track_1_frames) == 5
        assert sorted(track_1_frames) == [0, 1, 2, 3, 4]

    def test_load_skips_comments(self, sample_mot_file, mot_context):
        sequence = load_mot_from_file(sample_mot_file, mot_context)
        assert len(sequence.to_annotations()) == 10


class TestMOTZipImport:
    def test_load_from_zip_same_as_dir(self, sample_mot_dir):
        res_dir = Path(__file__).parent / "res"
        with tempfile.NamedTemporaryFile(suffix=".zip") as f:
            zip_path = Path(f.name)
            with ZipFile(zip_path, "w") as z:
                for file_path in (res_dir / "gt").iterdir():
                    z.write(file_path, f"gt/{file_path.name}")
                z.write(res_dir / "seqinfo.ini", "seqinfo.ini")
            dir_sequence, dir_ctx = load_mot_from_dir(sample_mot_dir)
            zip_sequence, zip_ctx = load_mot_from_zip(zip_path)
            assert len(dir_sequence.tracks) == len(zip_sequence.tracks)
            assert dir_sequence.annotations_by_frame().keys() == zip_sequence.annotations_by_frame().keys()
            assert dir_ctx.frame_rate == zip_ctx.frame_rate
            assert dir_ctx.video_width == zip_ctx.video_width
            assert dir_ctx.categories == zip_ctx.categories

    def test_load_from_zip_nested_structure(self, sample_mot_dir):
        res_dir = Path(__file__).parent / "res"
        with tempfile.NamedTemporaryFile(suffix=".zip") as f:
            zip_path = Path(f.name)
            with ZipFile(zip_path, "w") as z:
                z.write(res_dir / "gt" / "gt.txt", "myseq/gt/gt.txt")
                z.write(res_dir / "gt" / "labels.txt", "myseq/gt/labels.txt")
                z.write(res_dir / "seqinfo.ini", "myseq/seqinfo.ini")
            zip_sequence, zip_ctx = load_mot_from_zip(zip_path)
            assert len(zip_sequence.tracks) > 0
            assert len(zip_sequence.to_annotations()) == 10
            assert zip_ctx.categories[1].name == "person"
            assert zip_ctx.categories[2].name == "car"


class TestMOTDirectoryImport:
    def test_load_from_dir(self, sample_mot_dir):
        sequence, context = load_mot_from_dir(sample_mot_dir)

        assert context.frame_rate == 30
        assert context.video_width == 1920
        assert context.video_height == 1080
        assert context.sequence_name == "test_sequence"
        assert sequence.filename == "test_sequence"

        assert context.categories[1].name == "person"
        assert context.categories[2].name == "car"

    def test_load_from_fs_multiple_sequences(self, sample_mot_dir, tmp_path):
        shutil.copytree(sample_mot_dir, tmp_path / "seq_a")
        shutil.copytree(sample_mot_dir, tmp_path / "seq_b")

        loaded = load_mot_from_fs(tmp_path)
        assert set(loaded.keys()) == {Path("seq_a"), Path("seq_b")}
        assert all(len(sequence.tracks) > 0 for sequence, _ in loaded.values())

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

        from dagshub_annotation_converter.util.video import VideoProbeResult

        probed_paths = []

        def fake_probe(path):
            probed_paths.append(Path(path))
            return VideoProbeResult(width=720, height=480, fps=24.0, frame_count=400)

        monkeypatch.setattr("dagshub_annotation_converter.converters.mot.probe_video", fake_probe)

        loaded = load_mot_from_fs(labels_dir, datasource_path="data/videos")
        sequence, context = loaded[Path("earth-space-small.mp4.zip")]

        assert context.video_width == 720
        assert context.video_height == 480
        assert context.sequence_length == 400
        assert 0 in sequence.annotations_by_frame()
        assert probed_paths == [video_path]


class TestMOTFrameNumberConversion:
    def test_mot_import_converts_frame_to_0_based(self, mot_context):
        line1 = "1,1,100,150,50,120,1,1,1.0"
        _, ann1 = import_bbox_from_line(line1, mot_context)
        assert ann1.frame_number == 0

        line10 = "10,1,100,150,50,120,1,1,1.0"
        _, ann10 = import_bbox_from_line(line10, mot_context)
        assert ann10.frame_number == 9

    def test_mot_export_converts_frame_to_1_based(self, mot_context):
        ann0 = IRVideoBBoxFrameAnnotation(
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
        line0 = _export_bbox_to_line(ann0, 1, mot_context)
        assert line0.split(",")[0] == "1"

        ann9 = IRVideoBBoxFrameAnnotation(
            frame_number=9,
            left=100,
            top=150,
            width=50,
            height=120,
            video_width=1920,
            video_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
        )
        line9 = _export_bbox_to_line(ann9, 1, mot_context)
        assert line9.split(",")[0] == "10"
