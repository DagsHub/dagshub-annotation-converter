import configparser
import math
from pathlib import Path
from zipfile import ZipFile

import pytest

from dagshub_annotation_converter.converters.mot import (
    export_mot_sequences_to_dirs,
    export_mot_to_dir,
    export_to_mot,
    load_mot_from_fs,
    load_mot_from_file,
)
from dagshub_annotation_converter.formats.mot.bbox import _export_bbox_to_line
from dagshub_annotation_converter.util.video import VideoProbeResult
from dagshub_annotation_converter.formats.mot.context import MOTContext
from dagshub_annotation_converter.ir.video import (
    CoordinateStyle,
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
    IRVideoSequence,
)
from tests.video_helpers import sequence_from_annotations


class TestMOTLineExport:
    def test_export_basic_annotation(self, mot_context):
        ann = IRVideoBBoxFrameAnnotation(
            imported_id="1",
            frame_number=0,  # 0-based IR frame
            left=100,
            top=150,
            width=50,
            height=120,
            video_width=1920,
            video_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=1.0,
        )

        line = _export_bbox_to_line(ann, 1, mot_context)

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

    def test_export_normalized_annotation_auto_denormalizes(self, mot_context):
        ann = IRVideoBBoxFrameAnnotation(
            imported_id="1",
            frame_number=0,
            left=0.1,
            top=0.1,
            width=0.2,
            height=0.2,
            video_width=1920,
            video_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.NORMALIZED,
            visibility=1.0,
        )

        line = _export_bbox_to_line(ann, 1, mot_context)
        parts = line.split(",")
        assert float(parts[2]) == pytest.approx(0.1 * 1920)  # x
        assert float(parts[3]) == pytest.approx(0.1 * 1080)  # y
        assert float(parts[4]) == pytest.approx(0.2 * 1920)  # w
        assert float(parts[5]) == pytest.approx(0.2 * 1080)  # h

    def test_export_with_partial_visibility(self, mot_context):
        ann = IRVideoBBoxFrameAnnotation(
            imported_id="2",
            frame_number=3,
            left=120,
            top=154,
            width=50,
            height=120,
            video_width=1920,
            video_height=1080,
            categories={"car": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=0.5,
        )

        line = _export_bbox_to_line(ann, 2, mot_context)
        parts = line.split(",")

        assert parts[7] == "2"
        assert float(parts[8]) == 0.5


class TestMOTFileExport:
    def test_export_to_file_populates_categories_from_fresh_context(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=1920, video_height=1080, sequence_name="test_sequence")
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,
                keyframe=False,
                left=100,
                top=150,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"Man": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ]

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence_from_annotations(annotations), context, output_path)

        line = output_path.read_text().strip()
        assert line
        assert line.split(",")[7] == "1"
        assert context.categories["Man"].id == 1

    def test_export_to_file_populates_multiple_categories_in_stable_order(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=1920, video_height=1080, sequence_name="test_sequence")
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,
                keyframe=False,
                left=100,
                top=150,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"Man": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="2",
                frame_number=0,
                keyframe=False,
                left=500,
                top=200,
                width=150,
                height=100,
                video_width=1920,
                video_height=1080,
                categories={"Woman": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=1,
                keyframe=False,
                left=110,
                top=152,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"Man": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
        ]

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence_from_annotations(annotations), context, output_path)

        lines = output_path.read_text().splitlines()
        assert [line.split(",")[7] for line in lines] == ["1", "2", "1"]
        assert context.categories["Man"].id == 1
        assert context.categories["Woman"].id == 2

    def test_export_to_file(self, mot_context, tmp_path):
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,  # 0-based IR frame
                keyframe=False,
                left=100,
                top=150,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="2",
                frame_number=0,  # 0-based IR frame
                keyframe=False,
                left=500,
                top=200,
                width=150,
                height=100,
                video_width=1920,
                video_height=1080,
                categories={"car": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=1,  # 0-based IR frame
                keyframe=False,
                left=110,
                top=152,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            ),
        ]

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence_from_annotations(annotations), mot_context, output_path)

        content = output_path.read_text()
        lines = [line for line in content.strip().split("\n") if line and not line.startswith("#")]

        assert len(lines) == 3

        first_line_parts = lines[0].split(",")
        assert first_line_parts[0] == "1"

    def test_export_to_file_denormalizes_normalized_track(self, mot_context, tmp_path):
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,
                keyframe=False,
                left=0.1,
                top=0.1,
                width=0.2,
                height=0.2,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
            ),
        ]

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence_from_annotations(annotations), mot_context, output_path)

        line = output_path.read_text().strip()
        parts = line.split(",")
        assert math.isclose(float(parts[2]), 192, abs_tol=1)
        assert math.isclose(float(parts[3]), 108, abs_tol=1)

    def test_export_roundtrip(self, mot_context, sample_mot_file, tmp_path):
        sequence = load_mot_from_file(sample_mot_file, mot_context)

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence, mot_context, output_path)

        reimported = load_mot_from_file(output_path, mot_context)

        assert len(reimported.annotations_by_frame()) == len(sequence.annotations_by_frame())

        for frame_num, frame_entries in sequence.annotations_by_frame().items():
            assert len(reimported.annotations_by_frame()[frame_num]) == len(frame_entries)

    def test_export_converts_non_numeric_track_identifier_to_numeric_id(self, mot_context, tmp_path):
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
            visibility=1.0,
        )
        sequence = IRVideoSequence(
            tracks=[IRVideoAnnotationTrack.from_annotations([ann], object_id="track_person_1")]
        )

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence, mot_context, output_path)

        line = output_path.read_text().strip()
        assert line
        assert line.split(",")[1].isdigit()

    def test_export_interpolates_between_sparse_keyframes(self, mot_context, tmp_path):
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
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
                visibility=1.0,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=4,
                keyframe=True,
                left=140,
                top=158,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
                visibility=1.0,
            ),
        ]

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence_from_annotations(annotations), mot_context, output_path)
        lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]

        # Frames should be densified from 0..4 (MOT 1..5)
        assert len(lines) == 5
        mot_frames = [int(line.split(",")[0]) for line in lines]
        assert mot_frames == [1, 2, 3, 4, 5]

    def test_export_hides_outside_ranges_between_keyframes(self, mot_context, tmp_path):
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
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
                visibility=1.0,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=3,
                keyframe=True,
                left=130,
                top=156,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
                visibility=0.0,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=5,
                keyframe=True,
                left=150,
                top=160,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
                visibility=1.0,
            ),
        ]

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence_from_annotations(annotations), mot_context, output_path)
        lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]

        # Frame 3 is emitted as a hidden boundary marker, frame 4 remains hidden until re-appearance at frame 5.
        mot_frames = [int(line.split(",")[0]) for line in lines]
        assert mot_frames == [1, 2, 3, 4, 6]
        assert float(lines[3].split(",")[8]) == 0.0

    def test_export_extends_last_visible_segment_to_seq_length(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=1920, video_height=1080, sequence_length=12)
        context.categories.add("person", 1)
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
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
                visibility=1.0,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=3,
                keyframe=True,
                left=130,
                top=156,
                width=50,
                height=120,
                video_width=1920,
                video_height=1080,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
                visibility=1.0,
            ),
        ]

        output_path = tmp_path / "gt.txt"
        export_to_mot(sequence_from_annotations(annotations, sequence_length=12), context, output_path)
        lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]
        mot_frames = [int(line.split(",")[0]) for line in lines]
        assert mot_frames == list(range(1, 13))

    def test_export_to_dir_uses_probed_dimensions_for_seqinfo(self, tmp_path, monkeypatch):
        context = MOTContext(frame_rate=30, video_width=None, video_height=None, sequence_name="test_sequence")
        context.categories.add("person", 1)
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,
                left=100,
                top=150,
                width=50,
                height=120,
                video_width=0,
                video_height=0,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ]

        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.mot.probe_video",
            lambda _: VideoProbeResult(width=1280, height=720, fps=25.0, frame_count=100),
        )

        out_dir = tmp_path / "mot_out"
        export_mot_to_dir(
            sequence_from_annotations(annotations),
            context,
            out_dir,
            video_file="local_video.mp4",
            create_seqinfo=True,
        )

        seqinfo = configparser.ConfigParser()
        seqinfo.read(out_dir / "seqinfo.ini")
        seq = seqinfo["Sequence"]
        assert seq["imWidth"] == "1280"
        assert seq["imHeight"] == "720"
        assert seq["frameRate"] == "25"

    def test_export_to_dir_skips_seqinfo_by_default(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=1280, video_height=720, sequence_name="test_sequence")
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,
                left=100,
                top=150,
                width=50,
                height=120,
                video_width=1280,
                video_height=720,
                categories={"Man": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ]

        out_dir = tmp_path / "mot_out"
        export_mot_to_dir(sequence_from_annotations(annotations), context, out_dir)

        assert (out_dir / "gt" / "gt.txt").exists()
        assert (out_dir / "gt" / "labels.txt").exists()
        assert (out_dir / "gt" / "labels.txt").read_text() == "Man\n"
        assert not (out_dir / "seqinfo.ini").exists()

    def test_export_sequences_to_dirs_populates_categories_for_fresh_context(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=1920, video_height=1080, sequence_name="default")
        ann_a = IRVideoBBoxFrameAnnotation(
            imported_id="1",
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            video_width=1920,
            video_height=1080,
            categories={"Man": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="earth-space-small.mp4",
        )
        ann_b = IRVideoBBoxFrameAnnotation(
            imported_id="2",
            frame_number=0,
            left=500,
            top=200,
            width=150,
            height=100,
            video_width=1920,
            video_height=1080,
            categories={"Woman": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="jelly.mp4",
        )

        outputs = export_mot_sequences_to_dirs(
            [sequence_from_annotations([ann_a]), sequence_from_annotations([ann_b])],
            context,
            tmp_path,
        )

        assert set(outputs.keys()) == {"earth-space-small.mp4", "jelly.mp4"}
        with ZipFile(tmp_path / "labels" / "earth-space-small.mp4.zip") as z:
            assert z.read("gt/labels.txt").decode("utf-8") == "Man\n"
        with ZipFile(tmp_path / "labels" / "jelly.mp4.zip") as z:
            assert z.read("gt/labels.txt").decode("utf-8") == "Woman\n"

    def test_export_to_dir_uses_probed_frame_count_for_seqinfo(self, tmp_path, monkeypatch):
        context = MOTContext(frame_rate=30, video_width=None, video_height=None, sequence_name="test_sequence")
        context.categories.add("person", 1)
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,
                left=100,
                top=150,
                width=50,
                height=120,
                video_width=0,
                video_height=0,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ]

        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.mot.probe_video",
            lambda _: VideoProbeResult(width=1280, height=720, fps=25.0, frame_count=40),
        )

        out_dir = tmp_path / "mot_out"
        export_mot_to_dir(
            sequence_from_annotations(annotations),
            context,
            out_dir,
            video_file="local_video.mp4",
            create_seqinfo=True,
        )

        seqinfo = configparser.ConfigParser()
        seqinfo.read(out_dir / "seqinfo.ini")
        seq = seqinfo["Sequence"]
        assert seq["seqLength"] == "40"

        gt_lines = [line for line in (out_dir / "gt" / "gt.txt").read_text().splitlines() if line]
        max_gt_frame = max(int(line.split(",")[0]) for line in gt_lines)
        assert max_gt_frame == 40

    def test_export_raises_without_dimensions_or_video_file(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=None, video_height=None)
        context.categories.add("person", 1)
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=0,
                left=100,
                top=150,
                width=50,
                height=120,
                video_width=0,
                video_height=0,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        ]

        output_path = tmp_path / "gt.txt"
        with pytest.raises(ValueError, match="Cannot determine frame dimensions for MOT export"):
            export_to_mot(sequence_from_annotations(annotations), context, output_path)

    def test_export_sequences_to_dirs_groups_by_filename(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=1920, video_height=1080, sequence_name="default")
        context.categories.add("person", 1)
        ann_a = IRVideoBBoxFrameAnnotation(
            imported_id="1",
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            video_width=1920,
            video_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="earth-space-small.mp4",
        )
        ann_b = IRVideoBBoxFrameAnnotation(
            imported_id="2",
            frame_number=0,
            left=500,
            top=200,
            width=150,
            height=100,
            video_width=1920,
            video_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="jelly.mp4",
        )

        outputs = export_mot_sequences_to_dirs(
            [sequence_from_annotations([ann_a]), sequence_from_annotations([ann_b])],
            context,
            tmp_path,
        )
        assert set(outputs.keys()) == {"earth-space-small.mp4", "jelly.mp4"}
        assert (tmp_path / "labels" / "earth-space-small.mp4.zip").exists()
        assert (tmp_path / "labels" / "jelly.mp4.zip").exists()
        with ZipFile(tmp_path / "labels" / "earth-space-small.mp4.zip") as z:
            assert "gt/gt.txt" in z.namelist()
            assert "gt/labels.txt" in z.namelist()
            assert "seqinfo.ini" not in z.namelist()
        with ZipFile(tmp_path / "labels" / "jelly.mp4.zip") as z:
            assert "gt/gt.txt" in z.namelist()
            assert "gt/labels.txt" in z.namelist()
            assert "seqinfo.ini" not in z.namelist()

    def test_export_sequences_to_dirs_accepts_stem_video_file_keys(self, tmp_path, monkeypatch):
        context = MOTContext(frame_rate=30, video_width=None, video_height=None, sequence_name="default")
        context.categories.add("person", 1)
        ann = IRVideoBBoxFrameAnnotation(
            imported_id="1",
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            video_width=None,
            video_height=None,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="earth-space-small.mp4",
        )

        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.mot.probe_video",
            lambda _: VideoProbeResult(width=1280, height=720, fps=25.0, frame_count=100),
        )

        outputs = export_mot_sequences_to_dirs(
            [sequence_from_annotations([ann])],
            context,
            tmp_path,
            video_files={"earth-space-small": "local_video.mp4"},
            create_seqinfo=True,
        )
        assert "earth-space-small.mp4" in outputs
        with ZipFile(tmp_path / "labels" / "earth-space-small.mp4.zip") as z:
            seqinfo_text = z.read("seqinfo.ini").decode("utf-8")
        seqinfo = configparser.ConfigParser()
        seqinfo.read_string(seqinfo_text)
        seq = seqinfo["Sequence"]
        assert seq["imWidth"] == "1280"
        assert seq["imHeight"] == "720"
        assert seq["frameRate"] == "25"

    def test_export_sequences_to_dirs_roundtrips_via_load_from_fs(self, tmp_path, monkeypatch):
        context = MOTContext(frame_rate=30, video_width=1920, video_height=1080, sequence_name="default")
        context.categories.add("person", 1)

        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()
        (videos_dir / "earth-space-small.mp4").write_bytes(b"fake-video-a")
        (videos_dir / "jelly.mp4").write_bytes(b"fake-video-b")

        ann_a = IRVideoBBoxFrameAnnotation(
            imported_id="1",
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            video_width=1920,
            video_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="earth-space-small.mp4",
        )
        ann_b = IRVideoBBoxFrameAnnotation(
            imported_id="2",
            frame_number=0,
            left=500,
            top=200,
            width=150,
            height=100,
            video_width=1920,
            video_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            filename="jelly.mp4",
        )

        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.mot.probe_video",
            lambda _: VideoProbeResult(width=1920, height=1080, fps=30.0, frame_count=1),
        )

        export_mot_sequences_to_dirs(
            [sequence_from_annotations([ann_a]), sequence_from_annotations([ann_b])],
            context,
            tmp_path,
        )
        loaded = load_mot_from_fs(tmp_path)

        assert set(loaded.keys()) == {Path("earth-space-small.mp4"), Path("jelly.mp4")}
        assert {sequence.filename for sequence, _ in loaded.values()} == {"earth-space-small.mp4", "jelly.mp4"}

    def test_export_to_dir_seq_length_matches_exported_gt(self, tmp_path):
        context = MOTContext(frame_rate=30, video_width=720, video_height=480, sequence_name="test_sequence")
        context.categories.add("person", 2)
        context.categories.add("woman", 1)
        annotations = [
            IRVideoBBoxFrameAnnotation(
                imported_id="1",
                frame_number=377,
                left=576,
                top=209,
                width=40,
                height=34,
                video_width=720,
                video_height=480,
                categories={"person": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
                sequence_length=381,
            ),
            IRVideoBBoxFrameAnnotation(
                imported_id="2",
                frame_number=366,
                left=200,
                top=84,
                width=232,
                height=148,
                video_width=720,
                video_height=480,
                categories={"woman": 1.0},
                coordinate_style=CoordinateStyle.DENORMALIZED,
                sequence_length=381,
            ),
        ]

        out_dir = tmp_path / "mot_out"
        export_mot_to_dir(
            sequence_from_annotations(annotations, sequence_length=381),
            context,
            out_dir,
            create_seqinfo=True,
        )

        seqinfo = configparser.ConfigParser()
        seqinfo.read(out_dir / "seqinfo.ini")
        seq = seqinfo["Sequence"]
        assert seq["seqLength"] == "381"

        gt_lines = [line for line in (out_dir / "gt" / "gt.txt").read_text().splitlines() if line]
        max_gt_frame = max(int(line.split(",")[0]) for line in gt_lines)
        assert max_gt_frame == 381
