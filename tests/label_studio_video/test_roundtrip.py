import copy
import math
import tempfile
from pathlib import Path

import pytest

from dagshub_annotation_converter.converters.cvat import (
    export_cvat_video_to_xml_bytes,
    load_cvat_from_xml_bytes,
    load_cvat_from_xml_file,
)
from dagshub_annotation_converter.converters.label_studio_video import (
    ls_video_task_to_video_ir,
    video_ir_to_ls_video_tasks,
)
from dagshub_annotation_converter.converters.mot import export_to_mot, load_mot_from_file
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask
from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
    VideoRectangleSequenceItem,
    VideoRectangleValue,
)
from dagshub_annotation_converter.formats.mot import MOTContext
from dagshub_annotation_converter.ir.video import IRVideoSequence
from tests.video_helpers import annotations_by_track_frame, flatten_sequence_with_track_ids

LS_VIDEO_PATH = "/data/video.mp4"


def _sequence_from_ls_results(results, filename: str = "/data/video.mp4") -> IRVideoSequence:
    tracks = []
    frames_counts = []
    for result in results:
        ls_ann = VideoRectangleAnnotation.model_validate(result)
        tracks.append(ls_ann.to_ir_track())
        if ls_ann.value.framesCount is not None and ls_ann.value.framesCount > 0:
            frames_counts.append(ls_ann.value.framesCount)
    return IRVideoSequence(
        tracks=tracks,
        filename=filename,
        sequence_length=max(frames_counts) if frames_counts else None,
    )


def _single_track_sequence(ls_ann: VideoRectangleAnnotation) -> IRVideoSequence:
    frames_count = (
        ls_ann.value.framesCount
        if ls_ann.value.framesCount is not None and ls_ann.value.framesCount > 0
        else None
    )
    return IRVideoSequence(tracks=[ls_ann.to_ir_track()], sequence_length=frames_count)


def _ls_tasks(sequence: IRVideoSequence):
    return video_ir_to_ls_video_tasks(sequence, video_path=LS_VIDEO_PATH)


class TestMOTToLabelStudioRoundtrip:
    """Tests for MOT <-> Label Studio Video conversion.

    Uses CVAT MOT 1.1 format (9 columns):
    frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    """

    @pytest.fixture
    def mot_context(self) -> MOTContext:
        context = MOTContext(
            frame_rate=30,
            video_width=1920,
            video_height=1080,
        )
        context.categories.add("person", 1)
        context.categories.add("car", 2)
        return context

    @pytest.fixture
    def sample_mot_file(self) -> Path:
        return Path(__file__).parent.parent / "mot" / "res" / "gt" / "gt.txt"

    def test_mot_to_ls_video(self, sample_mot_file, mot_context):
        mot_sequence = load_mot_from_file(sample_mot_file, mot_context)
        ls_tasks = _ls_tasks(mot_sequence)

        assert len(ls_tasks) == 1
        task = ls_tasks[0]
        assert len(task.annotations) > 0
        assert len(task.annotations[0].result) == 2

    def test_mot_to_ls_treats_frames_as_independent(self, sample_mot_file, mot_context):
        mot_sequence = load_mot_from_file(sample_mot_file, mot_context)
        ls_tasks = _ls_tasks(mot_sequence)
        results = ls_tasks[0].annotations[0].result
        for result in results:
            assert all(item.enabled is False for item in result.value.sequence)

    def test_ls_video_to_mot(self, sample_ls_video_task_data, mot_context):
        sequence = _sequence_from_ls_results(sample_ls_video_task_data["annotations"][0]["result"])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
            output_path = Path(f.name)
            export_to_mot(sequence, mot_context, output_path)

            content = output_path.read_text()
            lines = [line for line in content.strip().split("\n") if line and not line.startswith("#")]

            assert len(lines) == 10
            for line in lines:
                assert len(line.split(",")) == 9

    def test_mot_ls_mot_roundtrip(self, sample_mot_file, mot_context):
        original_sequence = load_mot_from_file(sample_mot_file, mot_context)
        ls_tasks = _ls_tasks(original_sequence)
        reconstructed_sequence = ls_video_task_to_video_ir(ls_tasks[0])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
            output_path = Path(f.name)
            export_to_mot(reconstructed_sequence, mot_context, output_path)
            final_sequence = load_mot_from_file(output_path, mot_context)

            assert len(final_sequence.annotations_by_frame()) == len(original_sequence.annotations_by_frame())

            for frame_num, original_entries in original_sequence.annotations_by_frame().items():
                final_entries = final_sequence.annotations_by_frame()[frame_num]
                assert len(final_entries) == len(original_entries)

                orig_frame = sorted(
                    ((track.track_id, ann) for track, ann in original_entries),
                    key=lambda item: item[0],
                )
                final_frame = sorted(((track.track_id, ann) for track, ann in final_entries), key=lambda item: item[0])

                for (orig_track_id, orig), (final_track_id, final) in zip(orig_frame, final_frame):
                    assert orig_track_id == final_track_id
                    assert math.isclose(orig.left, final.left, abs_tol=1)
                    assert math.isclose(orig.top, final.top, abs_tol=1)

    def test_mot_ls_roundtrip_preserves_track_frames_and_keyframes(self, sample_mot_file, mot_context):
        original_sequence = load_mot_from_file(sample_mot_file, mot_context)
        ls_tasks = _ls_tasks(original_sequence)
        task = ls_tasks[0]
        for result in task.annotations[0].result:
            for item in result.value.sequence:
                assert isinstance(item.enabled, bool)

        reconstructed_sequence = ls_video_task_to_video_ir(task)

        original_by_track_frame = annotations_by_track_frame(original_sequence)
        reconstructed_by_track_frame = annotations_by_track_frame(reconstructed_sequence)

        assert reconstructed_by_track_frame.keys() == original_by_track_frame.keys()
        for key, original_ann in original_by_track_frame.items():
            reconstructed_ann = reconstructed_by_track_frame[key]
            assert reconstructed_ann.keyframe == original_ann.keyframe

    def test_ls_hidden_gap_is_not_interpolated_through_mot(self, mot_context):
        ls_ann = VideoRectangleAnnotation(
            id="track_person_gap",
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(frame=1, x=10.0, y=20.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=10, x=20.0, y=20.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=11, x=20.0, y=20.0, width=5.0, height=10.0, enabled=False),
                    VideoRectangleSequenceItem(frame=100, x=60.0, y=30.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=110, x=70.0, y=30.0, width=5.0, height=10.0, enabled=True),
                ],
                labels=["person"],
            ),
        )
        sequence = _single_track_sequence(ls_ann)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
            output_path = Path(f.name)
            export_to_mot(sequence, mot_context, output_path)
            mot_lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]
            mot_frames = [int(line.split(",")[0]) for line in mot_lines]

            assert 11 in mot_frames
            assert 100 in mot_frames
            assert all(frame <= 11 or frame >= 100 for frame in mot_frames)

            mot_sequence = load_mot_from_file(output_path, mot_context)
            ls_tasks = _ls_tasks(mot_sequence)
            seq = ls_tasks[0].annotations[0].result[0].value.sequence
            by_frame = {item.frame: item for item in seq}

            assert by_frame[11].enabled is False
            assert by_frame[100].enabled is False

    def test_ls_track_removed_after_frame_does_not_extend_to_end_through_mot(self, mot_context):
        ls_ann = VideoRectangleAnnotation(
            id="track_person_removed",
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(frame=1, x=10.0, y=20.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=10, x=20.0, y=20.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=11, x=20.0, y=20.0, width=5.0, height=10.0, enabled=False),
                ],
                labels=["person"],
            ),
        )
        sequence = _single_track_sequence(ls_ann)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
            output_path = Path(f.name)
            export_to_mot(sequence, mot_context, output_path)
            mot_lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]
            mot_frames = [int(line.split(",")[0]) for line in mot_lines]
            assert mot_frames == list(range(1, 12))

            last_visibility = float(mot_lines[-1].split(",")[8])
            assert last_visibility == 1.0

            mot_sequence = load_mot_from_file(output_path, mot_context)
            ls_tasks = _ls_tasks(mot_sequence)
            seq = ls_tasks[0].annotations[0].result[0].value.sequence

            assert seq[-1].frame == 11
            assert seq[-1].enabled is False

    def test_ls_segment_to_end_exports_dense_rows_using_frames_count(self, mot_context):
        ls_ann = VideoRectangleAnnotation(
            id="track_person_to_end",
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(frame=1, x=10.0, y=20.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=10, x=20.0, y=20.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=11, x=20.0, y=20.0, width=5.0, height=10.0, enabled=False),
                    VideoRectangleSequenceItem(frame=100, x=60.0, y=30.0, width=5.0, height=10.0, enabled=True),
                ],
                labels=["person"],
                framesCount=120,
            ),
        )
        sequence = _single_track_sequence(ls_ann)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
            output_path = Path(f.name)
            export_to_mot(sequence, mot_context, output_path)
            mot_lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]
            mot_frames = [int(line.split(",")[0]) for line in mot_lines]

            assert 100 in mot_frames
            assert 120 in mot_frames
            assert all(frame <= 11 or frame >= 100 for frame in mot_frames)

            mot_sequence = load_mot_from_file(output_path, mot_context)
            ls_tasks = _ls_tasks(mot_sequence)
            seq = ls_tasks[0].annotations[0].result[0].value.sequence
            seq_frames = [item.frame for item in seq]

            assert 100 in seq_frames
            assert 120 in seq_frames
            assert seq_frames[-1] == 120


class TestCVATVideoToLabelStudioRoundtrip:
    @pytest.fixture
    def sample_cvat_video_xml(self) -> Path:
        return Path(__file__).parent.parent / "cvat_video" / "res" / "sample_video.xml"

    def test_cvat_to_ls_video(self, sample_cvat_video_xml):
        cvat_sequence = load_cvat_from_xml_file(sample_cvat_video_xml)
        ls_tasks = _ls_tasks(cvat_sequence)

        assert len(ls_tasks) == 1
        task = ls_tasks[0]
        assert len(task.annotations[0].result) == 2

    def test_cvat_to_ls_preserves_categories(self, sample_cvat_video_xml):
        cvat_sequence = load_cvat_from_xml_file(sample_cvat_video_xml)
        ls_tasks = _ls_tasks(cvat_sequence)

        labels = set()
        for result in ls_tasks[0].annotations[0].result:
            labels.update(result.value.labels)

        assert "person" in labels
        assert "car" in labels

    def test_cvat_to_ls_coordinate_conversion(self, sample_cvat_video_xml, epsilon):
        cvat_sequence = load_cvat_from_xml_file(sample_cvat_video_xml)
        ls_tasks = _ls_tasks(cvat_sequence)
        task = ls_tasks[0]

        person_result = next(result for result in task.annotations[0].result if "person" in result.value.labels)
        first_seq = person_result.value.sequence[0]
        assert math.isclose(first_seq.x, 5.208333, abs_tol=epsilon)
        assert math.isclose(first_seq.y, 13.888889, abs_tol=epsilon)

    def test_ls_export_raises_without_video_path_or_filename(self, sample_cvat_video_xml):
        cvat_sequence = load_cvat_from_xml_file(sample_cvat_video_xml)
        with pytest.raises(ValueError, match="Cannot determine video path for Label Studio video export"):
            video_ir_to_ls_video_tasks(cvat_sequence)

    def test_ls_cvat_ls_roundtrip_keeps_sparse_keyframe_interpolation(self):
        ls_ann = VideoRectangleAnnotation(
            id="track_sparse",
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(frame=1, x=10.0, y=20.0, width=5.0, height=10.0, enabled=True),
                    VideoRectangleSequenceItem(frame=9, x=20.0, y=20.0, width=5.0, height=10.0, enabled=False),
                    VideoRectangleSequenceItem(frame=13, x=25.0, y=21.0, width=5.0, height=10.0, enabled=False),
                    VideoRectangleSequenceItem(frame=14, x=26.0, y=22.0, width=5.0, height=10.0, enabled=False),
                    VideoRectangleSequenceItem(frame=15, x=27.0, y=23.0, width=5.0, height=10.0, enabled=False),
                    VideoRectangleSequenceItem(frame=25, x=30.0, y=30.0, width=4.0, height=7.0, enabled=False),
                    VideoRectangleSequenceItem(frame=127, x=35.0, y=35.0, width=6.0, height=7.0, enabled=True),
                    VideoRectangleSequenceItem(frame=175, x=35.0, y=35.0, width=6.0, height=7.0, enabled=False),
                    VideoRectangleSequenceItem(frame=378, x=40.0, y=40.0, width=6.0, height=7.0, enabled=True),
                ],
                labels=["person"],
            ),
        )
        sequence = _single_track_sequence(ls_ann)

        xml_bytes = export_cvat_video_to_xml_bytes(sequence, video_name="video.mp4")
        cvat_sequence = load_cvat_from_xml_bytes(xml_bytes)
        ls_tasks = _ls_tasks(cvat_sequence)

        seq = ls_tasks[0].annotations[0].result[0].value.sequence
        got = [(item.frame, item.enabled) for item in seq]
        expected = [
            (1, True),
            (9, False),
            (13, False),
            (14, False),
            (15, False),
            (25, False),
            (127, True),
            (175, False),
            (378, True),
        ]
        assert got == expected

    def test_cvat_dense_interpolated_rows_collapse_to_sparse_ls_keyframes(self):
        boxes = "\n".join(
            [
                (
                    f'<box frame="{frame}" outside="0" occluded="0" '
                    f'keyframe="{1 if frame == 0 else 0}" '
                    'xtl="100" ytl="100" xbr="200" ybr="200" z_order="0"/>'
                )
                for frame in range(0, 11)
            ]
        )
        boxes += (
            '\n<box frame="11" outside="1" occluded="0" keyframe="1" '
            'xtl="100" ytl="100" xbr="200" ybr="200" z_order="0"/>'
        )
        xml_bytes = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations><version>1.1</version><meta><task>"
            "<size>20</size><mode>interpolation</mode>"
            "<original_size><width>1920</width><height>1080</height></original_size>"
            f'</task></meta><track id="0" label="person" source="manual">{boxes}</track></annotations>'
        ).encode("utf-8")

        cvat_sequence = load_cvat_from_xml_bytes(xml_bytes)
        ls_tasks = _ls_tasks(cvat_sequence)
        seq = ls_tasks[0].annotations[0].result[0].value.sequence

        assert [(item.frame, item.enabled) for item in seq] == [(1, True), (11, False)]

    def test_cvat_middle_outside_boundary_is_preserved_in_ls(self, sample_cvat_video_xml):
        cvat_sequence = load_cvat_from_xml_file(sample_cvat_video_xml)
        ls_tasks = _ls_tasks(cvat_sequence)
        car_result = next(result for result in ls_tasks[0].annotations[0].result if "car" in result.value.labels)

        assert [(item.frame, item.enabled) for item in car_result.value.sequence] == [
            (1, True),
            (3, False),
            (5, True),
        ]

    def test_cvat_terminal_keyframe_interpolates_to_end_when_size_known(self):
        xml_bytes = (
            '<?xml version="1.0" encoding="utf-8"?>'
            "<annotations><version>1.1</version><meta><task>"
            "<size>10</size><mode>interpolation</mode>"
            "<original_size><width>1920</width><height>1080</height></original_size>"
            '</task></meta><track id="0" label="person" source="manual">'
            '<box frame="5" outside="0" occluded="0" keyframe="1" '
            'xtl="100" ytl="100" xbr="200" ybr="200" z_order="0"/>'
            "</track></annotations>"
        ).encode("utf-8")

        cvat_sequence = load_cvat_from_xml_bytes(xml_bytes)
        ls_tasks = _ls_tasks(cvat_sequence)
        seq = ls_tasks[0].annotations[0].result[0].value.sequence

        assert [(item.frame, item.enabled) for item in seq] == [(6, True)]
        assert ls_tasks[0].annotations[0].result[0].value.framesCount == 10


class TestCrossFormatConversion:
    @pytest.fixture
    def mot_context(self) -> MOTContext:
        context = MOTContext(
            frame_rate=30,
            video_width=1920,
            video_height=1080,
        )
        context.categories.add("person", 1)
        context.categories.add("car", 2)
        return context

    @pytest.fixture
    def sample_cvat_video_xml(self) -> Path:
        return Path(__file__).parent.parent / "cvat_video" / "res" / "sample_video.xml"

    def test_cvat_to_mot(self, sample_cvat_video_xml, mot_context):
        cvat_sequence = load_cvat_from_xml_file(sample_cvat_video_xml)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
            output_path = Path(f.name)
            export_to_mot(cvat_sequence, mot_context, output_path)

            content = output_path.read_text()
            lines = [line for line in content.strip().split("\n") if line and not line.startswith("#")]

            assert len(lines) == 9
            for line in lines:
                assert len(line.split(",")) == 9

    def test_all_formats_same_annotation_count(self, sample_cvat_video_xml, mot_context):
        cvat_sequence = load_cvat_from_xml_file(sample_cvat_video_xml)
        ls_tasks = _ls_tasks(cvat_sequence)
        ls_sequence = ls_video_task_to_video_ir(ls_tasks[0])

        cvat_annotations = [ann for _, ann in flatten_sequence_with_track_ids(cvat_sequence)]
        assert len(ls_sequence.to_annotations()) == len(cvat_annotations)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as f:
            output_path = Path(f.name)

            export_to_mot(ls_sequence, mot_context, output_path)
            mot_sequence = load_mot_from_file(output_path, mot_context)
            mot_total = len(mot_sequence.to_annotations())

            assert mot_total >= len(cvat_annotations)


class TestLabelStudioVideoLocalProbeFallback:
    def test_task_to_ir_succeeds_without_dimensions_or_video_file(self, sample_ls_video_task_data):
        task_data = copy.deepcopy(sample_ls_video_task_data)
        for result in task_data["annotations"][0]["result"]:
            del result["original_width"]
            del result["original_height"]

        task = LabelStudioTask.model_validate(task_data)
        annotations = task.to_ir_annotations(filename="repo/remote/path/video.mp4")
        assert len(annotations) == 10
        assert all(ann.video_width is None for ann in annotations)
        assert all(ann.video_height is None for ann in annotations)

    def test_task_to_ir_uses_video_file_argument_for_probing(
        self,
        sample_ls_video_task_data,
        tmp_path,
        monkeypatch,
    ):
        task_data = copy.deepcopy(sample_ls_video_task_data)
        for result in task_data["annotations"][0]["result"]:
            del result["original_width"]
            del result["original_height"]

        local_video = tmp_path / "video.mp4"
        local_video.write_text("stub")

        monkeypatch.setattr(
            "dagshub_annotation_converter.formats.label_studio.task.get_video_dimensions",
            lambda _: (1920, 1080, 30.0),
        )

        task = LabelStudioTask.model_validate(task_data)
        annotations = task.to_ir_annotations(
            filename="repo/remote/path/video.mp4",
            video_file=str(local_video),
        )
        assert len(annotations) == 10
        assert all(ann.video_width == 1920 for ann in annotations)
        assert all(ann.video_height == 1080 for ann in annotations)
