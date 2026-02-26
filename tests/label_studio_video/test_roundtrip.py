import pytest
from pathlib import Path
import tempfile
import math
import copy

from dagshub_annotation_converter.formats.mot import MOTContext
from dagshub_annotation_converter.converters.mot import load_mot_from_file, export_to_mot
from dagshub_annotation_converter.converters.cvat import (
    load_cvat_from_xml_file,
    load_cvat_from_xml_string,
    export_cvat_video_to_xml_string,
)
from dagshub_annotation_converter.converters.label_studio_video import (
    video_ir_to_ls_video_tasks,
    ls_video_task_to_video_ir,
)
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask
from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
    VideoRectangleValue,
    VideoRectangleSequenceItem,
)


class TestMOTToLabelStudioRoundtrip:
    """Tests for MOT <-> Label Studio Video conversion.
    
    Uses CVAT MOT 1.1 format (9 columns):
    frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    """

    @pytest.fixture
    def mot_context(self) -> MOTContext:
        context = MOTContext(
            frame_rate=30.0,
            image_width=1920,
            image_height=1080,
        )
        context.categories = {1: "person", 2: "car"}
        return context

    @pytest.fixture
    def sample_mot_file(self) -> Path:
        return Path(__file__).parent.parent / "mot" / "res" / "gt" / "gt.txt"

    def test_mot_to_ls_video(self, sample_mot_file, mot_context):
        mot_annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        all_annotations = []
        for frame_anns in mot_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        assert len(ls_tasks) == 1
        task = ls_tasks[0]
        
        assert len(task.annotations) > 0
        
        # Should have 2 videorectangle results (one per track)
        results = task.annotations[0].result
        assert len(results) == 2

    def test_mot_to_ls_treats_frames_as_independent(self, sample_mot_file, mot_context):
        mot_annotations = load_mot_from_file(sample_mot_file, mot_context)
        all_annotations = [ann for frame_anns in mot_annotations.values() for ann in frame_anns]

        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        results = ls_tasks[0].annotations[0].result
        for result in results:
            assert all(item.enabled is False for item in result.value.sequence)

    def test_ls_video_to_mot(self, sample_ls_video_task_data, mot_context):
        from dagshub_annotation_converter.formats.label_studio.videorectangle import VideoRectangleAnnotation
        
        ir_annotations = []
        for result in sample_ls_video_task_data["annotations"][0]["result"]:
            ls_ann = VideoRectangleAnnotation.model_validate(result)
            ir_annotations.extend(ls_ann.to_ir_annotations())
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(ir_annotations, mot_context, output_path)
            
            content = output_path.read_text()
            lines = [line for line in content.strip().split("\n") if line and not line.startswith("#")]
            
            # Should have 10 lines (5 frames x 2 tracks)
            assert len(lines) == 10
            
            # Each line should have 9 columns (CVAT MOT 1.1 format)
            for line in lines:
                parts = line.split(",")
                assert len(parts) == 9
            
        finally:
            output_path.unlink()

    def test_mot_ls_mot_roundtrip(self, sample_mot_file, mot_context):
        original_mot = load_mot_from_file(sample_mot_file, mot_context)
        
        original_annotations = []
        for frame_anns in original_mot.values():
            original_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(original_annotations)
        
        reconstructed_annotations = ls_video_task_to_video_ir(ls_tasks[0])
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(reconstructed_annotations, mot_context, output_path)
            
            final_mot = load_mot_from_file(output_path, mot_context)
            
            assert len(final_mot) == len(original_mot)
            
            for frame_num in original_mot:
                assert len(final_mot[frame_num]) == len(original_mot[frame_num])
            
            # Check coordinate preservation (within tolerance due to normalization)
            for frame_num in original_mot:
                orig_frame = sorted(original_mot[frame_num], key=lambda a: a.track_id)
                final_frame = sorted(final_mot[frame_num], key=lambda a: a.track_id)
                
                for orig, final in zip(orig_frame, final_frame):
                    assert orig.track_id == final.track_id
                    # Allow 1 pixel tolerance due to float conversion
                    assert math.isclose(orig.left, final.left, abs_tol=1)
                    assert math.isclose(orig.top, final.top, abs_tol=1)
                    
        finally:
            output_path.unlink()

    def test_mot_ls_roundtrip_preserves_track_frames_and_keyframes(self, sample_mot_file, mot_context):
        original_mot = load_mot_from_file(sample_mot_file, mot_context)
        original_annotations = [ann for frame in original_mot.values() for ann in frame]

        ls_tasks = video_ir_to_ls_video_tasks(original_annotations)
        task = ls_tasks[0]
        for result in task.annotations[0].result:
            for item in result.value.sequence:
                assert isinstance(item.enabled, bool)

        reconstructed_annotations = ls_video_task_to_video_ir(task)

        original_by_track_frame = {
            (ann.track_id, ann.frame_number): ann
            for ann in original_annotations
        }
        reconstructed_by_track_frame = {
            (ann.track_id, ann.frame_number): ann
            for ann in reconstructed_annotations
        }

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
            meta={"original_track_id": 1},
        )
        ir_annotations = ls_ann.to_ir_annotations()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        try:
            export_to_mot(ir_annotations, mot_context, output_path)
            mot_lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]
            mot_frames = [int(line.split(",")[0]) for line in mot_lines]

            assert 11 in mot_frames  # hidden boundary marker
            assert 100 in mot_frames
            assert all(frame <= 11 or frame >= 100 for frame in mot_frames)

            mot_annotations = load_mot_from_file(output_path, mot_context)
            all_anns = [ann for frame_anns in mot_annotations.values() for ann in frame_anns]
            ls_tasks = video_ir_to_ls_video_tasks(all_anns)
            seq = ls_tasks[0].annotations[0].result[0].value.sequence
            by_frame = {item.frame: item for item in seq}

            assert by_frame[11].enabled is False
            assert by_frame[100].enabled is False
        finally:
            output_path.unlink()

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
            meta={"original_track_id": 1},
        )
        ir_annotations = ls_ann.to_ir_annotations()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        try:
            export_to_mot(ir_annotations, mot_context, output_path)
            mot_lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]
            mot_frames = [int(line.split(",")[0]) for line in mot_lines]
            assert mot_frames == list(range(1, 12))

            last_visibility = float(mot_lines[-1].split(",")[8])
            assert last_visibility == 1.0

            mot_annotations = load_mot_from_file(output_path, mot_context)
            all_anns = [ann for frame_anns in mot_annotations.values() for ann in frame_anns]
            ls_tasks = video_ir_to_ls_video_tasks(all_anns)
            seq = ls_tasks[0].annotations[0].result[0].value.sequence

            assert seq[-1].frame == 11
            assert seq[-1].enabled is False
        finally:
            output_path.unlink()

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
            meta={"original_track_id": 1},
        )
        ir_annotations = ls_ann.to_ir_annotations()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        try:
            export_to_mot(ir_annotations, mot_context, output_path)
            mot_lines = [line for line in output_path.read_text().splitlines() if line and not line.startswith("#")]
            mot_frames = [int(line.split(",")[0]) for line in mot_lines]

            # Second segment should be dense from LS frame 100 through LS frame 120.
            assert 100 in mot_frames
            assert 120 in mot_frames
            assert all(frame <= 11 or frame >= 100 for frame in mot_frames)

            mot_annotations = load_mot_from_file(output_path, mot_context)
            all_anns = [ann for frame_anns in mot_annotations.values() for ann in frame_anns]
            ls_tasks = video_ir_to_ls_video_tasks(all_anns)
            seq = ls_tasks[0].annotations[0].result[0].value.sequence
            seq_frames = [item.frame for item in seq]

            assert 100 in seq_frames
            assert 120 in seq_frames
            assert seq_frames[-1] == 120
        finally:
            output_path.unlink()


class TestCVATVideoToLabelStudioRoundtrip:
    @pytest.fixture
    def sample_cvat_video_xml(self) -> Path:
        return Path(__file__).parent.parent / "cvat_video" / "res" / "sample_video.xml"

    def test_cvat_to_ls_video(self, sample_cvat_video_xml):
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        assert len(ls_tasks) == 1
        task = ls_tasks[0]
        
        results = task.annotations[0].result
        assert len(results) == 2

    def test_cvat_to_ls_preserves_categories(self, sample_cvat_video_xml):
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        task = ls_tasks[0]
        labels = set()
        for result in task.annotations[0].result:
            labels.update(result.value.labels)
        
        assert "person" in labels
        assert "car" in labels

    def test_cvat_to_ls_coordinate_conversion(self, sample_cvat_video_xml, epsilon):
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        task = ls_tasks[0]
        

        person_result = None
        for result in task.annotations[0].result:
            if "person" in result.value.labels:
                person_result = result
                break
        
        assert person_result is not None
        
        # CVAT coords (100, 150, 50x120) on 1920x1080 -> x=100/1920*100, y=150/1080*100
        first_seq = person_result.value.sequence[0]
        assert math.isclose(first_seq.x, 5.208333, abs_tol=epsilon)
        assert math.isclose(first_seq.y, 13.888889, abs_tol=epsilon)

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
            meta={"original_track_id": 1},
        )
        ir_annotations = ls_ann.to_ir_annotations()

        xml_bytes = export_cvat_video_to_xml_string(ir_annotations)
        cvat_annotations = load_cvat_from_xml_string(xml_bytes)
        all_annotations = [ann for frame_anns in cvat_annotations.values() for ann in frame_anns]
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)

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


class TestCrossFormatConversion:
    @pytest.fixture
    def mot_context(self) -> MOTContext:
        context = MOTContext(
            frame_rate=30.0,
            image_width=1920,
            image_height=1080,
        )
        context.categories = {1: "person", 2: "car"}
        return context

    @pytest.fixture
    def sample_cvat_video_xml(self) -> Path:
        return Path(__file__).parent.parent / "cvat_video" / "res" / "sample_video.xml"

    def test_cvat_to_mot(self, sample_cvat_video_xml, mot_context):
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(all_annotations, mot_context, output_path)
            
            content = output_path.read_text()
            lines = [line for line in content.strip().split("\n") if line and not line.startswith("#")]
            
            # Outside keyframes are kept as explicit visibility=0 boundary markers.
            assert len(lines) == 10
            
            # Verify each line has 9 columns (CVAT MOT 1.1 format)
            for line in lines:
                parts = line.split(",")
                assert len(parts) == 9
            
        finally:
            output_path.unlink()

    def test_all_formats_same_annotation_count(self, sample_cvat_video_xml, mot_context):
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        cvat_visible_total = sum(
            1 for anns in cvat_annotations.values() for ann in anns if not ann.meta.get("outside", False)
        )
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        ls_annotations = ls_video_task_to_video_ir(ls_tasks[0])
        
        assert len(ls_annotations) == cvat_visible_total
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(ls_annotations, mot_context, output_path)
            mot_annotations = load_mot_from_file(output_path, mot_context)
            mot_total = sum(len(anns) for anns in mot_annotations.values())

            assert mot_total == cvat_visible_total
            
        finally:
            output_path.unlink()


class TestLabelStudioVideoLocalProbeFallback:
    def test_task_to_ir_succeeds_without_dimensions_or_video_file(self, sample_ls_video_task_data):
        task_data = copy.deepcopy(sample_ls_video_task_data)
        for result in task_data["annotations"][0]["result"]:
            result.pop("original_width", None)
            result.pop("original_height", None)

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
            result.pop("original_width", None)
            result.pop("original_height", None)

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
