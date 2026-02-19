import pytest
from pathlib import Path
import tempfile
import math

from dagshub_annotation_converter.formats.mot import MOTContext
from dagshub_annotation_converter.converters.mot import load_mot_from_file, export_to_mot
from dagshub_annotation_converter.converters.cvat import load_cvat_from_xml_file
from dagshub_annotation_converter.converters.label_studio_video import (
    video_ir_to_ls_video_tasks,
    ls_video_task_to_video_ir,
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
            
            # Should have 10 lines (5 frames x 2 tracks)
            assert len(lines) == 10
            
            # Verify each line has 9 columns (CVAT MOT 1.1 format)
            for line in lines:
                parts = line.split(",")
                assert len(parts) == 9
            
        finally:
            output_path.unlink()

    def test_all_formats_same_annotation_count(self, sample_cvat_video_xml, mot_context):
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        cvat_total = sum(len(anns) for anns in cvat_annotations.values())
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        ls_annotations = ls_video_task_to_video_ir(ls_tasks[0])
        
        assert len(ls_annotations) == cvat_total
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(ls_annotations, mot_context, output_path)
            mot_annotations = load_mot_from_file(output_path, mot_context)
            mot_total = sum(len(anns) for anns in mot_annotations.values())
            
            assert mot_total == cvat_total
            
        finally:
            output_path.unlink()
