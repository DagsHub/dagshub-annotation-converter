"""Roundtrip tests for video format conversions."""
import pytest
from pathlib import Path
import tempfile

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
        """MOT context for testing."""
        context = MOTContext(
            frame_rate=30.0,
            image_width=1920,
            image_height=1080,
        )
        context.categories = {1: "person", 2: "car"}
        return context

    @pytest.fixture
    def sample_mot_file(self) -> Path:
        """Path to sample MOT file."""
        return Path(__file__).parent.parent / "mot" / "res" / "gt" / "gt.txt"

    def test_mot_to_ls_video(self, sample_mot_file, mot_context):
        """Test converting MOT to Label Studio Video format."""
        # Load MOT annotations
        mot_annotations = load_mot_from_file(sample_mot_file, mot_context)
        
        # Flatten to list
        all_annotations = []
        for frame_anns in mot_annotations.values():
            all_annotations.extend(frame_anns)
        
        # Convert to Label Studio Video tasks
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        # Should have one task with 2 tracks
        assert len(ls_tasks) == 1
        task = ls_tasks[0]
        
        # Task should have annotations
        assert len(task.annotations) > 0
        
        # Should have 2 videorectangle results (one per track)
        results = task.annotations[0].result
        assert len(results) == 2

    def test_ls_video_to_mot(self, sample_ls_video_task_data, mot_context):
        """Test converting Label Studio Video to MOT format."""
        from dagshub_annotation_converter.formats.label_studio.videorectangle import VideoRectangleAnnotation
        
        # Parse LS video annotations
        ir_annotations = []
        for result in sample_ls_video_task_data["annotations"][0]["result"]:
            ls_ann = VideoRectangleAnnotation.model_validate(result)
            ir_annotations.extend(ls_ann.to_ir_annotations())
        
        # Export to MOT
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(ir_annotations, mot_context, output_path)
            
            # Verify output
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
        """Test full roundtrip: MOT -> LS Video -> MOT."""
        # Load original MOT
        original_mot = load_mot_from_file(sample_mot_file, mot_context)
        
        # Flatten
        original_annotations = []
        for frame_anns in original_mot.values():
            original_annotations.extend(frame_anns)
        
        # Convert to LS Video
        ls_tasks = video_ir_to_ls_video_tasks(original_annotations)
        
        # Convert back to IR
        reconstructed_annotations = ls_video_task_to_video_ir(ls_tasks[0])
        
        # Export to MOT
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(reconstructed_annotations, mot_context, output_path)
            
            # Re-import
            final_mot = load_mot_from_file(output_path, mot_context)
            
            # Same number of frames
            assert len(final_mot) == len(original_mot)
            
            # Same number of annotations per frame
            for frame_num in original_mot:
                assert len(final_mot[frame_num]) == len(original_mot[frame_num])
            
            # Check coordinate preservation (within tolerance due to normalization)
            for frame_num in original_mot:
                orig_frame = sorted(original_mot[frame_num], key=lambda a: a.track_id)
                final_frame = sorted(final_mot[frame_num], key=lambda a: a.track_id)
                
                for orig, final in zip(orig_frame, final_frame):
                    assert orig.track_id == final.track_id
                    # Allow 1 pixel tolerance due to float conversion
                    assert abs(orig.left - final.left) <= 1
                    assert abs(orig.top - final.top) <= 1
                    
        finally:
            output_path.unlink()


class TestCVATVideoToLabelStudioRoundtrip:
    """Tests for CVAT Video -> Label Studio Video conversion."""

    @pytest.fixture
    def sample_cvat_video_xml(self) -> Path:
        """Path to sample CVAT video XML."""
        return Path(__file__).parent.parent / "cvat_video" / "res" / "sample_video.xml"

    def test_cvat_to_ls_video(self, sample_cvat_video_xml):
        """Test converting CVAT Video to Label Studio Video format."""
        # Load CVAT video annotations
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        # Flatten to list
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        # Convert to Label Studio Video tasks
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        # Should have one task
        assert len(ls_tasks) == 1
        task = ls_tasks[0]
        
        # Should have 2 tracks
        results = task.annotations[0].result
        assert len(results) == 2

    def test_cvat_to_ls_preserves_categories(self, sample_cvat_video_xml):
        """Test that categories are preserved during conversion."""
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        # Check that labels are present
        task = ls_tasks[0]
        labels = set()
        for result in task.annotations[0].result:
            labels.update(result.value.labels)
        
        assert "person" in labels
        assert "car" in labels

    def test_cvat_to_ls_coordinate_conversion(self, sample_cvat_video_xml):
        """Test that CVAT pixel coords are converted to LS percentages."""
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        task = ls_tasks[0]
        
        # Find the person track result
        person_result = None
        for result in task.annotations[0].result:
            if "person" in result.value.labels:
                person_result = result
                break
        
        assert person_result is not None
        
        # First frame: CVAT coords (100, 150, 50x120) on 1920x1080
        # Should be: x=100/1920*100=5.208%, y=150/1080*100=13.889%
        first_seq = person_result.value.sequence[0]
        assert abs(first_seq.x - 5.208333) < 0.01
        assert abs(first_seq.y - 13.888889) < 0.01


class TestCrossFormatConversion:
    """Tests for conversion between all video formats."""

    @pytest.fixture
    def mot_context(self) -> MOTContext:
        """MOT context for testing."""
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
        """Test converting CVAT Video directly to MOT format."""
        # Load CVAT
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        # Flatten
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        # Export to MOT
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(all_annotations, mot_context, output_path)
            
            # Verify output
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
        """Test that all format conversions preserve annotation count."""
        # Load CVAT
        cvat_annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        cvat_total = sum(len(anns) for anns in cvat_annotations.values())
        
        # Convert to IR list
        all_annotations = []
        for frame_anns in cvat_annotations.values():
            all_annotations.extend(frame_anns)
        
        # Convert to LS Video
        ls_tasks = video_ir_to_ls_video_tasks(all_annotations)
        
        # Convert back to IR
        ls_annotations = ls_video_task_to_video_ir(ls_tasks[0])
        
        # Should have same count
        assert len(ls_annotations) == cvat_total
        
        # Export to MOT and reimport
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            output_path = Path(f.name)
        
        try:
            export_to_mot(ls_annotations, mot_context, output_path)
            mot_annotations = load_mot_from_file(output_path, mot_context)
            mot_total = sum(len(anns) for anns in mot_annotations.values())
            
            assert mot_total == cvat_total
            
        finally:
            output_path.unlink()
