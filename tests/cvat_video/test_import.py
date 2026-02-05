"""Tests for CVAT Video format import."""
import pytest
from pathlib import Path

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.formats.cvat.video import parse_video_track
from dagshub_annotation_converter.converters.cvat import load_cvat_from_xml_file


class TestCVATVideoTrackParsing:
    """Tests for parsing CVAT video track elements."""

    def test_parse_track_basic(self, sample_cvat_video_xml):
        """Test parsing a basic CVAT video track."""
        from lxml import etree
        
        tree = etree.parse(str(sample_cvat_video_xml))
        tracks = tree.findall(".//track")
        
        # Parse first track (person)
        annotations = parse_video_track(tracks[0], image_width=1920, image_height=1080)
        
        assert len(annotations) == 5  # 5 frames
        
        # Verify first annotation
        first_ann = annotations[0]
        assert first_ann.track_id == 0
        assert first_ann.frame_number == 0
        assert first_ann.left == 100
        assert first_ann.top == 150
        assert first_ann.width == 50  # xbr - xtl = 150 - 100
        assert first_ann.height == 120  # ybr - ytl = 270 - 150
        assert "person" in first_ann.categories

    def test_parse_track_with_occlusion(self, sample_cvat_video_xml):
        """Test parsing track with occluded frames."""
        from lxml import etree
        
        tree = etree.parse(str(sample_cvat_video_xml))
        tracks = tree.findall(".//track")
        
        # Parse first track (person) - frame 2 is occluded
        annotations = parse_video_track(tracks[0], image_width=1920, image_height=1080)
        
        # Frame 2 (index 2) should have visibility < 1
        frame_2_ann = [a for a in annotations if a.frame_number == 2][0]
        assert frame_2_ann.visibility < 1.0

    def test_parse_track_keyframe_metadata(self, sample_cvat_video_xml):
        """Test that keyframe info is preserved in metadata."""
        from lxml import etree
        
        tree = etree.parse(str(sample_cvat_video_xml))
        tracks = tree.findall(".//track")
        
        annotations = parse_video_track(tracks[0], image_width=1920, image_height=1080)
        
        # Frame 0 should be keyframe
        frame_0_ann = [a for a in annotations if a.frame_number == 0][0]
        assert frame_0_ann.meta.get("keyframe") == True
        
        # Frame 1 should not be keyframe
        frame_1_ann = [a for a in annotations if a.frame_number == 1][0]
        assert frame_1_ann.meta.get("keyframe") == False


class TestCVATVideoFileImport:
    """Tests for importing full CVAT video XML files."""

    def test_load_from_xml_file(self, sample_cvat_video_xml):
        """Test loading annotations from a CVAT video XML file."""
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        # Should have annotations grouped by frame (video mode detected)
        assert len(annotations) > 0
        
        # Frame 0 should have 2 annotations (person and car tracks)
        frame_0_anns = annotations.get(0, [])
        assert len(frame_0_anns) == 2

    def test_load_extracts_image_dimensions(self, sample_cvat_video_xml):
        """Test that image dimensions are extracted from meta."""
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        # All annotations should have correct image dimensions
        for frame_anns in annotations.values():
            for ann in frame_anns:
                assert ann.image_width == 1920
                assert ann.image_height == 1080

    def test_track_ids_consistent(self, sample_cvat_video_xml):
        """Test that track IDs are consistent across frames."""
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        # Collect all track IDs
        all_track_ids = set()
        for frame_anns in annotations.values():
            for ann in frame_anns:
                all_track_ids.add(ann.track_id)
        
        # Should have exactly 2 tracks
        assert len(all_track_ids) == 2
        assert all_track_ids == {0, 1}

    def test_categories_from_labels(self, sample_cvat_video_xml):
        """Test that categories are extracted from track labels."""
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        # Collect categories by track
        categories_by_track = {}
        for frame_anns in annotations.values():
            for ann in frame_anns:
                if ann.track_id not in categories_by_track:
                    categories_by_track[ann.track_id] = ann.ensure_has_one_category()
        
        # Track 0 should be "person", Track 1 should be "car"
        assert categories_by_track[0] == "person"
        assert categories_by_track[1] == "car"

    def test_coordinate_style_denormalized(self, sample_cvat_video_xml):
        """Test that CVAT imports as denormalized coordinates."""
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        for frame_anns in annotations.values():
            for ann in frame_anns:
                assert ann.coordinate_style == CoordinateStyle.DENORMALIZED
