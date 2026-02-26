from lxml import etree
import shutil

from dagshub_annotation_converter.ir.video import CoordinateStyle
from dagshub_annotation_converter.formats.cvat.video import (
    parse_video_track,
    export_video_track_to_xml,
)
from dagshub_annotation_converter.converters.cvat import load_cvat_from_xml_file, load_cvat_from_fs


class TestCVATVideoTrackParsing:
    def test_parse_track_basic(self, sample_cvat_video_xml):
        tree = etree.parse(str(sample_cvat_video_xml))
        tracks = tree.findall(".//track")
        
        annotations = parse_video_track(tracks[0], image_width=1920, image_height=1080)
        
        assert len(annotations) == 5
        
        first_ann = annotations[0]
        assert first_ann.track_id == 0
        assert first_ann.frame_number == 0
        assert first_ann.left == 100
        assert first_ann.top == 150
        assert first_ann.width == 50
        assert first_ann.height == 120
        assert "person" in first_ann.categories

    def test_parse_track_with_occlusion(self, sample_cvat_video_xml):
        tree = etree.parse(str(sample_cvat_video_xml))
        tracks = tree.findall(".//track")
        
        annotations = parse_video_track(tracks[0], image_width=1920, image_height=1080)
        
        frame_2_ann = [a for a in annotations if a.frame_number == 2][0]
        assert frame_2_ann.visibility == 0.5
        assert not frame_2_ann.meta.get("outside")

    def test_parse_track_with_outside(self, sample_cvat_video_xml):
        tree = etree.parse(str(sample_cvat_video_xml))
        tracks = tree.findall(".//track")
        
        annotations = parse_video_track(tracks[1], image_width=1920, image_height=1080)
        
        assert len(annotations) == 5
        
        frame_3_ann = [a for a in annotations if a.frame_number == 3][0]
        assert frame_3_ann.visibility == 0.0
        assert frame_3_ann.meta.get("outside")
        
        for ann in annotations:
            if ann.frame_number != 3:
                assert not ann.meta.get("outside")
                assert ann.visibility > 0.0

    def test_parse_track_keyframe_metadata(self, sample_cvat_video_xml):
        tree = etree.parse(str(sample_cvat_video_xml))
        tracks = tree.findall(".//track")
        
        annotations = parse_video_track(tracks[0], image_width=1920, image_height=1080)
        
        # Frame 0 should be keyframe
        frame_0_ann = [a for a in annotations if a.frame_number == 0][0]
        assert frame_0_ann.keyframe
        
        # Frame 1 should not be keyframe
        frame_1_ann = [a for a in annotations if a.frame_number == 1][0]
        assert not frame_1_ann.keyframe


class TestCVATVideoOutsideRoundtrip:
    def test_outside_roundtrip(self):
        """Test that outside=1 survives import -> export -> import."""
        xml_str = b"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <size>5</size>
      <mode>interpolation</mode>
      <original_size><width>1920</width><height>1080</height></original_size>
    </task>
  </meta>
  <track id="0" label="dog" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1"
         xtl="10" ytl="20" xbr="110" ybr="120" z_order="0"/>
    <box frame="1" outside="0" occluded="1" keyframe="0"
         xtl="12" ytl="22" xbr="112" ybr="122" z_order="0"/>
    <box frame="2" outside="1" occluded="0" keyframe="1"
         xtl="14" ytl="24" xbr="114" ybr="124" z_order="0"/>
    <box frame="3" outside="0" occluded="0" keyframe="1"
         xtl="16" ytl="26" xbr="116" ybr="126" z_order="0"/>
  </track>
</annotations>"""
        root = etree.fromstring(xml_str)
        track = root.findall(".//track")[0]
        annotations = parse_video_track(track, image_width=1920, image_height=1080)
        
        assert len(annotations) == 4
        assert annotations[2].visibility == 0.0
        assert annotations[2].meta["outside"]
        assert annotations[1].visibility == 0.5  # occluded, not outside
        assert not annotations[1].meta["outside"]
        
        exported_track = export_video_track_to_xml(0, annotations)
        boxes = exported_track.findall("box")
        
        assert len(boxes) == 4
        assert boxes[0].attrib["outside"] == "0"
        assert boxes[0].attrib["keyframe"] == "1"
        assert boxes[1].attrib["outside"] == "0"
        assert boxes[1].attrib["occluded"] == "1"
        assert boxes[1].attrib["keyframe"] == "0"
        assert boxes[2].attrib["outside"] == "1"
        assert boxes[2].attrib["occluded"] == "0"  # outside takes priority
        assert boxes[2].attrib["keyframe"] == "1"
        assert boxes[3].attrib["outside"] == "0"
        assert boxes[3].attrib["keyframe"] == "1"
        
        reimported = parse_video_track(exported_track, image_width=1920, image_height=1080)
        assert len(reimported) == 4
        for orig, re in zip(annotations, reimported):
            assert orig.frame_number == re.frame_number
            assert orig.visibility == re.visibility
            assert orig.meta.get("outside") == re.meta.get("outside")


class TestCVATVideoFileImport:
    def test_load_from_xml_file(self, sample_cvat_video_xml):
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        assert len(annotations) > 0
        
        # Frame 0 should have 2 annotations (person and car tracks)
        frame_0_anns = annotations.get(0, [])
        assert len(frame_0_anns) == 2

    def test_load_extracts_image_dimensions(self, sample_cvat_video_xml):
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        # All annotations should have correct image dimensions
        for frame_anns in annotations.values():
            for ann in frame_anns:
                assert ann.video_width == 1920
                assert ann.video_height == 1080

    def test_track_ids_consistent(self, sample_cvat_video_xml):
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        all_track_ids = set()
        for frame_anns in annotations.values():
            for ann in frame_anns:
                all_track_ids.add(ann.track_id)
        
        # Should have exactly 2 tracks
        assert len(all_track_ids) == 2
        assert all_track_ids == {0, 1}

    def test_categories_from_labels(self, sample_cvat_video_xml):
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        categories_by_track = {}
        for frame_anns in annotations.values():
            for ann in frame_anns:
                if ann.track_id not in categories_by_track:
                    categories_by_track[ann.track_id] = ann.ensure_has_one_category()
        
        # Track 0 should be "person", Track 1 should be "car"
        assert categories_by_track[0] == "person"
        assert categories_by_track[1] == "car"

    def test_coordinate_style_denormalized(self, sample_cvat_video_xml):
        annotations = load_cvat_from_xml_file(sample_cvat_video_xml)
        
        for frame_anns in annotations.values():
            for ann in frame_anns:
                assert ann.coordinate_style == CoordinateStyle.DENORMALIZED

    def test_load_from_fs_multiple_files(self, sample_cvat_video_xml, tmp_path):
        shutil.copy(sample_cvat_video_xml, tmp_path / "a.xml")
        shutil.copy(sample_cvat_video_xml, tmp_path / "b.xml")

        loaded = load_cvat_from_fs(tmp_path)
        assert set(loaded.keys()) == {"a.xml", "b.xml"}
        assert all(len(anns) > 0 for anns in loaded.values())
