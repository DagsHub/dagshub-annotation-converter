import pytest
from zipfile import ZipFile

from dagshub_annotation_converter.converters.cvat import (
    export_cvat_video_to_xml_string,
    export_cvat_video_to_zip,
    export_cvat_videos_to_zips,
)
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle


def _make_annotation(image_width: int, image_height: int) -> IRVideoBBoxAnnotation:
    return IRVideoBBoxAnnotation(
        track_id=1,
        frame_number=0,
        left=100,
        top=150,
        width=50,
        height=120,
        image_width=image_width,
        image_height=image_height,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )


class TestCVATVideoExport:
    def test_export_uses_probed_dimensions_when_missing(self, monkeypatch):
        ann = _make_annotation(image_width=0, image_height=0)
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.cvat.get_video_dimensions",
            lambda _: (1280, 720, 24.0),
        )

        xml_bytes = export_cvat_video_to_xml_string([ann], video_file="local_video.mp4")
        xml_text = xml_bytes.decode("utf-8")
        assert "<width>1280</width>" in xml_text
        assert "<height>720</height>" in xml_text

    def test_export_raises_without_dimensions_or_video_file(self):
        ann = _make_annotation(image_width=0, image_height=0)
        with pytest.raises(ValueError, match="Cannot determine frame dimensions for CVAT video export"):
            export_cvat_video_to_xml_string([ann])

    def test_export_keeps_explicit_dimensions(self):
        ann = _make_annotation(image_width=0, image_height=0)
        xml_bytes = export_cvat_video_to_xml_string([ann], image_width=1920, image_height=1080)
        xml_text = xml_bytes.decode("utf-8")
        assert "<width>1920</width>" in xml_text
        assert "<height>1080</height>" in xml_text

    def test_export_uses_annotation_filename_as_default_source(self):
        ann = _make_annotation(image_width=1920, image_height=1080)
        ann.filename = "earth-space-small.mp4"
        xml_bytes = export_cvat_video_to_xml_string([ann])
        xml_text = xml_bytes.decode("utf-8")
        assert "<source>earth-space-small.mp4</source>" in xml_text

    def test_export_raises_for_multiple_source_filenames(self):
        ann_a = _make_annotation(image_width=1920, image_height=1080)
        ann_a.filename = "earth-space-small.mp4"
        ann_b = _make_annotation(image_width=1920, image_height=1080)
        ann_b.filename = "jelly.mp4"
        with pytest.raises(ValueError, match="single source video per export"):
            export_cvat_video_to_xml_string([ann_a, ann_b])

    def test_zip_export_splits_multiple_videos(self, tmp_path):
        ann_a = _make_annotation(image_width=1920, image_height=1080)
        ann_a.track_id = 1
        ann_a.filename = "earth-space-small.mp4"
        ann_b = _make_annotation(image_width=1920, image_height=1080)
        ann_b.track_id = 2
        ann_b.filename = "jelly.mp4"

        output = tmp_path / "multi.zip"
        export_cvat_video_to_zip([ann_a, ann_b], output)

        with ZipFile(output) as z:
            names = sorted(z.namelist())
            assert "earth-space-small.mp4/annotations.xml" in names
            assert "jelly.mp4/annotations.xml" in names
            earth_xml = z.read("earth-space-small.mp4/annotations.xml").decode("utf-8")
            jelly_xml = z.read("jelly.mp4/annotations.xml").decode("utf-8")
            assert "<source>earth-space-small.mp4</source>" in earth_xml
            assert "<source>jelly.mp4</source>" in jelly_xml

    def test_zip_export_raises_when_video_file_given_for_multi_video(self, tmp_path):
        ann_a = _make_annotation(image_width=1920, image_height=1080)
        ann_a.filename = "earth-space-small.mp4"
        ann_b = _make_annotation(image_width=1920, image_height=1080)
        ann_b.filename = "jelly.mp4"
        with pytest.raises(ValueError, match="video_file is ambiguous"):
            export_cvat_video_to_zip([ann_a, ann_b], tmp_path / "multi.zip", video_file="local_video.mp4")

    def test_multi_export_writes_one_zip_per_video(self, tmp_path):
        ann_a = _make_annotation(image_width=1920, image_height=1080)
        ann_a.track_id = 1
        ann_a.filename = "earth-space-small.mp4"
        ann_b = _make_annotation(image_width=1920, image_height=1080)
        ann_b.track_id = 2
        ann_b.filename = "jelly.mp4"

        outputs = export_cvat_videos_to_zips([ann_a, ann_b], tmp_path)
        out_names = sorted([p.name for p in outputs])
        assert out_names == ["earth-space-small.zip", "jelly.zip"]

        with ZipFile(tmp_path / "earth-space-small.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<source>earth-space-small.mp4</source>" in xml
        with ZipFile(tmp_path / "jelly.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<source>jelly.mp4</source>" in xml
