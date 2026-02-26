import pytest
from zipfile import ZipFile
from lxml import etree

from dagshub_annotation_converter.converters.cvat import (
    export_cvat_video_to_xml_string,
    export_cvat_video_to_zip,
    export_cvat_videos_to_zips,
)
from dagshub_annotation_converter.converters.label_studio_video import ls_video_task_to_video_ir
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask, AnnotationsContainer
from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
    VideoRectangleValue,
    VideoRectangleSequenceItem,
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
        video_width=image_width,
        video_height=image_height,
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
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.cvat.get_video_frame_count",
            lambda _: 100,
        )

        xml_bytes = export_cvat_video_to_xml_string([ann], video_file="local_video.mp4")
        xml_text = xml_bytes.decode("utf-8")
        assert "<width>1280</width>" in xml_text
        assert "<height>720</height>" in xml_text

    def test_export_uses_probed_frame_count_when_seq_length_missing(self, monkeypatch):
        ann = _make_annotation(image_width=1920, image_height=1080)
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.cvat.get_video_frame_count",
            lambda _: 400,
        )

        xml_bytes = export_cvat_video_to_xml_string([ann], video_file="local_video.mp4")
        xml_text = xml_bytes.decode("utf-8")
        assert "<size>400</size>" in xml_text

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
        assert out_names == ["earth-space-small.mp4.zip", "jelly.mp4.zip"]

        with ZipFile(tmp_path / "earth-space-small.mp4.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<source>earth-space-small.mp4</source>" in xml
        with ZipFile(tmp_path / "jelly.mp4.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<source>jelly.mp4</source>" in xml

    def test_multi_export_uses_video_files_for_missing_dimensions(self, tmp_path, monkeypatch):
        ann_a = _make_annotation(image_width=0, image_height=0)
        ann_a.track_id = 1
        ann_a.filename = "earth-space-small.mp4"
        ann_b = _make_annotation(image_width=0, image_height=0)
        ann_b.track_id = 2
        ann_b.filename = "jelly.mp4"

        def fake_dimensions(path):
            name = path.name
            if name == "earth-space-small.mp4":
                return 1920, 1080, 24.0
            if name == "jelly.mp4":
                return 640, 360, 25.0
            raise AssertionError(f"Unexpected probe path: {path}")

        monkeypatch.setattr("dagshub_annotation_converter.converters.cvat.get_video_dimensions", fake_dimensions)
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.cvat.get_video_frame_count",
            lambda _: 100,
        )

        outputs = export_cvat_videos_to_zips(
            [ann_a, ann_b],
            tmp_path,
            video_files={
                "earth-space-small": "/tmp/earth-space-small.mp4",
                "assets/jelly.mp4": "/tmp/jelly.mp4",
            },
        )
        out_names = sorted([p.name for p in outputs])
        assert out_names == ["earth-space-small.mp4.zip", "jelly.mp4.zip"]

        with ZipFile(tmp_path / "earth-space-small.mp4.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<width>1920</width>" in xml
            assert "<height>1080</height>" in xml
            assert "<source>earth-space-small.mp4</source>" in xml

        with ZipFile(tmp_path / "jelly.mp4.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<width>640</width>" in xml
            assert "<height>360</height>" in xml
            assert "<source>jelly.mp4</source>" in xml

    def test_export_ls_segments_with_string_outside_false(self):
        video_rect = VideoRectangleAnnotation(
            id="track_1",
            original_width=1920,
            original_height=1080,
            value=VideoRectangleValue(
                sequence=[
                    VideoRectangleSequenceItem(
                        frame=1,
                        x=10.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=True,
                        outside="false",
                    ),
                    VideoRectangleSequenceItem(
                        frame=9,
                        x=20.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                        outside="false",
                    ),
                    VideoRectangleSequenceItem(
                        frame=13,
                        x=30.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                        outside="false",
                    ),
                    VideoRectangleSequenceItem(
                        frame=14,
                        x=31.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                        outside="false",
                    ),
                    VideoRectangleSequenceItem(
                        frame=15,
                        x=32.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                        outside="false",
                    ),
                ],
                labels=["person"],
                framesCount=200,
            ),
        )
        task = LabelStudioTask(data={"video": "/data/video.mp4"})
        task.annotations = [
            AnnotationsContainer.model_construct(completed_by=None, result=[video_rect], ground_truth=False),
        ]

        annotations = ls_video_task_to_video_ir(task)
        xml_bytes = export_cvat_video_to_xml_string(annotations)

        root = etree.fromstring(xml_bytes)
        assert int(root.findtext(".//meta/task/size")) == 200

        boxes = root.findall(".//track/box")
        by_frame = {int(box.attrib["frame"]): box for box in boxes}

        assert by_frame[8].attrib["outside"] == "0"
        assert by_frame[9].attrib["outside"] == "1"
        assert by_frame[12].attrib["outside"] == "0"
        assert by_frame[13].attrib["outside"] == "0"
        assert by_frame[14].attrib["outside"] == "0"
        assert 15 not in by_frame
