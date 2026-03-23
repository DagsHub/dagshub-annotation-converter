from typing import Optional
from zipfile import ZipFile

import pytest
from lxml import etree

from dagshub_annotation_converter.converters.cvat import (
    export_cvat_video_to_xml_bytes,
    export_cvat_video_to_zip,
    export_cvat_videos_to_zips,
)
from dagshub_annotation_converter.converters.label_studio_video import ls_video_task_to_video_ir
from dagshub_annotation_converter.formats.label_studio.task import AnnotationsContainer, LabelStudioTask
from dagshub_annotation_converter.formats.label_studio.videorectangle import (
    VideoRectangleAnnotation,
    VideoRectangleSequenceItem,
    VideoRectangleValue,
)
from dagshub_annotation_converter.ir.video import (
    CoordinateStyle,
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
    IRVideoSequence,
)


def _make_sequence(
    image_width: int,
    image_height: int,
    *,
    track_id: str = "1",
    filename: Optional[str] = None,
) -> IRVideoSequence:
    ann = IRVideoBBoxFrameAnnotation(
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
    track = IRVideoAnnotationTrack.from_annotations([ann], track_id=track_id)
    return IRVideoSequence(tracks=[track], filename=filename)


class TestCVATVideoExport:
    def test_export_uses_probed_dimensions_when_missing(self, monkeypatch):
        sequence = _make_sequence(image_width=0, image_height=0)
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.cvat.get_video_dimensions",
            lambda _: (1280, 720, 24.0),
        )
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.cvat.get_video_frame_count",
            lambda _: 100,
        )

        xml_bytes = export_cvat_video_to_xml_bytes(sequence, video_file="local_video.mp4")
        xml_text = xml_bytes.decode("utf-8")
        assert "<width>1280</width>" in xml_text
        assert "<height>720</height>" in xml_text

    def test_export_uses_probed_frame_count_when_seq_length_missing(self, monkeypatch):
        sequence = _make_sequence(image_width=1920, image_height=1080)
        monkeypatch.setattr(
            "dagshub_annotation_converter.converters.cvat.get_video_frame_count",
            lambda _: 400,
        )

        xml_bytes = export_cvat_video_to_xml_bytes(sequence, video_file="local_video.mp4")
        xml_text = xml_bytes.decode("utf-8")
        assert "<size>400</size>" in xml_text

    def test_export_raises_without_dimensions_or_video_file(self):
        sequence = _make_sequence(image_width=0, image_height=0)
        with pytest.raises(ValueError, match="Cannot determine frame dimensions for CVAT video export"):
            export_cvat_video_to_xml_bytes(sequence, video_name="video.mp4")

    def test_export_raises_without_video_reference(self):
        sequence = _make_sequence(image_width=1920, image_height=1080)
        with pytest.raises(ValueError, match="Cannot determine video name for CVAT video export"):
            export_cvat_video_to_xml_bytes(sequence)

    def test_export_keeps_explicit_dimensions(self):
        sequence = _make_sequence(image_width=0, image_height=0)
        xml_bytes = export_cvat_video_to_xml_bytes(
            sequence,
            video_name="video.mp4",
            image_width=1920,
            image_height=1080,
        )
        xml_text = xml_bytes.decode("utf-8")
        assert "<width>1920</width>" in xml_text
        assert "<height>1080</height>" in xml_text

    def test_export_uses_sequence_filename_as_default_source(self):
        sequence = _make_sequence(image_width=1920, image_height=1080, filename="earth-space-small.mp4")
        xml_bytes = export_cvat_video_to_xml_bytes(sequence)
        xml_text = xml_bytes.decode("utf-8")
        assert "<source>earth-space-small.mp4</source>" in xml_text

    def test_export_uses_video_file_name_as_default_source(self):
        sequence = _make_sequence(image_width=1920, image_height=1080)
        xml_bytes = export_cvat_video_to_xml_bytes(sequence, video_file="/tmp/earth-space-small.mp4")
        xml_text = xml_bytes.decode("utf-8")
        assert "<source>earth-space-small.mp4</source>" in xml_text

    def test_export_preserves_non_numeric_track_identifier(self):
        sequence = _make_sequence(image_width=1920, image_height=1080, track_id="track_person_1")
        xml_root = etree.fromstring(export_cvat_video_to_xml_bytes(sequence, video_name="video.mp4"))

        track_elem = xml_root.find("track")
        assert track_elem is not None
        assert track_elem.attrib["id"] == "track_person_1"

    def test_zip_export_writes_annotations_xml(self, tmp_path):
        sequence = _make_sequence(image_width=1920, image_height=1080, filename="earth-space-small.mp4")
        output = tmp_path / "single.zip"

        export_cvat_video_to_zip(sequence, output)

        with ZipFile(output) as z:
            assert z.namelist() == ["annotations.xml"]
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<source>earth-space-small.mp4</source>" in xml

    def test_multi_export_writes_one_zip_per_video(self, tmp_path):
        seq_a = _make_sequence(image_width=1920, image_height=1080, track_id="1", filename="earth-space-small.mp4")
        seq_b = _make_sequence(image_width=1920, image_height=1080, track_id="2", filename="jelly.mp4")

        outputs = export_cvat_videos_to_zips([seq_a, seq_b], tmp_path)
        out_names = sorted([p.name for p in outputs])
        assert out_names == ["earth-space-small.mp4.zip", "jelly.mp4.zip"]

        with ZipFile(tmp_path / "earth-space-small.mp4.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<source>earth-space-small.mp4</source>" in xml
        with ZipFile(tmp_path / "jelly.mp4.zip") as z:
            xml = z.read("annotations.xml").decode("utf-8")
            assert "<source>jelly.mp4</source>" in xml

    def test_multi_export_uses_video_files_for_missing_dimensions(self, tmp_path, monkeypatch):
        seq_a = _make_sequence(image_width=0, image_height=0, track_id="1", filename="earth-space-small.mp4")
        seq_b = _make_sequence(image_width=0, image_height=0, track_id="2", filename="jelly.mp4")

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
            [seq_a, seq_b],
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

    def test_multi_export_raises_without_sequence_filename(self, tmp_path):
        sequence = _make_sequence(image_width=1920, image_height=1080)
        with pytest.raises(ValueError, match="Each sequence must have sequence.filename set"):
            export_cvat_videos_to_zips([sequence], tmp_path)

    def test_export_ls_segments(self):
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
                    ),
                    VideoRectangleSequenceItem(
                        frame=9,
                        x=20.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                    ),
                    VideoRectangleSequenceItem(
                        frame=13,
                        x=30.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                    ),
                    VideoRectangleSequenceItem(
                        frame=14,
                        x=31.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
                    ),
                    VideoRectangleSequenceItem(
                        frame=15,
                        x=32.0,
                        y=10.0,
                        width=5.0,
                        height=10.0,
                        enabled=False,
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

        sequence = ls_video_task_to_video_ir(task)
        xml_bytes = export_cvat_video_to_xml_bytes(sequence)

        root = etree.fromstring(xml_bytes)
        assert int(root.findtext(".//meta/task/size")) == 200

        boxes = root.findall(".//track/box")
        by_frame = {int(box.attrib["frame"]): box for box in boxes}

        assert by_frame[8].attrib["outside"] == "0"
        assert by_frame[9].attrib["outside"] == "1"
        assert by_frame[12].attrib["outside"] == "0"
        assert by_frame[13].attrib["outside"] == "0"
        assert by_frame[14].attrib["outside"] == "0"
        assert by_frame[15].attrib["outside"] == "1"
        assert 16 not in by_frame
