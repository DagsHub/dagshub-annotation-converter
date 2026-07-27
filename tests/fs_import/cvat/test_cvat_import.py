from pathlib import Path

from dagshub_annotation_converter.converters.cvat import load_cvat_from_xml_file
from dagshub_annotation_converter.formats.label_studio.task import (
    LabelStudioTask,
    parse_ls_task,
)
from dagshub_annotation_converter.formats.yolo.context import YoloContext
from dagshub_annotation_converter.formats.yolo.pose import export_pose
from dagshub_annotation_converter.ir.image import (
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
    IRPoseImageAnnotation,
)


def test_cvat_import():
    annotation_file = Path(__file__).parent / "annotations.xml"
    annotations = load_cvat_from_xml_file(annotation_file)

    expected_files = ["001.png", "002.png", "003.png", "004.png"]
    assert list(annotations.keys()) == list(expected_files)

    # Check only the annotation types, but not the annotations themselves (otherwise the parsing tests would fail)
    expected_annotations = [
        [IRBBoxImageAnnotation],
        [
            IRSegmentationImageAnnotation,
            IRBBoxImageAnnotation,
            IRSegmentationImageAnnotation,
            IRSegmentationImageAnnotation,
            IRBBoxImageAnnotation,
            IRSegmentationImageAnnotation,
        ],
        [IRBBoxImageAnnotation, IRPoseImageAnnotation],
        [IRPoseImageAnnotation],
    ]

    actual_annotations = [[type(ann) for ann in annotations[file]] for file in expected_files]

    assert expected_annotations == actual_annotations


def test_cvat_image_export_with_interpolation_metadata_imports_as_file_mapping(tmp_path):
    annotation_file = tmp_path / "annotations.xml"
    annotation_file.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta><task><mode>interpolation</mode></task></meta>
  <image id="0" name="frames/000001.jpg" width="1920" height="1080" />
</annotations>
"""
    )

    annotations = load_cvat_from_xml_file(annotation_file)

    # The Data Engine importer consumes image annotations via Mapping.items().
    assert isinstance(annotations, dict)
    assert list(annotations.items()) == [("frames/000001.jpg", [])]


def test_cvat_occluded_pose_point_survives_label_studio_to_yolo():
    annotation_file = Path(__file__).parent / "annotations.xml"
    annotations = load_cvat_from_xml_file(annotation_file)

    assert isinstance(annotations, dict)
    [cvat_pose] = annotations["004.png"]
    assert isinstance(cvat_pose, IRPoseImageAnnotation)
    assert cvat_pose.points[0].visible is False

    label_studio_task = LabelStudioTask.from_ir_annotations([cvat_pose])
    parsed_task = parse_ls_task(label_studio_task.model_dump_json())
    [round_tripped_pose] = parsed_task.to_ir_annotations()

    assert isinstance(round_tripped_pose, IRPoseImageAnnotation)
    assert round_tripped_pose.points[0].visible is False

    yolo_context = YoloContext(annotation_type="pose")
    yolo_context.categories.add(name="Yoda")
    yolo_annotation = export_pose(round_tripped_pose, yolo_context)
    visibility_values = yolo_annotation.split()[7::3]

    assert visibility_values == ["0", "1", "1", "1", "1", "1", "1"]
