from pathlib import Path

from dagshub_annotation_converter.converters.cvat import parse_cvat_from_xml_file
from dagshub_annotation_converter.ir.image import (
    IRBBoxAnnotation,
    IRSegmentationAnnotation,
    IRPoseAnnotation,
)


def test_cvat_import():
    annotation_file = Path(__file__).parent / "annotations.xml"
    ctx, annotations = parse_cvat_from_xml_file(annotation_file)

    category_names = ["Baby Yoda", "Ship", "Person", "Robot", "Yoda"]
    assert [cat.name for cat in ctx.categories.categories] == category_names

    expected_files = ["001.png", "002.png", "003.png", "004.png"]
    assert list(annotations.keys()) == list(expected_files)

    # Check only the annotation types, but not the annotations themselves (otherwise the parsing tests would fail)
    expected_annotations = [
        [IRBBoxAnnotation],
        [
            IRSegmentationAnnotation,
            IRBBoxAnnotation,
            IRSegmentationAnnotation,
            IRSegmentationAnnotation,
            IRBBoxAnnotation,
            IRSegmentationAnnotation,
        ],
        [IRBBoxAnnotation, IRPoseAnnotation],
        [IRPoseAnnotation],
    ]

    actual_annotations = [[type(ann) for ann in annotations[file]] for file in expected_files]

    assert expected_annotations == actual_annotations
