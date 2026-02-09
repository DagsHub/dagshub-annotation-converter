from pathlib import Path

from dagshub_annotation_converter.converters.coco import load_coco_from_file
from dagshub_annotation_converter.ir.image import IRBBoxImageAnnotation, IRSegmentationImageAnnotation


def test_coco_import():
    annotation_file = Path(__file__).parents[2] / "coco" / "res" / "annotations.json"
    annotations, context = load_coco_from_file(annotation_file)

    assert context.categories == {1: "cat", 2: "dog"}
    assert set(annotations.keys()) == {"images/a.jpg", "images/b.jpg"}

    expected_types = {
        "images/a.jpg": [IRBBoxImageAnnotation, IRBBoxImageAnnotation, IRSegmentationImageAnnotation],
        "images/b.jpg": [IRSegmentationImageAnnotation, IRSegmentationImageAnnotation, IRSegmentationImageAnnotation],
    }
    actual_types = {filename: [type(ann) for ann in anns] for filename, anns in annotations.items()}

    assert actual_types == expected_types
