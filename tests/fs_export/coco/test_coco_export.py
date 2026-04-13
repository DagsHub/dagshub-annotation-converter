import json

from dagshub_annotation_converter.converters.coco import export_to_coco_file
from dagshub_annotation_converter.ir.image import (
    CoordinateStyle,
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
    IRSegmentationPoint,
)


def test_coco_export(tmp_path):
    annotations = [
        IRBBoxImageAnnotation(
            filename="images/a.jpg",
            categories={"cat": 1.0},
            top=20.0,
            left=10.0,
            width=30.0,
            height=40.0,
            image_width=100,
            image_height=80,
            coordinate_style=CoordinateStyle.DENORMALIZED,
        ),
        IRSegmentationImageAnnotation(
            filename="images/b.jpg",
            categories={"dog": 1.0},
            image_width=50,
            image_height=40,
            coordinate_style=CoordinateStyle.DENORMALIZED,
            points=[
                IRSegmentationPoint(x=1.0, y=2.0),
                IRSegmentationPoint(x=4.0, y=2.0),
                IRSegmentationPoint(x=4.0, y=6.0),
                IRSegmentationPoint(x=1.0, y=6.0),
            ],
        ),
    ]

    output = tmp_path / "annotations.json"
    export_to_coco_file(annotations, output)

    payload = json.loads(output.read_text())
    assert len(payload["images"]) == 2
    assert len(payload["annotations"]) == 2
    assert {category["name"] for category in payload["categories"]} == {"cat", "dog"}
