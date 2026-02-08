from pathlib import Path

from dagshub_annotation_converter.converters.coco import load_coco_from_file
from dagshub_annotation_converter.ir.image import (
    CoordinateStyle,
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
    IRSegmentationPoint,
)


def test_coco_import():
    annotation_file = Path(__file__).parent / "res" / "annotations.json"
    annotations, context = load_coco_from_file(annotation_file)

    assert context.categories == {1: "cat", 2: "dog"}
    assert set(annotations.keys()) == {"images/a.jpg", "images/b.jpg"}

    assert annotations["images/a.jpg"] == [
        IRBBoxImageAnnotation(
            filename="images/a.jpg",
            imported_id="10",
            categories={"cat": 1.0},
            top=20.0,
            left=10.0,
            width=30.0,
            height=40.0,
            image_width=100,
            image_height=80,
            coordinate_style=CoordinateStyle.DENORMALIZED,
        ),
        IRBBoxImageAnnotation(
            filename="images/a.jpg",
            imported_id="11",
            categories={"dog": 1.0},
            top=10.0,
            left=5.0,
            width=10.0,
            height=20.0,
            image_width=100,
            image_height=80,
            coordinate_style=CoordinateStyle.DENORMALIZED,
        ),
        IRSegmentationImageAnnotation(
            filename="images/a.jpg",
            imported_id="11",
            categories={"dog": 1.0},
            image_width=100,
            image_height=80,
            coordinate_style=CoordinateStyle.DENORMALIZED,
            points=[
                IRSegmentationPoint(x=5.0, y=10.0),
                IRSegmentationPoint(x=15.0, y=10.0),
                IRSegmentationPoint(x=15.0, y=30.0),
                IRSegmentationPoint(x=5.0, y=30.0),
            ],
        ),
    ]

    assert annotations["images/b.jpg"] == [
        IRSegmentationImageAnnotation(
            filename="images/b.jpg",
            imported_id="12",
            categories={"cat": 1.0},
            image_width=50,
            image_height=40,
            coordinate_style=CoordinateStyle.DENORMALIZED,
            points=[
                IRSegmentationPoint(x=1.0, y=2.0),
                IRSegmentationPoint(x=4.0, y=2.0),
                IRSegmentationPoint(x=4.0, y=6.0),
                IRSegmentationPoint(x=1.0, y=6.0),
            ],
        )
    ]
