from dagshub_annotation_converter.formats.yolo.segmentation import import_segmentation_from_string
from dagshub_annotation_converter.ir.image import CoordinateStyle
from dagshub_annotation_converter.ir.image.annotations.segmentation import (
    IRSegmentationImageAnnotation,
    IRSegmentationPoint,
)


def test_segmentation_import(yolo_context):
    points = [
        IRSegmentationPoint(x=0.5, y=0.5),
        IRSegmentationPoint(x=0.75, y=0.75),
        IRSegmentationPoint(x=0.5, y=0.75),
    ]
    expected = IRSegmentationImageAnnotation(
        categories={yolo_context.categories[0].name: 1.0},
        points=points,
        image_width=100,
        image_height=200,
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    actual = import_segmentation_from_string(
        context=yolo_context, annotation="0 0.5 0.5 0.75 0.75 0.5 0.75", image_width=100, image_height=200
    )

    assert expected == actual
