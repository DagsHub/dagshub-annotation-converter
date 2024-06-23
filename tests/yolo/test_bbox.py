from dagshub_annotation_converter.refactored.formats.yolo.bbox import import_bbox_from_string, export_bbox
from dagshub_annotation_converter.refactored.ir.image import NormalizationState
from dagshub_annotation_converter.refactored.ir.image.annotations.bbox import IRBBoxAnnotation


def test_bbox_import(yolo_context):
    expected = IRBBoxAnnotation(
        category=yolo_context.categories[0],
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        image_width=100,
        image_height=200,
        state=NormalizationState.NORMALIZED,
    )

    actual = import_bbox_from_string(
        context=yolo_context, annotation="0 0.75 0.75 0.5 0.5", image_width=100, image_height=200
    )

    assert expected == actual


def test_bbox_export(yolo_context):
    expected = "0 0.75 0.75 0.5 0.5"

    annotation = IRBBoxAnnotation(
        category=yolo_context.categories[0],
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        image_width=100,
        image_height=200,
        state=NormalizationState.NORMALIZED,
    )

    assert expected == export_bbox(annotation, yolo_context)
