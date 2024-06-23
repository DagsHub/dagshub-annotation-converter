from typing import Optional, Union


from dagshub_annotation_converter.refactored.formats.common import (
    determine_category,
    ImageType,
    determine_image_dimensions,
)
from dagshub_annotation_converter.refactored.formats.yolo.context import YoloContext
from dagshub_annotation_converter.refactored.ir.image import NormalizationState
from dagshub_annotation_converter.refactored.ir.image.annotations.bbox import IRBBoxAnnotation


def import_bbox(
    category: Union[int, str],
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    context: YoloContext,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    image: Optional[ImageType] = None,
) -> IRBBoxAnnotation:
    image_width, image_height = determine_image_dimensions(
        image_width=image_width, image_height=image_height, image=image
    )
    parsed_category = determine_category(category, context.categories)
    return IRBBoxAnnotation(
        category=parsed_category,
        top=center_y - height / 2,
        left=center_x - width / 2,
        width=width,
        height=height,
        image_width=image_width,
        image_height=image_height,
        state=NormalizationState.NORMALIZED,
    )


def import_bbox_from_string(
    annotation: str,
    context: YoloContext,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    image: Optional[ImageType] = None,
) -> IRBBoxAnnotation:
    if len(annotation.split("\n")) > 1:
        raise ValueError("Please pass one annotation at a time")
    parts = annotation.strip().split(" ")
    category = int(parts[0])
    center_x = float(parts[1])
    center_y = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    return import_bbox(
        category=category,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
        context=context,
        image_width=image_width,
        image_height=image_height,
        image=image,
    )


def export_bbox(
    annotation: IRBBoxAnnotation,
    context: YoloContext,
) -> str:
    center_x = annotation.left + annotation.width / 2
    center_y = annotation.top + annotation.height / 2
    return f"{annotation.category.id} {center_x} {center_y} {annotation.width} {annotation.height}"
