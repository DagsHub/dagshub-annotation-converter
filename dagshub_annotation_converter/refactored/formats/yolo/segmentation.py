from typing import Union, Optional, Tuple, Sequence

from dagshub_annotation_converter.refactored.formats.common import (
    ImageType,
    determine_category,
    determine_image_dimensions,
)
from dagshub_annotation_converter.refactored.formats.yolo.context import YoloContext
from dagshub_annotation_converter.refactored.ir.image import NormalizationState
from dagshub_annotation_converter.refactored.ir.image.annotations.segmentation import (
    IRSegmentationAnnotation,
    IRSegmentationPoint,
)


def import_segment(
    category: Union[int, str],
    points: Sequence[Tuple[float, float]],
    context: YoloContext,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    image: Optional[ImageType] = None,
):
    image_width, image_height = determine_image_dimensions(
        image_width=image_width, image_height=image_height, image=image
    )
    parsed_category = determine_category(category, context.categories)

    return IRSegmentationAnnotation(
        category=parsed_category,
        image_width=image_width,
        image_height=image_height,
        state=NormalizationState.NORMALIZED,
        points=[IRSegmentationPoint(x=x, y=y) for x, y in points],
    )


def import_segment_from_string(
    annotation: str,
    context: YoloContext,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    image: Optional[ImageType] = None,
) -> IRSegmentationAnnotation:
    if len(annotation.split("\n")) > 1:
        raise ValueError("Please pass one annotation at a time")
    parts = annotation.strip().split(" ")
    category = int(parts[0])
    points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
    return import_segment(
        category=category,
        points=points,
        context=context,
        image_width=image_width,
        image_height=image_height,
        image=image,
    )


def export_segment(annotation: IRSegmentationAnnotation) -> str:
    return " ".join(
        [
            str(annotation.category.id),
            *[f"{point.x} {point.y}" for point in annotation.points],
        ]
    )
