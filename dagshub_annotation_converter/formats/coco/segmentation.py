import logging
from typing import Any, Dict, List, Sequence

from dagshub_annotation_converter.formats.coco.context import CocoContext
from dagshub_annotation_converter.ir.image import CoordinateStyle
from dagshub_annotation_converter.ir.image.annotations.segmentation import (
    IRSegmentationImageAnnotation,
    IRSegmentationPoint,
)

logger = logging.getLogger(__name__)


def _polygon_area(points: List[IRSegmentationPoint]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, point in enumerate(points):
        next_point = points[(idx + 1) % len(points)]
        total += point.x * next_point.y - next_point.x * point.y
    return abs(total) * 0.5


def import_segmentation(
    annotation: Dict[str, Any], image: Dict[str, Any], context: CocoContext
) -> List[IRSegmentationImageAnnotation]:
    segmentation = annotation["segmentation"]

    if not isinstance(segmentation, list):
        logger.warning(
            "Skipping non-polygon COCO segmentation annotation id=%s (RLE segmentation is not supported)",
            annotation.get("id"),
        )
        return []

    category_id = int(annotation["category_id"])
    category_name = context.categories[category_id].name
    imported_id = str(annotation["id"])
    parsed: List[IRSegmentationImageAnnotation] = []

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
            logger.warning(
                "Skipping invalid polygon in COCO segmentation annotation id=%s",
                annotation.get("id"),
            )
            continue

        points = [IRSegmentationPoint(x=float(polygon[i]), y=float(polygon[i + 1])) for i in range(0, len(polygon), 2)]
        parsed.append(
            IRSegmentationImageAnnotation(
                imported_id=imported_id,
                categories={category_name: 1.0},
                points=points,
                image_width=int(image["width"]),
                image_height=int(image["height"]),
                coordinate_style=CoordinateStyle.DENORMALIZED,
            )
        )

    return parsed


def export_segmentation(
    annotation: IRSegmentationImageAnnotation,
    context: CocoContext,
    image_id: int,
    annotation_id: int,
) -> Dict[str, Any]:
    denormalized = annotation.denormalized()
    category_name = denormalized.ensure_has_one_category()
    category_id = context.categories.get_or_create(category_name).id

    segmentation = []
    for point in denormalized.points:
        segmentation.extend([point.x, point.y])

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [segmentation],
        "iscrowd": 0,
        "area": _polygon_area(denormalized.points),
    }


def export_segmentation_group(
    annotations: Sequence[IRSegmentationImageAnnotation],
    context: CocoContext,
    image_id: int,
    annotation_id: int,
) -> Dict[str, Any]:
    if len(annotations) == 0:
        raise ValueError("Cannot export an empty segmentation group")

    first = annotations[0].denormalized()
    category_name = first.ensure_has_one_category()
    category_id = context.categories.get_or_create(category_name).id

    segmentation: List[List[float]] = []
    area = 0.0
    for annotation in annotations:
        denormalized = annotation.denormalized()
        if denormalized.ensure_has_one_category() != category_name:
            raise ValueError("All segmentation annotations in a group must have the same category")
        polygon: List[float] = []
        for point in denormalized.points:
            polygon.extend([point.x, point.y])
        segmentation.append(polygon)
        area += _polygon_area(denormalized.points)

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "iscrowd": 0,
        "area": area,
    }
