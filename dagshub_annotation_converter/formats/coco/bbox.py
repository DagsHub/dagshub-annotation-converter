from typing import Any, Dict

from dagshub_annotation_converter.formats.coco.context import CocoContext
from dagshub_annotation_converter.ir.image import CoordinateStyle
from dagshub_annotation_converter.ir.image.annotations.bbox import IRBBoxImageAnnotation


def import_bbox(annotation: Dict[str, Any], image: Dict[str, Any], context: CocoContext) -> IRBBoxImageAnnotation:
    category_id = int(annotation["category_id"])
    x, y, width, height = annotation["bbox"]

    return IRBBoxImageAnnotation(
        imported_id=str(annotation["id"]),
        categories={context.get_category_name(category_id): 1.0},
        top=float(y),
        left=float(x),
        width=float(width),
        height=float(height),
        image_width=int(image["width"]),
        image_height=int(image["height"]),
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )


def export_bbox(
    annotation: IRBBoxImageAnnotation,
    context: CocoContext,
    image_id: int,
    annotation_id: int,
) -> Dict[str, Any]:
    denormalized = annotation.denormalized()
    category_name = denormalized.ensure_has_one_category()
    category_id = context.get_category_id(category_name)

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [denormalized.left, denormalized.top, denormalized.width, denormalized.height],
        "iscrowd": 0,
        "area": denormalized.width * denormalized.height,
    }
