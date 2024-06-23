from lxml.etree import ElementBase

from dagshub_annotation_converter.refactored.formats.cvat.context import CVATContext, parse_image_tag
from dagshub_annotation_converter.refactored.ir.image import IRSegmentationAnnotation, NormalizationState


def parse_polygon(context: CVATContext, elem: ElementBase, containing_image: ElementBase) -> IRSegmentationAnnotation:
    category = context.categories.get_or_create(str(elem.attrib["label"]))

    image_info = parse_image_tag(containing_image)

    res = IRSegmentationAnnotation(
        category=category,
        state=NormalizationState.DENORMALIZED,
        image_width=image_info.width,
        image_height=image_info.height,
        filename=image_info.name,
    )

    for point_str in elem.attrib["points"].split(";"):
        x, y = point_str.split(",")
        res.add_point(x=float(x), y=float(y))

    return res
