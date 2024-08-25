from lxml.etree import ElementBase

from dagshub_annotation_converter.formats.cvat.context import parse_image_tag
from dagshub_annotation_converter.ir.image import IRSegmentationImageAnnotation, CoordinateStyle


def parse_polygon(elem: ElementBase, containing_image: ElementBase) -> IRSegmentationImageAnnotation:
    category = str(elem.attrib["label"])

    image_info = parse_image_tag(containing_image)

    res = IRSegmentationImageAnnotation(
        categories={category: 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        image_width=image_info.width,
        image_height=image_info.height,
        filename=image_info.name,
    )

    for point_str in elem.attrib["points"].split(";"):
        x, y = point_str.split(",")
        res.add_point(x=float(x), y=float(y))

    return res