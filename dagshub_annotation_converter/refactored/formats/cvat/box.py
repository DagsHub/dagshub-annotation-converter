from lxml.etree import ElementBase

from dagshub_annotation_converter.refactored.formats.cvat.context import CVATContext, parse_image_tag
from dagshub_annotation_converter.refactored.ir.image import IRBBoxAnnotation, NormalizationState


def parse_box(context: CVATContext, elem: ElementBase, containing_image: ElementBase) -> IRBBoxAnnotation:
    top = float(elem.attrib["ytl"])
    bottom = float(elem.attrib["ybr"])
    left = float(elem.attrib["xtl"])
    right = float(elem.attrib["xbr"])

    width = right - left
    height = bottom - top

    image_info = parse_image_tag(containing_image)

    category = context.categories.get_or_create(str(elem.attrib["label"]))

    return IRBBoxAnnotation(
        top=top,
        left=left,
        width=width,
        height=height,
        category=category,
        image_width=image_info.width,
        image_height=image_info.height,
        filename=image_info.name,
        state=NormalizationState.DENORMALIZED,
    )
