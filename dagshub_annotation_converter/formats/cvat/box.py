from lxml.etree import ElementBase

from dagshub_annotation_converter.formats.cvat.context import parse_image_tag
from dagshub_annotation_converter.ir.image import IRBBoxAnnotation, NormalizationState


def parse_box(elem: ElementBase, containing_image: ElementBase) -> IRBBoxAnnotation:
    top = float(elem.attrib["ytl"])
    bottom = float(elem.attrib["ybr"])
    left = float(elem.attrib["xtl"])
    right = float(elem.attrib["xbr"])

    width = right - left
    height = bottom - top

    image_info = parse_image_tag(containing_image)

    return IRBBoxAnnotation(
        category=str(elem.attrib["label"]),
        top=top,
        left=left,
        width=width,
        height=height,
        image_width=image_info.width,
        image_height=image_info.height,
        filename=image_info.name,
        state=NormalizationState.DENORMALIZED,
    )