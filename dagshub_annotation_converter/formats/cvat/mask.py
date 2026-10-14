import numpy as np
from lxml.etree import ElementBase

from dagshub_annotation_converter.formats.cvat.context import parse_image_tag, parse_metadata
from dagshub_annotation_converter.ir.image import IRBinaryMaskImageAnnotation

_keys = {"label", "rle", "left", "top", "width", "height"}


def parse_mask(elem: ElementBase, containing_image: ElementBase) -> IRBinaryMaskImageAnnotation:
    """
    Parses a CVAT ``<mask>`` (brush) annotation.

    CVAT stores the mask only inside its bounding box (``left``, ``top``, ``width``, ``height``).
    The ``rle`` attribute holds run lengths over the box pixels in row-major order,
    starting with a background run.
    """
    category = str(elem.attrib["label"])
    image_info = parse_image_tag(containing_image)

    left = int(elem.attrib["left"])
    top = int(elem.attrib["top"])
    box_width = int(elem.attrib["width"])
    box_height = int(elem.attrib["height"])
    rle = [int(c) for c in str(elem.attrib["rle"]).split(",")]

    if sum(rle) != box_width * box_height:
        raise ValueError(
            f"CVAT mask on image {image_info.name} has RLE of {sum(rle)} pixels, "
            f"but its box is {box_width}x{box_height}"
        )

    box_values = np.arange(len(rle)) % 2 == 1
    box_mask = np.repeat(box_values, rle).reshape(box_height, box_width)

    mask = np.zeros((image_info.height, image_info.width), dtype=bool)
    # Clip the box to the image, in case CVAT exported a box that goes over the edge
    visible = mask[top : top + box_height, left : left + box_width]
    visible[:] = box_mask[: visible.shape[0], : visible.shape[1]]

    return IRBinaryMaskImageAnnotation.from_mask(
        mask,
        categories={category: 1.0},
        filename=image_info.name,
        meta=parse_metadata(elem, _keys),
    )
