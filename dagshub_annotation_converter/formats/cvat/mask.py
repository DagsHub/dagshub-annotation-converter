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
    try:
        rle = [int(c) for c in str(elem.attrib["rle"]).split(",")]
    except ValueError as e:
        raise ValueError(f"CVAT mask on image {image_info.name} has a malformed RLE: {e}") from e

    if any(run < 0 for run in rle):
        raise ValueError(f"CVAT mask on image {image_info.name} has a negative RLE run")
    if sum(rle) != box_width * box_height:
        raise ValueError(
            f"CVAT mask on image {image_info.name} has RLE of {sum(rle)} pixels, "
            f"but its box is {box_width}x{box_height}"
        )

    box_values = np.arange(len(rle)) % 2 == 1
    box_mask = np.repeat(box_values, rle).reshape(box_height, box_width)

    mask = np.zeros((image_info.height, image_info.width), dtype=bool)
    # Clip the box to the image, in case CVAT exported a box that goes over the edge
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + box_width, image_info.width), min(top + box_height, image_info.height)
    if x0 < x1 and y0 < y1:
        mask[y0:y1, x0:x1] = box_mask[y0 - top : y1 - top, x0 - left : x1 - left]

    return IRBinaryMaskImageAnnotation.from_mask(
        mask,
        categories={category: 1.0},
        filename=image_info.name,
        meta=parse_metadata(elem, _keys),
    )
