import lxml.etree
import numpy as np
import pytest

from dagshub_annotation_converter.converters.cvat import load_cvat_from_xml_bytes
from dagshub_annotation_converter.formats.cvat import parse_mask
from dagshub_annotation_converter.ir.image import IRBinaryMaskImageAnnotation


def _parse(mask_xml: str, width: int = 6, height: int = 4) -> IRBinaryMaskImageAnnotation:
    image = lxml.etree.fromstring(f'<image id="0" name="img.png" width="{width}" height="{height}">{mask_xml}</image>')
    return parse_mask(image[0], image)


def test_mask_is_placed_in_its_box():
    # 3x2 box at (2, 1):
    #   1 1 0
    #   0 1 1
    ann = _parse(
        '<mask label="cat" source="manual" occluded="0" rle="0, 2, 2, 2" '
        'left="2" top="1" width="3" height="2" z_order="0"></mask>'
    )

    expected = np.zeros((4, 6), dtype=bool)
    expected[1, 2:4] = True
    expected[2, 3:5] = True

    assert ann.categories == {"cat": 1.0}
    assert ann.filename == "img.png"
    assert (ann.image_width, ann.image_height) == (6, 4)
    assert (ann.to_mask() == expected).all()
    assert ann.meta == {"source": "manual", "occluded": "0", "z_order": "0"}


def test_mask_box_over_image_edge_is_clipped():
    ann = _parse('<mask label="cat" rle="0, 4" left="5" top="3" width="2" height="2"></mask>')

    expected = np.zeros((4, 6), dtype=bool)
    expected[3, 5] = True
    assert (ann.to_mask() == expected).all()


def test_mask_box_over_negative_origin_is_clipped():
    # 3x3 box at (-1, -1), only its bottom-right 2x2 corner is inside the image:
    #   0 0 0
    #   0 1 0
    #   0 0 1
    ann = _parse('<mask label="cat" rle="4, 1, 3, 1" left="-1" top="-1" width="3" height="3"></mask>')

    expected = np.zeros((4, 6), dtype=bool)
    expected[0, 0] = True
    expected[1, 1] = True
    assert (ann.to_mask() == expected).all()


@pytest.mark.parametrize("rle", ["-1, 5", "1, x, 3", "1,,3"])
def test_mask_with_malformed_rle_raises(rle):
    with pytest.raises(ValueError, match="img.png"):
        _parse(f'<mask label="cat" rle="{rle}" left="0" top="0" width="2" height="2"></mask>')


def test_mask_with_wrong_rle_length_raises():
    with pytest.raises(ValueError):
        _parse('<mask label="cat" rle="1, 2" left="0" top="0" width="2" height="2"></mask>')


def test_mask_loads_from_cvat_xml():
    xml = b"""<?xml version="1.0" encoding="utf-8"?>
    <annotations>
      <version>1.1</version>
      <image id="0" name="empty.png" width="4" height="4"></image>
      <image id="1" name="masked.png" width="4" height="4">
        <mask label="cat" source="manual" occluded="0" rle="1, 1, 1, 1" left="1" top="1" width="2" height="2">
        </mask>
      </image>
    </annotations>
    """
    actual = load_cvat_from_xml_bytes(xml)

    assert len(actual["empty.png"]) == 0
    assert len(actual["masked.png"]) == 1
    ann = actual["masked.png"][0]
    assert isinstance(ann, IRBinaryMaskImageAnnotation)
    assert ann.to_mask()[1:3, 1:3].tolist() == [[False, True], [False, True]]
    assert ann.to_mask().sum() == 2
