import pytest

from dagshub_annotation_converter.formats.cvat.box import calculate_bbox


@pytest.mark.parametrize(
    "in_vals, expected",
    [
        ((100, 200, 300, 400, 0), (100, 200, 200, 200, 0)),
        ((122, 124, 495, 290, 45), (235, 16, 373, 166, 45)),
        (
            (469, 338, 594, 393, 345),
            (464, 355, 125, 55, 345),
        ),
    ],
)
def test_calculate_bbox_rotation(in_vals, expected):
    xtl, ytl, xbr, ybr, deg = in_vals
    expected_xtl, expected_ytl, expected_width, expected_height, expected_deg = expected

    assert calculate_bbox(xtl, ytl, xbr, ybr, deg) == (
        expected_xtl,
        expected_ytl,
        expected_width,
        expected_height,
        expected_deg,
    )
