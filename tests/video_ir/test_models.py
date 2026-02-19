import math

import pytest
from dagshub_annotation_converter.ir.video import (
    IRVideoBBoxAnnotation,
    CoordinateStyle,
)


@pytest.fixture
def epsilon():
    return 1e-6


def test_normalize_coordinates(epsilon):
    ann = IRVideoBBoxAnnotation(
        track_id=1,
        frame_number=0,
        left=192,  # 192/1920 = 0.1
        top=108,  # 108/1080 = 0.1
        width=384,  # 384/1920 = 0.2
        height=216,  # 216/1080 = 0.2
        image_width=1920,
        image_height=1080,
        categories={"person": 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
    )

    normalized = ann.normalized()

    assert normalized.coordinate_style == CoordinateStyle.NORMALIZED
    assert math.isclose(normalized.left, 0.1, abs_tol=epsilon)
    assert math.isclose(normalized.top, 0.1, abs_tol=epsilon)
    assert math.isclose(normalized.width, 0.2, abs_tol=epsilon)
    assert math.isclose(normalized.height, 0.2, abs_tol=epsilon)
    assert normalized.track_id == 1
    assert normalized.frame_number == 0


def test_denormalize_coordinates(epsilon):
    ann = IRVideoBBoxAnnotation(
        track_id=2,
        frame_number=10,
        left=0.1,
        top=0.1,
        width=0.2,
        height=0.2,
        image_width=1920,
        image_height=1080,
        categories={"car": 1.0},
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    denormalized = ann.denormalized()

    assert denormalized.coordinate_style == CoordinateStyle.DENORMALIZED
    assert math.isclose(denormalized.left, 192, abs_tol=epsilon)
    assert math.isclose(denormalized.top, 108, abs_tol=epsilon)
    assert math.isclose(denormalized.width, 384, abs_tol=epsilon)
    assert math.isclose(denormalized.height, 216, abs_tol=epsilon)
    assert denormalized.track_id == 2
    assert denormalized.frame_number == 10
