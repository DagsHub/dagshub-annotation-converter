from typing import Dict

import pytest

from dagshub_annotation_converter.formats.label_studio.ellipselabels import EllipseLabelsAnnotation
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask, parse_ls_task
from dagshub_annotation_converter.ir.image import IREllipseImageAnnotation, CoordinateStyle
from tests.label_studio.common import generate_annotation, generate_task


@pytest.fixture
def ellipse_annotation() -> Dict:
    annotation = {
        "x": 10,
        "y": 20,
        "radiusX": 30,
        "radiusY": 40,
        "ellipselabels": ["dog"],
    }

    return generate_annotation(annotation, "ellipselabels", "deadbeef")


@pytest.fixture
def ellipse_task(ellipse_annotation) -> str:
    return generate_task([ellipse_annotation])


@pytest.fixture
def parsed_ellipse_task(ellipse_task) -> LabelStudioTask:
    return parse_ls_task(ellipse_task)


def test_ellipse_parsing(parsed_ellipse_task):
    actual = parsed_ellipse_task
    assert len(actual.annotations) == 1
    assert len(actual.annotations[0].result) == 1

    ann = actual.annotations[0].result[0]
    assert isinstance(ann, EllipseLabelsAnnotation)
    assert ann.value.x == 10
    assert ann.value.y == 20
    assert ann.value.radiusX == 30
    assert ann.value.radiusY == 40


def test_ellipse_ir(parsed_ellipse_task):
    actual = parsed_ellipse_task.annotations[0].result[0].to_ir_annotation()

    assert len(actual) == 1
    ann = actual[0]
    assert isinstance(ann, IREllipseImageAnnotation)

    assert ann.center_x == 0.1
    assert ann.center_y == 0.2
    assert ann.radius_x == 0.3
    assert ann.radius_y == 0.4


def test_ir_ellipse_addition():
    task = LabelStudioTask()
    ellipse = IREllipseImageAnnotation(
        categories={"dog": 1.0},
        center_x=0.1,
        center_y=0.2,
        radius_x=0.3,
        radius_y=0.4,
        rotation=60.0,
        image_width=100,
        image_height=100,
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    task.add_ir_annotation(ellipse)

    assert len(task.annotations) == 1
    assert len(task.annotations[0].result) == 1

    ann = task.annotations[0].result[0]
    assert isinstance(ann, EllipseLabelsAnnotation)
    assert ann.value.x == 10
    assert ann.value.y == 20
    assert ann.value.radiusX == 30
    assert ann.value.radiusY == 40
    assert ann.value.rotation == 60.0
