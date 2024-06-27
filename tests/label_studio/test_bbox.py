from typing import Dict

import pytest

from dagshub_annotation_converter.formats.label_studio.rectanglelabels import RectangleLabelsAnnotation
from dagshub_annotation_converter.formats.label_studio.task import parse_ls_task, LabelStudioTask
from dagshub_annotation_converter.ir.image import IRBBoxAnnotation, NormalizationState
from tests.label_studio.common import generate_task, generate_annotation


@pytest.fixture
def bbox_annotation() -> Dict:
    annotation = {
        "x": 25,
        "y": 25,
        "width": 50,
        "height": 50,
        "rectanglelabels": ["dog"],
    }
    return generate_annotation(annotation, "rectanglelabels", "deadbeef")


@pytest.fixture
def bbox_task(bbox_annotation) -> str:
    return generate_task([bbox_annotation])


@pytest.fixture
def parsed_bbox_task(bbox_task) -> LabelStudioTask:
    return parse_ls_task(bbox_task)


def test_bbox_parsing(parsed_bbox_task):
    actual = parsed_bbox_task
    assert len(actual.annotations) == 1
    assert len(actual.annotations[0].result) == 1

    ann = actual.annotations[0].result[0]
    assert isinstance(ann, RectangleLabelsAnnotation)
    assert ann.value.x == 25
    assert ann.value.y == 25
    assert ann.value.width == 50
    assert ann.value.height == 50


def test_bbox_ir(parsed_bbox_task):
    actual = parsed_bbox_task.annotations[0].result[0].to_ir_annotation()

    assert len(actual) == 1
    ann = actual[0]
    assert isinstance(ann, IRBBoxAnnotation)

    assert ann.top == 0.25
    assert ann.left == 0.25
    assert ann.width == 0.5
    assert ann.height == 0.5
    assert ann.state == NormalizationState.NORMALIZED
