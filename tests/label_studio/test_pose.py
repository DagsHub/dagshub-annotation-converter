import pytest

from dagshub_annotation_converter.formats.label_studio.keypointlabels import KeyPointLabelsAnnotation
from dagshub_annotation_converter.formats.label_studio.task import parse_ls_task, LabelStudioTask
from dagshub_annotation_converter.ir.image import IRPoseAnnotation, NormalizationState
from tests.label_studio.common import generate_annotation, generate_task


@pytest.fixture
def keypoint_annotation() -> dict:
    annotation = {
        "x": 50,
        "y": 50,
        "keypointlabels": ["cat"],
    }
    return generate_annotation(annotation, "keypointlabels", "deadbeef")


@pytest.fixture
def keypoint_task(keypoint_annotation) -> str:
    return generate_task([keypoint_annotation])


@pytest.fixture
def parsed_keypoint_task(keypoint_task) -> LabelStudioTask:
    return parse_ls_task(keypoint_task)


def test_keypoint_parsing(parsed_keypoint_task):
    actual = parsed_keypoint_task
    assert len(actual.annotations) == 1
    assert len(actual.annotations[0].result) == 1

    ann = actual.annotations[0].result[0]
    assert isinstance(ann, KeyPointLabelsAnnotation)
    assert ann.value.x == 50
    assert ann.value.y == 50


def test_keypoint_ir(parsed_keypoint_task):
    actual = parsed_keypoint_task.annotations[0].result[0].to_ir_annotation()

    assert len(actual) == 1
    ann = actual[0]
    assert isinstance(ann, IRPoseAnnotation)

    assert len(ann.points) == 1
    assert ann.points[0].x == 0.5
    assert ann.points[0].y == 0.5

    assert ann.state == NormalizationState.NORMALIZED
