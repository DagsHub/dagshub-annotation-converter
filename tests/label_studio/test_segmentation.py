import pytest

from dagshub_annotation_converter.formats.label_studio.polygonlabels import PolygonLabelsAnnotation
from dagshub_annotation_converter.formats.label_studio.task import parse_ls_task, LabelStudioTask
from dagshub_annotation_converter.ir.image import (
    IRSegmentationAnnotation,
    NormalizationState,
    IRSegmentationPoint,
)
from tests.label_studio.common import generate_annotation, generate_task


@pytest.fixture
def segmentation_points() -> list[list[int]]:
    return [
        [50, 50],
        [75, 75],
        [75, 50],
    ]


@pytest.fixture
def segmentation_annotation(categories, segmentation_points) -> dict:
    annotation = {
        "points": segmentation_points,
        "polygonlabels": [categories[0].name],
        "closed": True,
    }
    return generate_annotation(annotation, "polygonlabels", "deadbeef")


@pytest.fixture
def segmentation_task(segmentation_annotation) -> str:
    return generate_task([segmentation_annotation])


@pytest.fixture
def parsed_segmentation_task(segmentation_task) -> LabelStudioTask:
    return parse_ls_task(segmentation_task)


def test_segmentation_parsing(parsed_segmentation_task, segmentation_points):
    actual = parsed_segmentation_task
    assert len(actual.annotations) == 1
    assert len(actual.annotations[0].result) == 1

    ann = actual.annotations[0].result[0]
    assert isinstance(ann, PolygonLabelsAnnotation)
    assert ann.value.points == segmentation_points


def test_segmentation_ir(parsed_segmentation_task, segmentation_points, categories):
    actual = parsed_segmentation_task.annotations[0].result[0].to_ir_annotation(categories)

    assert len(actual) == 1
    ann = actual[0]
    assert isinstance(ann, IRSegmentationAnnotation)

    converted_points = [IRSegmentationPoint(x=x / 100, y=y / 100) for x, y in segmentation_points]

    assert ann.points == converted_points
    assert ann.state == NormalizationState.NORMALIZED
