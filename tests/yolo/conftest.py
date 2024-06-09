import pytest
from dagshub_annotation_converter.refactored.formats.yolo.context import YoloContext


@pytest.fixture
def yolo_context(categories) -> YoloContext:
    context = YoloContext(categories=categories)
    return context
