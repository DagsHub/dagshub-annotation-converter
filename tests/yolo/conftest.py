import pytest
from dagshub_annotation_converter.refactored.formats.yolo.context import YoloContext


@pytest.fixture
def yolo_context(categories) -> YoloContext:
    # Using bbox here because it doesn't matter for the export tests
    context = YoloContext(annotation_type="bbox", categories=categories)
    return context
