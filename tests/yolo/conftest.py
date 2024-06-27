import pytest
from dagshub_annotation_converter.formats.yolo.context import YoloContext


@pytest.fixture
def yolo_context() -> YoloContext:
    # Using bbox here because it doesn't matter for the export tests
    context = YoloContext(annotation_type="bbox")
    context.categories.add(name="cat")
    context.categories.add(name="dog")
    return context
