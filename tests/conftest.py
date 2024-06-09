import pytest

from dagshub_annotation_converter.refactored.ir.image import Categories


@pytest.fixture
def categories() -> Categories:
    categories = Categories()
    categories.add("cat")
    return categories
