from typing import Dict

from pydantic import Field

from dagshub_annotation_converter.formats.categories import Categories
from dagshub_annotation_converter.util.pydantic_util import ParentModel


class CocoContext(ParentModel):
    """
    Context for COCO import/export.

    Keeps a mapping of category id -> category name.
    """

    categories: Categories = Categories()

