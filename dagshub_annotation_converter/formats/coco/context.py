from typing import Dict

from pydantic import Field

from dagshub_annotation_converter.util.pydantic_util import ParentModel


class CocoContext(ParentModel):
    """
    Context for COCO import/export.

    Keeps a mapping of category id -> category name.
    """

    categories: Dict[int, str] = Field(default_factory=dict)

    def get_category_name(self, category_id: int) -> str:
        return self.categories.get(category_id, str(category_id))

    def get_category_id(self, category_name: str) -> int:
        for category_id, name in self.categories.items():
            if name == category_name:
                return category_id
        new_id = max(self.categories.keys(), default=0) + 1
        self.categories[new_id] = category_name
        return new_id
