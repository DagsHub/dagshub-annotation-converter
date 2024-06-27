from enum import Enum
from typing import List, Optional, Union, Dict

from pydantic import BaseModel


class Category(BaseModel):
    name: str
    id: int

    def __hash__(self):
        return hash(self.name)


class NormalizationState(Enum):
    NORMALIZED = (0,)
    DENORMALIZED = (1,)


class Categories(BaseModel):
    categories: List[Category] = []
    _id_lookup: Dict[int, Category] = {}
    _name_lookup: Dict[str, Category] = {}

    def __getitem__(self, item: Union[int, str]) -> Category:
        if isinstance(item, int):
            return self._id_lookup[item]
        else:
            return self._name_lookup[item]

    def __iter__(self):
        return self.categories.__iter__()

    def __len__(self):
        return len(self.categories)

    def get(self, item: Union[int, str], default=None) -> Category:
        try:
            return self[item]
        except KeyError:
            return default

    def get_or_create(self, name: str) -> Category:
        if name not in self:
            return self.add(name)
        return self[name]

    def __contains__(self, item: str):
        return item in self._name_lookup

    def add(self, name: str, id: Optional[int] = None) -> Category:
        if id is None:
            if len(self._id_lookup):
                id = max(self._id_lookup.keys()) + 1
            else:
                id = 0
        new_category = Category(name=name, id=id)
        self.categories.append(new_category)
        self.regenerate_dicts()
        return new_category

    def regenerate_dicts(self):
        self._id_lookup = {k.id: k for k in self.categories}
        self._name_lookup = {k.name: k for k in self.categories}
