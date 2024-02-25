from abc import abstractmethod
from enum import Enum
from os import PathLike
from typing import List, Optional, Union, Dict
from typing_extensions import Self

from pydantic import BaseModel

"""
This file contains classes for the intermediate representation of the annotations
"""


class Category(BaseModel):
    name: str
    id: int


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

    def get(self, item: Union[int, str], default=None) -> Category:
        try:
            return self[item]
        except KeyError:
            return default

    def add(self, name: str, id: Optional[int] = None):
        if id is None:
            id = max(self._id_lookup.keys()) + 1
        self.categories.append(Category(name=name, id=id))
        self.regenerate_dicts()

    def regenerate_dicts(self):
        self._id_lookup = {k.id: k for k in self.categories}
        self._name_lookup = {k.name: k for k in self.categories}


class AnnotationABC(BaseModel):
    @abstractmethod
    def normalized(self, image_width: int, image_height: int) -> Self:
        """
        Returns a copy with all parameters in the annotation normalized
        """
        pass

    @abstractmethod
    def denormalized(self, image_width: int, image_height: int) -> Self:
        """
        Returns a copy with all parameters in the annotation denormalized
        """
        pass


class BBoxAnnotation(AnnotationABC):
    category: Category
    # Pixel values in the image
    # The intermediate representation is assumed to be denormalized,
    # but normalized objects might be used in the process of importing/exporting
    top: float
    left: float
    width: float
    height: float
    state: NormalizationState = NormalizationState.DENORMALIZED

    def denormalized(self, image_width: int, image_height: int) -> "BBoxAnnotation":
        if self.state == NormalizationState.DENORMALIZED:
            return self.copy()

        return BBoxAnnotation(
            category=self.category,
            top=self.top * image_height,
            left=self.left * image_width,
            width=self.width * image_width,
            height=self.height * image_height,
            state=NormalizationState.DENORMALIZED,
        )

    def normalized(self, image_width: int, image_height: int) -> "BBoxAnnotation":
        if self.state == NormalizationState.NORMALIZED:
            return self.copy()

        return BBoxAnnotation(
            category=self.category,
            top=self.top / image_height,
            left=self.left / image_width,
            width=self.width / image_width,
            height=self.height / image_height,
            state=NormalizationState.NORMALIZED,
        )


class SegmentationPoint(BaseModel):
    # Pixel values in the image
    # The intermediate representation is assumed to be denormalized,
    # but normalized objects might be used in the process of importing/exporting
    x: float
    y: float


class SegmentationAnnotation(AnnotationABC):
    category: Category
    points: List[SegmentationPoint] = []
    state: NormalizationState = NormalizationState.DENORMALIZED

    def denormalized(self, image_width: int, image_height: int) -> "SegmentationAnnotation":
        if self.state == NormalizationState.DENORMALIZED:
            return self.copy()

        return SegmentationAnnotation(
            category=self.category,
            points=[
                SegmentationPoint(x=p.x * image_width, y=p.y * image_height)
                for p in self.points
            ],
            state=NormalizationState.DENORMALIZED,
        )

    def normalized(self, image_width: int, image_height: int) -> "SegmentationAnnotation":
        if self.state == NormalizationState.NORMALIZED:
            return self.copy()

        return SegmentationAnnotation(
            category=self.category,
            points=[
                SegmentationPoint(x=p.x / image_width, y=p.y / image_height)
                for p in self.points
            ],
            state=NormalizationState.NORMALIZED,
        )

    def add_point(self, x: float, y: float):
        self.points.append(SegmentationPoint(x=x, y=y))


class AnnotatedFile(BaseModel):
    file: PathLike
    annotations: List[AnnotationABC] = []
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    @property
    def categories(self) -> List[Category]:
        res = []
        for ann in self.annotations:
            if hasattr(ann, "category"):
                category = ann.category
                if category not in res:
                    res.append(category)
        return res


class AnnotationProject(BaseModel):
    categories: Categories = Categories()
    files: List[AnnotatedFile] = []
