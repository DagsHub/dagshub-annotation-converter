from abc import abstractmethod
from enum import Enum
from os import PathLike
from typing import List, Optional, Union, Dict, Literal
from typing_extensions import Self

from pydantic import BaseModel

"""
This file contains classes for the intermediate representation of the annotations
"""


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
                id = 1
        new_category = Category(name=name, id=id)
        self.categories.append(new_category)
        self.regenerate_dicts()
        return new_category

    def regenerate_dicts(self):
        self._id_lookup = {k.id: k for k in self.categories}
        self._name_lookup = {k.name: k for k in self.categories}


class AnnotationABC(BaseModel):
    imported_id: Optional[str] = None

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
            points=[SegmentationPoint(x=p.x * image_width, y=p.y * image_height) for p in self.points],
            state=NormalizationState.DENORMALIZED,
        )

    def normalized(self, image_width: int, image_height: int) -> "SegmentationAnnotation":
        if self.state == NormalizationState.NORMALIZED:
            return self.copy()

        return SegmentationAnnotation(
            category=self.category,
            points=[SegmentationPoint(x=p.x / image_width, y=p.y / image_height) for p in self.points],
            state=NormalizationState.NORMALIZED,
        )

    def add_point(self, x: float, y: float):
        self.points.append(SegmentationPoint(x=x, y=y))


class KeyPoint(BaseModel):
    x: float
    y: float
    is_visible: Optional[bool] = None


class PoseAnnotation(AnnotationABC):
    category: Category

    # Parameters of the bounding box
    top: float
    left: float
    width: float
    height: float

    points: List[KeyPoint] = []
    state: NormalizationState = NormalizationState.DENORMALIZED

    def normalized(self, image_width: int, image_height: int) -> Self:
        if self.state == NormalizationState.NORMALIZED:
            return self.copy()

        res: "Self" = self.copy()
        new_points: List[KeyPoint] = [point.copy() for point in self.points]
        for point in new_points:
            point.x = point.x / image_width
            point.y = point.y / image_height

        res.top = res.top / image_height
        res.left = res.left / image_width
        res.width = res.width / image_width
        res.height = res.height / image_height
        res.points = new_points
        res.state = NormalizationState.NORMALIZED

        return res

    def denormalized(self, image_width: int, image_height: int) -> Self:
        if self.state == NormalizationState.DENORMALIZED:
            return self.copy()

        res: "Self" = self.copy()
        new_points: List[KeyPoint] = [point.copy() for point in self.points]
        for point in new_points:
            point.x = point.x * image_width
            point.y = point.y * image_height

        res.top = res.top * image_height
        res.left = res.left * image_width
        res.width = res.width * image_width
        res.height = res.height * image_height
        res.points = new_points
        res.state = NormalizationState.DENORMALIZED

        return res

    def add_point(self, x: float, y: float, is_visible: Optional[bool]):
        self.points.append(KeyPoint(x=x, y=y, is_visible=is_visible))

    @staticmethod
    def from_points(category: Category, points: List[KeyPoint], state: NormalizationState) -> "PoseAnnotation":
        point_xs = list(map(lambda p: p.x, points))
        point_ys = list(map(lambda p: p.y, points))

        min_x = min(point_xs)
        max_x = max(point_xs)
        min_y = min(point_ys)
        max_y = max(point_ys)

        return PoseAnnotation(
            top=min_y,
            left=min_x,
            width=max_x - min_x,
            height=max_y - min_y,
            points=points,
            category=category,
            state=state,
        )


class AnnotatedFile(BaseModel):
    file: Union[PathLike, str]
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


class PoseConfig(BaseModel):
    pose_points: dict[Category, int] = {}
    flipped_points: dict[Category, list[int]] = {}


class YoloImportConfig(BaseModel):
    keypoint_shape: Literal[2, 3] = 2


class ImportConfig(BaseModel):
    yolo: YoloImportConfig = YoloImportConfig()


class AnnotationProject(BaseModel):
    categories: Categories = Categories()
    files: List[AnnotatedFile] = []
    pose_config: PoseConfig = PoseConfig()
    import_config: ImportConfig = ImportConfig()
