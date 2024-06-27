from typing import List, Optional, TYPE_CHECKING

from pydantic import BaseModel

from dagshub_annotation_converter.ir.image.annotations.base import IRAnnotationBase

if TYPE_CHECKING:
    from dagshub_annotation_converter.ir.image import Category, NormalizationState


class IRPosePoint(BaseModel):
    x: float
    y: float
    visible: Optional[bool] = None


class IRPoseAnnotation(IRAnnotationBase):
    # Parameters of the bounding box
    top: float
    left: float
    width: float
    height: float

    points: List[IRPosePoint] = []

    def _normalize(self):
        self.top = self.top / self.image_height
        self.left = self.left / self.image_width
        self.width = self.width / self.image_width
        self.height = self.height / self.image_height

        for point in self.points:
            point.x = point.x / self.image_width
            point.y = point.y / self.image_height

    def _denormalize(self):
        self.top = self.top * self.image_height
        self.left = self.left * self.image_width
        self.width = self.width * self.image_width
        self.height = self.height * self.image_height

        for point in self.points:
            point.x = point.x * self.image_width
            point.y = point.y * self.image_height

    def add_point(self, x: float, y: float, visible: Optional[bool] = None):
        self.points.append(IRPosePoint(x=x, y=y, visible=visible))

    @staticmethod
    def from_points(
        category: "Category",
        points: List[IRPosePoint],
        state: "NormalizationState",
        image_width: int,
        image_height: int,
        filename: Optional[str] = None,
    ) -> "IRPoseAnnotation":
        point_xs = list(map(lambda p: p.x, points))
        point_ys = list(map(lambda p: p.y, points))

        min_x = min(point_xs)
        max_x = max(point_xs)
        min_y = min(point_ys)
        max_y = max(point_ys)

        return IRPoseAnnotation(
            category=category,
            filename=filename,
            top=min_y,
            left=min_x,
            width=max_x - min_x,
            height=max_y - min_y,
            points=points,
            state=state,
            image_height=image_height,
            image_width=image_width,
        )
