from typing import List

from pydantic import BaseModel

from dagshub_annotation_converter.refactored.ir.image.annotations.base import IRAnnotationBase


class IRSegmentationPoint(BaseModel):
    x: float
    y: float


class IRSegmentationAnnotation(IRAnnotationBase):
    points: List[IRSegmentationPoint] = []

    def _normalize(self):
        self.points = [IRSegmentationPoint(x=p.x / self.image_width, y=p.y / self.image_height) for p in self.points]

    def _denormalize(self):
        self.points = [IRSegmentationPoint(x=p.x * self.image_width, y=p.y * self.image_height) for p in self.points]

    def add_point(self, x: float, y: float):
        self.points.append(IRSegmentationPoint(x=x, y=y))
