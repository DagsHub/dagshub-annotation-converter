from typing import Sequence

from pydantic import BaseModel

from dagshub_annotation_converter.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.formats.label_studio.rectanglelabels import (
    RectangleLabelsAnnotationValue,
    RectangleLabelsAnnotation,
)
from dagshub_annotation_converter.ir.image import (
    IRPoseAnnotation,
    IRPosePoint,
    CoordinateStyle,
    IRAnnotationBase,
)


class KeyPointLabelsAnnotationValue(BaseModel):
    x: float
    y: float
    width: float = 1.0
    keypointlabels: list[str]


class KeyPointLabelsAnnotation(ImageAnnotationResultABC):
    value: KeyPointLabelsAnnotationValue
    type: str = "keypointlabels"

    def to_ir_annotation(self) -> list[IRPoseAnnotation]:
        ann = IRPoseAnnotation.from_points(
            category=self.value.keypointlabels[0],
            points=[IRPosePoint(x=self.value.x / 100, y=self.value.y / 100)],
            state=CoordinateStyle.NORMALIZED,
            image_width=self.original_width,
            image_height=self.original_height,
        )
        ann.imported_id = self.id
        return [ann]

    @staticmethod
    def from_ir_annotation(ir_annotation: IRAnnotationBase) -> Sequence["ImageAnnotationResultABC"]:
        assert isinstance(ir_annotation, IRPoseAnnotation)

        ir_annotation = ir_annotation.normalized()

        bbox = RectangleLabelsAnnotation(
            original_width=ir_annotation.image_width,
            original_height=ir_annotation.image_height,
            value=RectangleLabelsAnnotationValue(
                x=ir_annotation.left * 100,
                y=ir_annotation.top * 100,
                width=ir_annotation.width * 100,
                height=ir_annotation.height * 100,
                rectanglelabels=[ir_annotation.category],
            ),
        )

        points = []
        for point in ir_annotation.points:
            points.append(
                KeyPointLabelsAnnotation(
                    original_width=ir_annotation.image_width,
                    original_height=ir_annotation.image_height,
                    value=KeyPointLabelsAnnotationValue(
                        x=point.x * 100,
                        y=point.y * 100,
                        keypointlabels=[ir_annotation.category],
                    ),
                )
            )

        return [bbox, *points]
