from typing import Sequence

from pydantic import BaseModel

from dagshub_annotation_converter.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.ir.image import IRBBoxAnnotation, NormalizationState, IRAnnotationBase


class RectangleLabelsAnnotationValue(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rectanglelabels: list[str]


class RectangleLabelsAnnotation(ImageAnnotationResultABC):
    value: RectangleLabelsAnnotationValue
    type: str = "rectanglelabels"

    def to_ir_annotation(self) -> list[IRBBoxAnnotation]:
        res = IRBBoxAnnotation(
            category=self.value.rectanglelabels[0],
            state=NormalizationState.NORMALIZED,
            top=self.value.y / 100,
            left=self.value.x / 100,
            width=self.value.width / 100,
            height=self.value.height / 100,
            image_width=self.original_width,
            image_height=self.original_height,
        )
        res.imported_id = self.id
        return [res]

    @staticmethod
    def from_ir_annotation(ir_annotation: IRAnnotationBase) -> Sequence[ImageAnnotationResultABC]:
        assert isinstance(ir_annotation, IRBBoxAnnotation)

        ir_annotation = ir_annotation.normalized()

        return [
            RectangleLabelsAnnotation(
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
        ]
