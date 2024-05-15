from pydantic import BaseModel

from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationABC,
    AnnotationProject,
    BBoxAnnotation,
    NormalizationState,
)
from dagshub_annotation_converter.schema.label_studio.abc import ImageAnnotationResultABC


class RectangleLabelsAnnotationValue(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rectanglelabels: list[str]


class RectangleLabelsAnnotation(ImageAnnotationResultABC):
    value: RectangleLabelsAnnotationValue
    type: str = "rectanglelabels"

    def to_ir_annotation(self, project: AnnotationProject) -> list[AnnotationABC]:
        res = BBoxAnnotation(
            category=project.categories.get_or_create(self.value.rectanglelabels[0]),
            state=NormalizationState.NORMALIZED,
            top=self.value.y / 100.0,
            left=self.value.x / 100.0,
            width=self.value.width / 100.0,
            height=self.value.height / 100.0,
        )
        return [res]
