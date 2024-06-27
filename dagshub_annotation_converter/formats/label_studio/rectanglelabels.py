from pydantic import BaseModel

from dagshub_annotation_converter.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.ir.image import IRBBoxAnnotation, NormalizationState


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
