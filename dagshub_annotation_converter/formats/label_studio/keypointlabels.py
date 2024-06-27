from pydantic import BaseModel

from dagshub_annotation_converter.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.ir.image import (
    IRPoseAnnotation,
    IRPosePoint,
    NormalizationState,
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
            state=NormalizationState.NORMALIZED,
            image_width=self.original_width,
            image_height=self.original_height,
        )
        ann.imported_id = self.id
        return [ann]
