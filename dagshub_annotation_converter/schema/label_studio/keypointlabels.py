from typing import List

from pydantic import BaseModel

from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationABC,
    AnnotationProject,
    PoseAnnotation,
    NormalizationState,
    KeyPoint,
)
from dagshub_annotation_converter.schema.label_studio.abc import ImageAnnotationResultABC


class KeyPointLabelsAnnotationValue(BaseModel):
    x: float
    y: float
    width: float = 1.0
    keypointlabels: List[str]


class KeyPointLabelsAnnotation(ImageAnnotationResultABC):
    value: KeyPointLabelsAnnotationValue
    type: str = "keypointlabels"

    def to_ir_annotation(self, project: AnnotationProject) -> List[AnnotationABC]:
        category = project.categories.get_or_create(self.value.keypointlabels[0])

        ann = PoseAnnotation.from_points(
            category=category,
            points=[KeyPoint(x=self.value.x / 100, y=self.value.y / 100)],
            state=NormalizationState.NORMALIZED,
        )
        ann.imported_id = self.id
        return [ann]
