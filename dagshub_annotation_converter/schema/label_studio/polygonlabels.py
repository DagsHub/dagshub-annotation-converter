from typing import List

from pydantic import BaseModel

from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationABC,
    AnnotationProject,
    SegmentationAnnotation,
    NormalizationState,
)
from dagshub_annotation_converter.schema.label_studio.abc import ImageAnnotationResultABC


class PolygonLabelsAnnotationValue(BaseModel):
    points: List[List[float]]
    polygonlabels: List[str]
    closed: bool = True


class PolygonLabelsAnnotation(ImageAnnotationResultABC):
    value: PolygonLabelsAnnotationValue
    type: str = "polygonlabels"

    def to_ir_annotation(self, project: AnnotationProject) -> List[AnnotationABC]:
        category = project.categories.get_or_create(self.value.polygonlabels[0])
        res = SegmentationAnnotation(category=category, state=NormalizationState.NORMALIZED)
        for p in self.value.points:
            res.add_point(p[0] / 100.0, p[1] / 100.0)
        res.imported_id = self.id
        return [res]
