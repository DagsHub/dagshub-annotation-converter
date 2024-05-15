from pydantic import BaseModel

from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationABC,
    AnnotationProject,
    SegmentationAnnotation,
    NormalizationState,
)
from dagshub_annotation_converter.schema.label_studio.abc import ImageAnnotationResultABC


class PolygonLabelsAnnotationValue(BaseModel):
    points: list[list[float]]
    polygonlabels: list[str]
    closed: bool = True


class PolygonLabelsAnnotation(ImageAnnotationResultABC):
    value: PolygonLabelsAnnotationValue
    type: str = "polygonlabels"

    def to_ir_annotation(self, project: AnnotationProject) -> list[AnnotationABC]:
        category = project.categories.get_or_create(self.value.polygonlabels[0])
        res = SegmentationAnnotation(category=category, state=NormalizationState.NORMALIZED)
        for p in self.value.points:
            res.add_point(p[0] / 100.0, p[1] / 100.0)
        return [res]
