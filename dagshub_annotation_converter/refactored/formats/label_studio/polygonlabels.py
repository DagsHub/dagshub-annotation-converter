from pydantic import BaseModel


from dagshub_annotation_converter.refactored.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.refactored.ir.image import Categories, IRSegmentationAnnotation, NormalizationState


class PolygonLabelsAnnotationValue(BaseModel):
    points: list[list[float]]
    polygonlabels: list[str]
    closed: bool = True


class PolygonLabelsAnnotation(ImageAnnotationResultABC):
    value: PolygonLabelsAnnotationValue
    type: str = "polygonlabels"

    def to_ir_annotation(self, categories: Categories) -> list[IRSegmentationAnnotation]:
        category = categories.get_or_create(self.value.polygonlabels[0])
        res = IRSegmentationAnnotation(
            category=category,
            state=NormalizationState.NORMALIZED,
            image_width=self.original_width,
            image_height=self.original_height,
        )
        for p in self.value.points:
            res.add_point(p[0] / 100, p[1] / 100)
        res.imported_id = self.id
        return [res]
