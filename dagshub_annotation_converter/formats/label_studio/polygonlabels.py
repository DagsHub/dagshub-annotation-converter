from pydantic import BaseModel


from dagshub_annotation_converter.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.ir.image import IRSegmentationAnnotation, NormalizationState


class PolygonLabelsAnnotationValue(BaseModel):
    points: list[list[float]]
    polygonlabels: list[str]
    closed: bool = True


class PolygonLabelsAnnotation(ImageAnnotationResultABC):
    value: PolygonLabelsAnnotationValue
    type: str = "polygonlabels"

    def to_ir_annotation(self) -> list[IRSegmentationAnnotation]:
        res = IRSegmentationAnnotation(
            category=self.value.polygonlabels[0],
            state=NormalizationState.NORMALIZED,
            image_width=self.original_width,
            image_height=self.original_height,
        )
        for p in self.value.points:
            res.add_point(p[0] / 100, p[1] / 100)
        res.imported_id = self.id
        return [res]
