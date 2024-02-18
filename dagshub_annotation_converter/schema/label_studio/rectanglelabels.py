from pydantic import BaseModel

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
