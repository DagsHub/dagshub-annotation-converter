from pydantic import BaseModel

from dagshub_annotation_converter.schema.label_studio.abc import ImageAnnotationResultABC

class KeyPointLabelsAnnotationValue(BaseModel):
    x: float
    y: float
    width: float = 1.0
    keypointlabels: list[str]

class KeyPointLabelsAnnotation(ImageAnnotationResultABC):
    value: KeyPointLabelsAnnotationValue
    type: str = "keypointlabels"
