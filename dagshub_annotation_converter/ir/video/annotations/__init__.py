"""Video annotation IR models."""

from dagshub_annotation_converter.ir.video.annotations.base import IRVideoAnnotationBase
from dagshub_annotation_converter.ir.video.annotations.bbox import IRVideoBBoxAnnotation

__all__ = [
    "IRVideoAnnotationBase",
    "IRVideoBBoxAnnotation",
]
