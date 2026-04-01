"""Video annotation IR models."""

from dagshub_annotation_converter.ir.video.annotations.base import IRVideoFrameAnnotationBase
from dagshub_annotation_converter.ir.video.annotations.bbox import IRVideoBBoxFrameAnnotation

__all__ = [
    IRVideoFrameAnnotationBase.__name__,
    IRVideoBBoxFrameAnnotation.__name__,
]
