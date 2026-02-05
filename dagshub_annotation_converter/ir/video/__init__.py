"""
Video Intermediate Representation (IR) for video annotation formats.

This module provides data structures for representing video annotations
with tracking support, including track IDs and frame numbers.
"""

from dagshub_annotation_converter.ir.image.common import CoordinateStyle
from dagshub_annotation_converter.ir.video.annotations.base import IRVideoAnnotationBase
from dagshub_annotation_converter.ir.video.annotations.bbox import IRVideoBBoxAnnotation

__all__ = [
    "CoordinateStyle",
    "IRVideoAnnotationBase",
    "IRVideoBBoxAnnotation",
]
