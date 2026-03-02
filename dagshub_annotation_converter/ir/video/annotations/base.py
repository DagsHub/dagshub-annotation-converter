from abc import abstractmethod
from typing import Optional

from dagshub_annotation_converter.ir.image.annotations.base import IRAnnotationBase


class IRVideoAnnotationBase(IRAnnotationBase):
    """
    Base class for video annotations with tracking support.
    """

    track_id: int
    """ID of an object (number of the track)."""
    frame_number: int
    """0-based frame index."""
    keyframe: bool = True
    """True - annotation is a keyframe (interpolated up to the next frame), False - single frame annotation"""
    timestamp: Optional[float] = None
    """Time in seconds of the frame"""
    video_width: Optional[int] = None
    """Width of the original video"""
    video_height: Optional[int] = None
    """Height of the original video"""

    @abstractmethod
    def _normalize(self):
        ...

    @abstractmethod
    def _denormalize(self):
        ...
