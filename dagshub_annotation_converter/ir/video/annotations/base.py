from abc import abstractmethod
from typing import Optional

from dagshub_annotation_converter.ir.image.annotations.base import IRImageAnnotationBase


class IRVideoAnnotationBase(IRImageAnnotationBase):
    """
    Base class for video annotations with tracking support.

    - track_id: Unique identifier for an object across frames
    - frame_number: 0-based frame index
    - timestamp: Optional timestamp in seconds
    - video_path: Optional reference to source video
    """

    track_id: int
    frame_number: int
    """0-based frame index."""
    timestamp: Optional[float] = None
    video_path: Optional[str] = None

    @abstractmethod
    def _normalize(self):
        ...

    @abstractmethod
    def _denormalize(self):
        ...
