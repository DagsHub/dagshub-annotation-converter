"""Base class for video annotations with tracking support."""

from abc import abstractmethod
from typing import Optional

from dagshub_annotation_converter.ir.image.annotations.base import IRImageAnnotationBase


class IRVideoAnnotationBase(IRImageAnnotationBase):
    """
    Base class for all video annotations with tracking support.
    
    Extends IRImageAnnotationBase with video-specific fields:
    - track_id: Unique identifier for an object across frames
    - frame_number: Frame index (0-based; first frame is 0)
    - timestamp: Optional timestamp in seconds
    - video_path: Optional reference to source video
    """

    track_id: int
    """Unique identifier for tracking an object across frames."""
    
    frame_number: int
    """Frame number (0-based index; first frame is 0)."""
    
    timestamp: Optional[float] = None
    """Timestamp in seconds (optional)."""
    
    video_path: Optional[str] = None
    """Reference to the source video file (optional)."""

    @abstractmethod
    def _normalize(self):
        """
        Every annotation should implement this to normalize itself.
        """
        ...

    @abstractmethod
    def _denormalize(self):
        """
        Every annotation should implement this to denormalize itself.
        """
        ...
