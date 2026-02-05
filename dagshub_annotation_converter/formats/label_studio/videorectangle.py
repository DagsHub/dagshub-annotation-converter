"""Label Studio VideoRectangle format for video object tracking."""

import uuid
from typing import List, Optional, Dict, Any, Sequence

from pydantic import Field

from dagshub_annotation_converter.formats.label_studio.base import AnnotationResultABC
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.ir.image.annotations.base import IRAnnotationBase
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase
from dagshub_annotation_converter.util.pydantic_util import ParentModel


class VideoRectangleSequenceItem(ParentModel):
    """A single frame in a VideoRectangle sequence."""
    
    frame: int
    """Frame number (1-based)."""
    
    x: float
    """X coordinate as percentage (0-100)."""
    
    y: float
    """Y coordinate as percentage (0-100)."""
    
    width: float
    """Width as percentage (0-100)."""
    
    height: float
    """Height as percentage (0-100)."""
    
    enabled: bool = True
    """Whether the frame is active."""
    
    time: Optional[float] = None
    """Timestamp in seconds."""
    
    rotation: float = 0.0
    """Rotation in degrees."""


class VideoRectangleValue(ParentModel):
    """Value object for VideoRectangle annotation."""
    
    sequence: List[VideoRectangleSequenceItem]
    """List of keyframes defining the track."""
    
    labels: List[str]
    """Object class labels."""
    
    framesCount: Optional[int] = None
    """Total number of frames in the video."""
    
    duration: Optional[float] = None
    """Duration of the video in seconds."""


class VideoRectangleAnnotation(AnnotationResultABC):
    """
    Label Studio VideoRectangle annotation for video object tracking.
    
    Each VideoRectangle represents a single tracked object across multiple frames.
    Coordinates are stored as percentages (0-100).
    """
    
    id: str = Field(default_factory=lambda: f"track_{uuid.uuid4().hex[:8]}")
    """Unique track identifier."""
    
    type: str = "videorectangle"
    """Annotation type (always 'videorectangle')."""
    
    value: VideoRectangleValue
    """Annotation value containing sequence and labels."""
    
    original_width: int
    """Video frame width in pixels."""
    
    original_height: int
    """Video frame height in pixels."""
    
    from_name: str = "box"
    """Name of the control tag."""
    
    to_name: str = "video"
    """Name of the video element."""
    
    origin: str = "manual"
    """Annotation origin."""
    
    meta: Optional[Dict[str, Any]] = None
    """Additional metadata."""

    def to_ir_annotation(self) -> Sequence[IRAnnotationBase]:
        """
        ABC implementation - delegates to to_ir_annotations().
        """
        return self.to_ir_annotations()

    @staticmethod
    def from_ir_annotation(ir_annotation: IRImageAnnotationBase) -> Sequence["VideoRectangleAnnotation"]:
        """
        ABC implementation - not directly applicable for video annotations.
        Use from_ir_annotations() with a list of video annotations instead.
        """
        raise NotImplementedError(
            "VideoRectangleAnnotation requires multiple IR annotations per track. "
            "Use from_ir_annotations() instead."
        )

    def to_ir_annotations(self) -> List[IRVideoBBoxAnnotation]:
        """
        Convert VideoRectangle to a list of IR video annotations.
        
        Each sequence item becomes a separate IRVideoBBoxAnnotation.
        Label Studio uses 1-based frame numbering, so we convert to 0-based for IR.
        
        Returns:
            List of IRVideoBBoxAnnotation, one per frame in the sequence
        """
        # Try to extract original track_id from meta, otherwise derive from id
        if self.meta and "original_track_id" in self.meta:
            track_id = self.meta["original_track_id"]
        else:
            # Derive track_id from annotation id (hash to int)
            track_id = hash(self.id) % (2**31)  # Keep it positive and bounded
        
        label = self.value.labels[0] if self.value.labels else "object"
        
        annotations = []
        for seq_item in self.value.sequence:
            # Convert percentage (0-100) to normalized (0-1)
            # Label Studio uses 1-based frames, IR uses 0-based
            ann = IRVideoBBoxAnnotation(
                track_id=track_id,
                frame_number=seq_item.frame - 1,  # Convert 1-based to 0-based
                left=seq_item.x / 100.0,
                top=seq_item.y / 100.0,
                width=seq_item.width / 100.0,
                height=seq_item.height / 100.0,
                rotation=seq_item.rotation,
                image_width=self.original_width,
                image_height=self.original_height,
                categories={label: 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                timestamp=seq_item.time,
                visibility=1.0 if seq_item.enabled else 0.0,
                meta={"ls_id": self.id},
            )
            ann.imported_id = self.id
            annotations.append(ann)
        
        return annotations

    @staticmethod
    def from_ir_annotations(ir_annotations: List[IRVideoBBoxAnnotation]) -> "VideoRectangleAnnotation":
        """
        Create a VideoRectangleAnnotation from a list of IR video annotations.
        
        All annotations should belong to the same track (same track_id).
        
        Args:
            ir_annotations: List of video bbox annotations for a single track
            
        Returns:
            VideoRectangleAnnotation combining all frames
        """
        if not ir_annotations:
            raise ValueError("Cannot create VideoRectangleAnnotation from empty list")
        
        # Use first annotation for common fields
        first = ir_annotations[0]
        
        # Get track ID for the annotation ID
        track_id = first.track_id
        ls_id = first.meta.get("ls_id", f"track_{track_id}")
        
        # Get label
        label = first.ensure_has_one_category()
        
        # Sort by frame number
        sorted_anns = sorted(ir_annotations, key=lambda a: a.frame_number)
        
        # Build sequence
        sequence = []
        for ann in sorted_anns:
            # Normalize if needed
            if ann.coordinate_style == CoordinateStyle.DENORMALIZED:
                ann = ann.normalized()
            
            # Convert normalized (0-1) to percentage (0-100)
            # IR uses 0-based frames, Label Studio uses 1-based
            seq_item = VideoRectangleSequenceItem(
                frame=ann.frame_number + 1,  # Convert 0-based to 1-based
                x=ann.left * 100.0,
                y=ann.top * 100.0,
                width=ann.width * 100.0,
                height=ann.height * 100.0,
                rotation=ann.rotation,
                enabled=ann.visibility > 0.5,
                time=ann.timestamp,
            )
            sequence.append(seq_item)
        
        # Store original track_id in meta for roundtrip preservation
        return VideoRectangleAnnotation(
            id=ls_id,
            original_width=first.image_width,
            original_height=first.image_height,
            value=VideoRectangleValue(
                sequence=sequence,
                labels=[label],
            ),
            meta={"original_track_id": track_id},
        )
