import hashlib
import uuid
from typing import List, Optional, Dict, Any, Sequence

from pydantic import Field

from dagshub_annotation_converter.formats.label_studio.base import AnnotationResultABC
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.ir.image.annotations.base import IRAnnotationBase
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase
from dagshub_annotation_converter.util.pydantic_util import ParentModel


class VideoRectangleSequenceItem(ParentModel):
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
    time: Optional[float] = None
    rotation: float = 0.0


class VideoRectangleValue(ParentModel):
    sequence: List[VideoRectangleSequenceItem]
    labels: List[str]
    framesCount: Optional[int] = None
    duration: Optional[float] = None


class VideoRectangleAnnotation(AnnotationResultABC):
    """
    Label Studio VideoRectangle annotation for video object tracking.

    Each VideoRectangle represents a single tracked object across multiple frames.
    Coordinates are stored as percentages (0-100).
    """

    id: str = Field(default_factory=lambda: f"track_{uuid.uuid4().hex[:8]}")
    type: str = "videorectangle"
    value: VideoRectangleValue
    original_width: Optional[int] = None
    original_height: Optional[int] = None
    from_name: str = "box"
    to_name: str = "video"
    origin: str = "manual"
    meta: Optional[Dict[str, Any]] = None

    def to_ir_annotation(self) -> Sequence[IRAnnotationBase]:
        return self.to_ir_annotations()

    @staticmethod
    def from_ir_annotation(ir_annotation: IRImageAnnotationBase) -> Sequence["VideoRectangleAnnotation"]:
        raise NotImplementedError(
            "VideoRectangleAnnotation requires multiple IR annotations per track. "
            "Use from_ir_annotations() instead."
        )

    def to_ir_annotations(self) -> List[IRVideoBBoxAnnotation]:
        """
        Convert VideoRectangle to IR video annotations.

        Label Studio uses 1-based frame numbering, IR uses 0-based.
        """
        if self.meta and "original_track_id" in self.meta:
            track_id = self.meta["original_track_id"]
        else:
            # Deterministic track_id from id (hash() is randomized per PYTHONHASHSEED)
            track_id = int(hashlib.md5(self.id.encode("utf-8")).hexdigest()[:8], 16) % (2**31)

        label = self.value.labels[0] if self.value.labels else "object"

        annotations = []
        for seq_item in self.value.sequence:
            if seq_item.frame < 1:
                raise ValueError("Frame numbers must be 1-based (>= 1)")
            if not (
                0.0 <= seq_item.x <= 100.0 and
                0.0 <= seq_item.y <= 100.0 and
                0.0 <= seq_item.width <= 100.0 and
                0.0 <= seq_item.height <= 100.0
            ):
                raise ValueError("Coordinates must be percentages in [0, 100]")
            
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
        """Create a VideoRectangleAnnotation from IR video annotations for a single track."""
        if not ir_annotations:
            raise ValueError("Cannot create VideoRectangleAnnotation from empty list")

        first = ir_annotations[0]
        track_id = first.track_id
        if any(ann.track_id != track_id for ann in ir_annotations):
            raise ValueError("All annotations must share the same track_id")
        
        ls_id = first.meta.get("ls_id", f"track_{track_id}") if first.meta else f"track_{track_id}"
        label = first.ensure_has_one_category()

        sorted_anns = sorted(ir_annotations, key=lambda a: a.frame_number)

        sequence = []
        for ann in sorted_anns:
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
