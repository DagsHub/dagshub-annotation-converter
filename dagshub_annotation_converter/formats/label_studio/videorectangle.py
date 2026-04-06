import uuid
from typing import Any, Dict, List, Optional, Sequence

from pydantic import Field

from dagshub_annotation_converter.formats.label_studio.base import AnnotationResultABC
from dagshub_annotation_converter.ir.video import (
    CoordinateStyle,
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
)
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
    """Whether interpolation is enabled from this keyframe onward in Label Studio."""
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

    def to_ir_annotation(self) -> Sequence[IRVideoAnnotationTrack]:
        return [self.to_ir_track()]

    @staticmethod
    def from_ir_annotation(ir_annotation: IRVideoAnnotationTrack) -> Sequence["VideoRectangleAnnotation"]:
        return [VideoRectangleAnnotation.from_ir_track(ir_annotation)]

    def to_ir_track(self) -> IRVideoAnnotationTrack:
        if not self.value.labels:
            raise ValueError("VideoRectangleAnnotation has no labels")
        label = self.value.labels[0]

        annotations = []
        for seq_item in self.value.sequence:
            if not (
                0.0 <= seq_item.x <= 100.0
                and 0.0 <= seq_item.y <= 100.0
                and 0.0 <= seq_item.width <= 100.0
                and 0.0 <= seq_item.height <= 100.0
            ):
                raise ValueError("Coordinates must be percentages in [0, 100]")

            extra = seq_item.__pydantic_extra__ or {}
            visibility = extra.get("visibility", 1.0)
            if not (0.0 <= visibility <= 1.0):
                raise ValueError(f"Visibility must be in [0, 1], got {visibility}")

            ann = IRVideoBBoxFrameAnnotation(
                frame_number=seq_item.frame - 1,
                keyframe=seq_item.enabled,
                left=seq_item.x / 100.0,
                top=seq_item.y / 100.0,
                width=seq_item.width / 100.0,
                height=seq_item.height / 100.0,
                rotation=seq_item.rotation,
                video_width=self.original_width,
                video_height=self.original_height,
                categories={label: 1.0},
                coordinate_style=CoordinateStyle.NORMALIZED,
                timestamp=seq_item.time,
                visibility=visibility,
            )
            annotations.append(ann)

        return IRVideoAnnotationTrack.from_annotations(annotations, object_id=self.id)

    @staticmethod
    def from_ir_track(
        track: IRVideoAnnotationTrack,
        frames_count: Optional[int] = None,
    ) -> "VideoRectangleAnnotation":
        if not track.annotations:
            raise ValueError("Cannot create VideoRectangleAnnotation from empty track")

        if not all(isinstance(ann, IRVideoBBoxFrameAnnotation) for ann in track.annotations):
            raise ValueError("All annotations in the track must be IRVideoBBoxFrameAnnotation")

        first = track.annotations[0]
        ls_id = track.object_id
        label = first.ensure_has_one_category()

        track = track.normalized()
        sorted_anns: List[IRVideoBBoxFrameAnnotation] = sorted(track.annotations, key=lambda a: a.frame_number)  # type: ignore[assignment]

        sequence = []
        for idx, ann in enumerate(sorted_anns):
            if not ann.is_visible:
                continue

            enabled = ann.keyframe
            next_ann = sorted_anns[idx + 1] if idx + 1 < len(sorted_anns) else None
            if enabled and next_ann is not None and not next_ann.is_visible:
                # Label Studio only stops interpolation at the next serialized
                # keyframe. Invisible IR frames are omitted from the sequence,
                # so disable interpolation on the last visible keyframe before
                # a gap to avoid drawing boxes through absent-object frames.
                enabled = False

            seq_item = VideoRectangleSequenceItem(
                frame=ann.frame_number + 1,
                x=ann.left * 100.0,
                y=ann.top * 100.0,
                width=ann.width * 100.0,
                height=ann.height * 100.0,
                rotation=ann.rotation,
                enabled=enabled,
                time=ann.timestamp,
            )
            sequence.append(seq_item)

        return VideoRectangleAnnotation(
            id=ls_id,
            original_width=first.video_width,
            original_height=first.video_height,
            value=VideoRectangleValue(
                sequence=sequence,
                labels=[label],
                framesCount=frames_count if frames_count is not None and frames_count > 0 else None,
            ),
        )
