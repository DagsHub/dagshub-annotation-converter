import uuid
from typing import Any, Dict, List, Optional, Sequence

from pydantic import Field

from dagshub_annotation_converter.formats.label_studio.base import AnnotationResultABC
from dagshub_annotation_converter.ir.image.annotations.base import IRAnnotationBase
from dagshub_annotation_converter.ir.video import (
    CoordinateStyle,
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
)
from dagshub_annotation_converter.util.pydantic_util import ParentModel


def _coerce_bool_like(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f", ""}:
            return False
    return bool(value)


def _coerce_float_like(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


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

    def to_ir_annotation(self) -> Sequence[IRVideoBBoxFrameAnnotation]:
        return self.to_ir_annotations()

    @staticmethod
    def from_ir_annotation(ir_annotation: IRAnnotationBase) -> Sequence["VideoRectangleAnnotation"]:
        raise NotImplementedError(
            "VideoRectangleAnnotation requires multiple IR annotations per track. Use from_ir_annotations() instead."
        )

    def to_ir_annotations(self) -> List[IRVideoBBoxFrameAnnotation]:
        return self.to_ir_track().to_annotations()

    def to_ir_track(self) -> IRVideoAnnotationTrack:
        """
        Convert VideoRectangle to an IR video track.

        Label Studio uses 1-based frame numbering, IR uses 0-based.
        """
        label = self.value.labels[0] if self.value.labels else "object"

        frame_base = 0 if any(item.frame == 0 for item in self.value.sequence) else 1

        annotations = []
        for seq_item in self.value.sequence:
            if seq_item.frame < frame_base:
                if frame_base == 0:
                    raise ValueError("Frame numbers must be 0-based (>= 0)")
                raise ValueError("Frame numbers must be 1-based (>= 1)")
            if not (
                0.0 <= seq_item.x <= 100.0
                and 0.0 <= seq_item.y <= 100.0
                and 0.0 <= seq_item.width <= 100.0
                and 0.0 <= seq_item.height <= 100.0
            ):
                raise ValueError("Coordinates must be percentages in [0, 100]")

            extra = seq_item.__pydantic_extra__ or {}
            outside_extra = extra.get("outside")
            visibility_extra = extra.get("visibility")

            outside = _coerce_bool_like(outside_extra) if outside_extra is not None else False
            visibility = 0.0 if outside else 1.0
            parsed_visibility = _coerce_float_like(visibility_extra)
            if parsed_visibility is not None:
                visibility = parsed_visibility

            keyframe = bool(seq_item.enabled)

            ann = IRVideoBBoxFrameAnnotation(
                frame_number=seq_item.frame - frame_base,
                keyframe=keyframe,
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

        return IRVideoAnnotationTrack.from_annotations(annotations, id=self.id)

    @staticmethod
    def from_ir_annotations(
        ir_annotations: List[IRVideoBBoxFrameAnnotation],
        frames_count: Optional[int] = None,
    ) -> "VideoRectangleAnnotation":
        if not ir_annotations:
            raise ValueError("Cannot create VideoRectangleAnnotation from empty annotations")
        track_id = ir_annotations[0].imported_id or "0"
        track = IRVideoAnnotationTrack.from_annotations(ir_annotations, id=track_id)
        return VideoRectangleAnnotation.from_ir_track(track, frames_count=frames_count)

    @staticmethod
    def from_ir_track(
        track: IRVideoAnnotationTrack,
        frames_count: Optional[int] = None,
    ) -> "VideoRectangleAnnotation":
        """Create a VideoRectangleAnnotation from IR video annotations for a single track."""
        if not track.annotations:
            raise ValueError("Cannot create VideoRectangleAnnotation from empty track")

        first = track.annotations[0]
        ls_id = track.id
        label = first.ensure_has_one_category()

        sorted_anns = sorted(track.annotations, key=lambda a: a.frame_number)

        def _is_visible(annotation: IRVideoBBoxFrameAnnotation) -> bool:
            return annotation.visibility > 0.0

        has_cvat_style_metadata = any("z_order" in ann.meta for ann in sorted_anns)
        has_keyframes = any(ann.keyframe for ann in sorted_anns)
        effective_anns = []
        for idx, ann in enumerate(sorted_anns):
            next_ann = sorted_anns[idx + 1] if idx + 1 < len(sorted_anns) else None
            keep_nonkey_pre_outside = next_ann is not None and not _is_visible(next_ann)
            # CVAT interpolation exports often include dense keyframe=0 rows.
            # Keep only true keyframes/outside boundaries when keyframes are present.
            if (
                has_cvat_style_metadata
                and has_keyframes
                and not ann.keyframe
                and _is_visible(ann)
                and not keep_nonkey_pre_outside
            ):
                continue
            effective_anns.append(ann)

        sequence = []
        for idx, ann in enumerate(effective_anns):
            if ann.coordinate_style == CoordinateStyle.DENORMALIZED:
                if ann.video_width is None or ann.video_height is None:
                    raise ValueError(
                        f"Cannot normalize annotation at frame {ann.frame_number} without image dimensions"
                    )
                ann = ann.normalized()

            if not _is_visible(ann):
                continue

            enabled = bool(ann.keyframe)
            if enabled:
                next_ann = next(iter(effective_anns[idx + 1 :]), None)
                if next_ann is not None:
                    # CVAT uses an outside control point on the next frame to mark stop.
                    if not _is_visible(next_ann):
                        enabled = next_ann.frame_number > ann.frame_number + 1
                    elif next_ann.frame_number == ann.frame_number + 1 and next_ann.keyframe:
                        enabled = False

            seq_item = VideoRectangleSequenceItem(
                frame=ann.frame_number + 1,  # Convert 0-based to 1-based
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
