import hashlib
import uuid
from typing import Any, Dict, List, Optional, Sequence

from pydantic import Field

from dagshub_annotation_converter.formats.label_studio.base import AnnotationResultABC
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase
from dagshub_annotation_converter.ir.image.annotations.base import IRAnnotationBase
from dagshub_annotation_converter.ir.video import CoordinateStyle, IRVideoBBoxAnnotation
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

    def to_ir_annotation(self) -> Sequence[IRAnnotationBase]:
        return self.to_ir_annotations()

    @staticmethod
    def from_ir_annotation(ir_annotation: IRImageAnnotationBase) -> Sequence["VideoRectangleAnnotation"]:
        raise NotImplementedError(
            "VideoRectangleAnnotation requires multiple IR annotations per track. Use from_ir_annotations() instead."
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

            meta: Dict[str, Any] = {"ls_id": self.id}
            if self.value.framesCount is not None and self.value.framesCount > 0:
                meta["ls_frames_count"] = self.value.framesCount

            keyframe = bool(seq_item.enabled)

            ann = IRVideoBBoxAnnotation(
                track_id=track_id,
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
                meta=meta,
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

        frames_count_values = {
            int(ann.meta["ls_frames_count"]) for ann in sorted_anns if isinstance(ann.meta.get("ls_frames_count"), int)
        }
        frames_count = max(frames_count_values) if frames_count_values else None

        def _is_visible(annotation: IRVideoBBoxAnnotation) -> bool:
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
                if not ann.meta.get("ls_id"):
                    continue
                seq_item_kwargs = dict(
                    frame=ann.frame_number + 1,
                    x=ann.left * 100.0,
                    y=ann.top * 100.0,
                    width=ann.width * 100.0,
                    height=ann.height * 100.0,
                    rotation=ann.rotation,
                    enabled=False,
                    time=ann.timestamp,
                )
                seq_item = VideoRectangleSequenceItem(**seq_item_kwargs)
                sequence.append(seq_item)
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
                framesCount=frames_count,
            ),
            meta={"original_track_id": track_id},
        )
