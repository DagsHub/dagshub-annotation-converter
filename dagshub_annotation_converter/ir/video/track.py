from typing import List, Optional, Sequence

from dagshub_annotation_converter.ir.common import CoordinateStyle
from dagshub_annotation_converter.ir.base import IRAnnotationCollection
from dagshub_annotation_converter.ir.video.annotations.base import IRVideoFrameAnnotationBase


class IRVideoAnnotationTrack(IRAnnotationCollection):
    """A single object track across video frames.

    Assumptions:
      - A track follows **one object** throughout the video.
      - All annotations in the track share the same category / label.
      - ``annotations`` is a list of per-frame snapshots (bbox, keypoint, etc.)
        ordered by ``frame_number``.  Not every frame needs an entry — gaps are
        filled by interpolation at export time where the format requires it.
      - ``object_id`` is a stable identifier for the object within the sequence.
        It is propagated to each annotation's ``imported_id`` on construction.
      - Coordinate style (normalized vs. denormalized) may vary across
        annotations; use :meth:`normalized` / :meth:`denormalized` to ensure
        a uniform style before export.
    """

    annotations: List[IRVideoFrameAnnotationBase]

    def resolved_video_width(self) -> Optional[int]:
        for ann in self.annotations:
            if ann.video_width is not None and ann.video_width > 0:
                return ann.video_width
        return None

    def resolved_video_height(self) -> Optional[int]:
        for ann in self.annotations:
            if ann.video_height is not None and ann.video_height > 0:
                return ann.video_height
        return None

    @classmethod
    def from_annotations(
        cls,
        annotations: Sequence[IRVideoFrameAnnotationBase],
        object_id: str,
    ) -> "IRVideoAnnotationTrack":
        if not annotations:
            raise ValueError("Cannot create IRVideoAnnotationTrack from empty annotations")

        copied_annotations = sorted(
            (ann.model_copy(deep=True) for ann in annotations),
            key=lambda a: a.frame_number,
        )
        for ann in copied_annotations:
            ann.imported_id = object_id
        return cls(
            object_id=object_id,
            annotations=copied_annotations,
        )

    def _with_resolved_dimensions(
        self,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> "IRVideoAnnotationTrack":
        resolved_width = video_width if video_width is not None and video_width > 0 else self.resolved_video_width()
        resolved_height = (
            video_height if video_height is not None and video_height > 0 else self.resolved_video_height()
        )

        for ann in self.annotations:
            ann.imported_id = self.object_id
            if ann.video_width is None and resolved_width is not None:
                ann.video_width = resolved_width
            if ann.video_height is None and resolved_height is not None:
                ann.video_height = resolved_height
        return self

    def normalized(
        self,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> "IRVideoAnnotationTrack":
        track = self.model_copy(deep=True)
        track._with_resolved_dimensions(video_width, video_height)
        normalized_annotations: List[IRVideoFrameAnnotationBase] = []
        for ann in track.annotations:
            if ann.coordinate_style == CoordinateStyle.DENORMALIZED:
                ann = ann.normalized()
            else:
                ann = ann.model_copy(deep=True)
            normalized_annotations.append(ann)
        return IRVideoAnnotationTrack(object_id=track.object_id, annotations=normalized_annotations)

    def denormalized(
        self,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
    ) -> "IRVideoAnnotationTrack":
        track = self.model_copy(deep=True)
        track._with_resolved_dimensions(video_width, video_height)
        denormalized_annotations: List[IRVideoFrameAnnotationBase] = []
        for ann in track.annotations:
            if ann.coordinate_style == CoordinateStyle.NORMALIZED:
                ann = ann.denormalized()
            else:
                ann = ann.model_copy(deep=True)
            denormalized_annotations.append(ann)
        return IRVideoAnnotationTrack(object_id=track.object_id, annotations=denormalized_annotations)

    def to_annotations(self) -> List[IRVideoFrameAnnotationBase]:
        annotations: List[IRVideoFrameAnnotationBase] = []
        for ann in self.annotations:
            copied = ann.model_copy(deep=True)
            copied.imported_id = self.object_id
            annotations.append(copied)
        return annotations
