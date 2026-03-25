from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from dagshub_annotation_converter.ir.video.annotations.base import IRVideoFrameAnnotationBase
from dagshub_annotation_converter.ir.video.track import IRVideoAnnotationTrack
from dagshub_annotation_converter.util.pydantic_util import ParentModel


class IRVideoSequence(ParentModel):
    tracks: List[IRVideoAnnotationTrack]
    filename: Optional[str] = None
    sequence_length: Optional[int] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None

    @classmethod
    def from_annotations(
        cls,
        annotations: Sequence[IRVideoFrameAnnotationBase],
        filename: Optional[str] = None,
    ) -> "IRVideoSequence":
        if not annotations:
            raise ValueError("Cannot create IRVideoSequence from empty annotations")

        grouped: Dict[str, List[IRVideoFrameAnnotationBase]] = {}
        resolved_filename = filename
        resolved_sequence_length = None

        for ann in annotations:
            track_id = ann.imported_id or "0"
            if track_id not in grouped:
                grouped[track_id] = []
            grouped[track_id].append(ann)
            if resolved_filename is None and ann.filename:
                resolved_filename = ann.filename
            if resolved_sequence_length is None:
                ann_sequence_length = getattr(ann, "sequence_length", None)
                if ann_sequence_length:
                    resolved_sequence_length = ann_sequence_length

        tracks = [
            IRVideoAnnotationTrack.from_annotations(track_annotations, track_id=track_id)
            for track_id, track_annotations in grouped.items()
        ]
        return cls(
            tracks=tracks,
            filename=resolved_filename,
            sequence_length=resolved_sequence_length,
        )

    def iter_track_annotations(self) -> Iterator[Tuple[IRVideoAnnotationTrack, IRVideoFrameAnnotationBase]]:
        for track in self.tracks:
            for ann in track.annotations:
                yield track, ann

    def to_annotations(self) -> List[IRVideoFrameAnnotationBase]:
        annotations: List[IRVideoFrameAnnotationBase] = []
        for track in self.tracks:
            for ann in track.to_annotations():
                if ann.filename is None and self.filename is not None:
                    ann.filename = self.filename
                if ann.video_width is None and self.video_width is not None:
                    ann.video_width = self.video_width
                if ann.video_height is None and self.video_height is not None:
                    ann.video_height = self.video_height
                if getattr(ann, "sequence_length", None) is None and self.sequence_length is not None:
                    ann.sequence_length = self.sequence_length
                annotations.append(ann)
        return annotations

    def annotations_by_frame(self) -> Dict[int, List[Tuple[IRVideoAnnotationTrack, IRVideoFrameAnnotationBase]]]:
        by_frame: Dict[int, List[Tuple[IRVideoAnnotationTrack, IRVideoFrameAnnotationBase]]] = {}
        for track, ann in self.iter_track_annotations():
            if ann.frame_number not in by_frame:
                by_frame[ann.frame_number] = []
            by_frame[ann.frame_number].append((track, ann))
        return by_frame

    def resolved_video_width(self) -> Optional[int]:
        if self.video_width is not None and self.video_width > 0:
            return self.video_width
        for _, ann in self.iter_track_annotations():
            if ann.video_width is not None and ann.video_width > 0:
                return ann.video_width
        return None

    def resolved_video_height(self) -> Optional[int]:
        if self.video_height is not None and self.video_height > 0:
            return self.video_height
        for _, ann in self.iter_track_annotations():
            if ann.video_height is not None and ann.video_height > 0:
                return ann.video_height
        return None

    def resolved_sequence_length(self) -> Optional[int]:
        if self.sequence_length is not None and self.sequence_length > 0:
            return self.sequence_length
        max_frame_number: Optional[int] = None
        for _, ann in self.iter_track_annotations():
            max_frame_number = ann.frame_number if max_frame_number is None else max(max_frame_number, ann.frame_number)
        if max_frame_number is None:
            return None
        return max_frame_number + 1
