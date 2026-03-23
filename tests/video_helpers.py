from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from dagshub_annotation_converter.ir.video import IRVideoAnnotationTrack, IRVideoBBoxFrameAnnotation, IRVideoSequence


def _strip_legacy_video_fields(ann: IRVideoBBoxFrameAnnotation) -> IRVideoBBoxFrameAnnotation:
    copied = ann.model_copy(deep=True)
    extras = dict(copied.__pydantic_extra__ or {})
    extras.pop("track_id", None)
    extras.pop("sequence_length", None)
    copied.__pydantic_extra__ = extras or None
    return copied


def sequence_from_annotations(
    annotations: Iterable[IRVideoBBoxFrameAnnotation],
    *,
    filename: Optional[str] = None,
    sequence_length: Optional[int] = None,
) -> IRVideoSequence:
    grouped: Dict[str, List[IRVideoBBoxFrameAnnotation]] = defaultdict(list)
    resolved_filename = filename

    for ann in annotations:
        track_id = getattr(ann, "track_id", None)
        if track_id is None:
            raise ValueError("Expected legacy test annotation with track_id to build IRVideoSequence")
        grouped[str(track_id)].append(_strip_legacy_video_fields(ann))
        if resolved_filename is None and ann.filename:
            resolved_filename = ann.filename

    return IRVideoSequence(
        tracks=[
            IRVideoAnnotationTrack.from_annotations(track_annotations, id=track_id)
            for track_id, track_annotations in sorted(
                grouped.items(),
                key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]),
            )
        ],
        filename=resolved_filename,
        sequence_length=sequence_length,
    )


def flatten_sequence_with_track_ids(sequence: IRVideoSequence) -> List[Tuple[int, IRVideoBBoxFrameAnnotation]]:
    return [
        (track.track_id, ann)
        for track in sequence.tracks
        for ann in track.annotations
    ]


def annotations_by_track_frame(sequence: IRVideoSequence) -> Dict[Tuple[int, int], IRVideoBBoxFrameAnnotation]:
    return {
        (track.track_id, ann.frame_number): ann
        for track in sequence.tracks
        for ann in track.annotations
    }
