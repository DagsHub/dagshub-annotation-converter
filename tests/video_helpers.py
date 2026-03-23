from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from dagshub_annotation_converter.ir.video import IRVideoAnnotationTrack, IRVideoBBoxFrameAnnotation, IRVideoSequence


def sequence_from_annotations(
    annotations: Iterable[IRVideoBBoxFrameAnnotation],
    *,
    filename: Optional[str] = None,
    sequence_length: Optional[int] = None,
) -> IRVideoSequence:
    grouped: Dict[str, List[IRVideoBBoxFrameAnnotation]] = defaultdict(list)
    resolved_filename = filename

    for ann in annotations:
        track_id = ann.imported_id
        if track_id is None:
            raise ValueError("Expected imported_id on test annotation to build IRVideoSequence")
        grouped[track_id].append(ann.model_copy(deep=True))
        if resolved_filename is None and ann.filename:
            resolved_filename = ann.filename

    return IRVideoSequence(
        tracks=[
            IRVideoAnnotationTrack.from_annotations(track_annotations, track_id=track_id)
            for track_id, track_annotations in sorted(
                grouped.items(),
                key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0]),
            )
        ],
        filename=resolved_filename,
        sequence_length=sequence_length,
    )


def flatten_sequence_with_track_ids(sequence: IRVideoSequence) -> List[Tuple[str, IRVideoBBoxFrameAnnotation]]:
    return [
        (track.track_id, ann)
        for track in sequence.tracks
        for ann in track.annotations
    ]


def annotations_by_track_frame(sequence: IRVideoSequence) -> Dict[Tuple[str, int], IRVideoBBoxFrameAnnotation]:
    return {
        (track.track_id, ann.frame_number): ann
        for track in sequence.tracks
        for ann in track.annotations
    }
