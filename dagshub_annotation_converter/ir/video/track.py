import hashlib
import re
from typing import List, Sequence

from dagshub_annotation_converter.ir.video.annotations.base import IRVideoFrameAnnotationBase
from dagshub_annotation_converter.util.pydantic_util import ParentModel


_NUMERIC_TRACK_ID_RE = re.compile(r"^(?:track_)?(?P<track_id>\d+)$")


def track_id_from_identifier(identifier: str) -> int:
    match = _NUMERIC_TRACK_ID_RE.fullmatch(identifier.strip())
    if match is not None:
        return int(match.group("track_id"))
    return int(hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def track_identifier_from_id(track_id: int) -> str:
    return str(track_id)


class IRVideoAnnotationTrack(ParentModel):
    id: str
    annotations: List[IRVideoFrameAnnotationBase]

    @property
    def track_id(self) -> int:
        return track_id_from_identifier(self.id)

    @classmethod
    def from_annotations(
        cls,
        annotations: Sequence[IRVideoFrameAnnotationBase],
        id: str,
    ) -> "IRVideoAnnotationTrack":
        if not annotations:
            raise ValueError("Cannot create IRVideoAnnotationTrack from empty annotations")

        copied_annotations = [ann.model_copy(deep=True) for ann in annotations]
        for ann in copied_annotations:
            ann.imported_id = id
        return cls(
            id=id,
            annotations=copied_annotations,
        )

    def to_annotations(self) -> List[IRVideoFrameAnnotationBase]:
        annotations: List[IRVideoFrameAnnotationBase] = []
        for ann in self.annotations:
            copied = ann.model_copy(deep=True)
            copied.imported_id = self.id
            annotations.append(copied)
        return annotations
