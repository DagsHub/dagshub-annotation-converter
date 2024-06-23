import datetime
import random
from typing import Any, Sequence, Annotated, Type, Optional, Union

from pydantic import BaseModel, SerializeAsAny, Field, BeforeValidator

from dagshub_annotation_converter.refactored.formats.label_studio.base import AnnotationResultABC
from dagshub_annotation_converter.refactored.formats.label_studio.keypointlabels import KeyPointLabelsAnnotation
from dagshub_annotation_converter.refactored.formats.label_studio.polygonlabels import PolygonLabelsAnnotation
from dagshub_annotation_converter.refactored.formats.label_studio.rectanglelabels import RectangleLabelsAnnotation

task_lookup: dict[str, Type[AnnotationResultABC]] = {
    "polygonlabels": PolygonLabelsAnnotation,
    "rectanglelabels": RectangleLabelsAnnotation,
    "keypointlabels": KeyPointLabelsAnnotation,
}


def ls_annotation_validator(v: Any) -> list[AnnotationResultABC]:
    assert isinstance(v, list)

    annotations: list[AnnotationResultABC] = []

    for raw_annotation in v:
        assert isinstance(raw_annotation, dict)
        assert "type" in raw_annotation
        assert raw_annotation["type"] in task_lookup

        ann_class = task_lookup[raw_annotation["type"]]
        annotations.append(ann_class.parse_obj(raw_annotation))

    return annotations


AnnotationsList = Annotated[list[SerializeAsAny[AnnotationResultABC]], BeforeValidator(ls_annotation_validator)]


class AnnotationsContainer(BaseModel):
    completed_by: Optional[int] = None
    result: AnnotationsList = []
    ground_truth: bool = False


PosePointsLookupKey = "pose_points"
PoseBBoxLookupKey = "pose_boxes"


class LabelStudioTask(BaseModel):
    annotations: list[AnnotationsContainer] = Field(
        default_factory=lambda: [],
    )
    meta: dict[str, Any] = {}
    data: dict[str, Any] = {}
    project: int = 0
    created_at: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    updated_at: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    id: int = Field(default_factory=lambda: random.randint(0, 2**63 - 1))

    user_id: int = Field(exclude=True, default=1)

    def add_annotation(self, annotation: AnnotationResultABC):
        if len(self.annotations) == 0:
            self.annotations.append(AnnotationsContainer(completed_by=self.user_id))
        self.annotations[0].result.append(annotation)

    def add_annotations(self, annotations: Sequence[AnnotationResultABC]):
        for ann in annotations:
            self.add_annotation(ann)

    def log_pose_metadata(self, bbox: RectangleLabelsAnnotation, keypoints: list[KeyPointLabelsAnnotation]):
        """
        Log additional metadata for pose annotation, that can be used later to reconstruct the pose on import

        :param bbox: Bounding box of the pose
        :param keypoints: Pose points
        """
        if PosePointsLookupKey not in self.data:
            self.data[PosePointsLookupKey] = []
        if PoseBBoxLookupKey not in self.data:
            self.data[PoseBBoxLookupKey] = []

        self.data[PoseBBoxLookupKey].append(bbox.id)
        self.data[PosePointsLookupKey].append([point.id for point in keypoints])


def parse_ls_task(task: Union[str, bytes]) -> LabelStudioTask:
    return LabelStudioTask.model_validate_json(task)
