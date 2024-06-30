import datetime
import logging
import random
from typing import Any, Sequence, Annotated, Type, Optional, Union, cast

from pydantic import BaseModel, SerializeAsAny, Field, BeforeValidator

from dagshub_annotation_converter.formats.label_studio.base import AnnotationResultABC, ImageAnnotationResultABC
from dagshub_annotation_converter.formats.label_studio.keypointlabels import KeyPointLabelsAnnotation
from dagshub_annotation_converter.formats.label_studio.polygonlabels import PolygonLabelsAnnotation
from dagshub_annotation_converter.formats.label_studio.rectanglelabels import RectangleLabelsAnnotation
from dagshub_annotation_converter.ir.image import (
    IRAnnotationBase,
    IRPoseAnnotation,
    IRBBoxAnnotation,
    NormalizationState,
    IRPosePoint,
    IRSegmentationAnnotation,
)

task_lookup: dict[str, Type[AnnotationResultABC]] = {
    "polygonlabels": PolygonLabelsAnnotation,
    "rectanglelabels": RectangleLabelsAnnotation,
    "keypointlabels": KeyPointLabelsAnnotation,
}

ir_annotation_lookup: dict[Type[IRAnnotationBase], Type[ImageAnnotationResultABC]] = {
    IRPoseAnnotation: KeyPointLabelsAnnotation,
    IRBBoxAnnotation: RectangleLabelsAnnotation,
    IRSegmentationAnnotation: PolygonLabelsAnnotation,
}

logger = logging.getLogger(__name__)


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

    def to_ir_annotations(self, filename: Optional[str] = None) -> Sequence[IRAnnotationBase]:
        res: list[IRAnnotationBase] = []
        for anns in self.annotations:
            for ann in anns.result:
                to_add = ann.to_ir_annotation()
                if filename is not None:
                    for a in to_add:
                        a.filename = filename
                res.extend(to_add)
        res = self._reimport_poses(res)
        return res

    def _reimport_poses(self, annotations: list[IRAnnotationBase]) -> list[IRAnnotationBase]:
        if PosePointsLookupKey not in self.data or PoseBBoxLookupKey not in self.data:
            return annotations

        # Build a dictionary of all annotation indexes in the task by id
        # Keep the indexes instead of annotations, so we can pop them for convenience
        annotation_lookup = {ann.imported_id: ann for ann in annotations if ann.imported_id is not None}
        pose_bboxes: list[str] = self.data[PoseBBoxLookupKey]
        pose_points: list[list[str]] = self.data[PosePointsLookupKey]

        annotations_to_remove: set[str] = set()
        poses: list[IRPoseAnnotation] = []

        for bbox_id, point_ids in zip(pose_bboxes, pose_points):
            # Fetch the bbox of the pose
            maybe_bbox = annotation_lookup.get(bbox_id)
            bbox: Optional[IRBBoxAnnotation] = None
            category: Optional[str] = None
            image_width: Optional[int] = None
            image_height: Optional[int] = None
            if maybe_bbox is None:
                logger.warning(
                    f"Bounding box of pose with annotation ID {bbox_id} "
                    f"does not exist in the task but exists in metadata"
                )
            elif not isinstance(maybe_bbox, IRBBoxAnnotation):
                logger.warning(f"Bounding box of pose with annotation ID {bbox_id} is not a bounding box annotation")
            else:
                bbox = maybe_bbox
                category = bbox.category
                image_height = bbox.image_height
                image_width = bbox.image_width
                annotations_to_remove.add(bbox_id)
            # Fetch the points
            points: list[IRPosePoint] = []
            for point_id in point_ids:
                maybe_point = annotation_lookup.get(point_id)
                if maybe_point is None:
                    logger.warning(
                        f"Point of pose with annotation ID {bbox_id} "
                        f"does not exist in the task but exists in metadata"
                    )
                    continue
                elif not isinstance(maybe_point, IRPoseAnnotation):
                    logger.warning(f"Point of pose with annotation ID {point_id} is not a point annotation")
                    continue
                else:
                    if category is None:
                        category = maybe_point.category
                    if image_width is None:
                        image_width = maybe_point.image_width
                    if image_height is None:
                        image_height = maybe_point.image_height
                    points.extend(maybe_point.points)
                    annotations_to_remove.add(point_id)

            if len(points) == 0:
                logger.warning(f"No points found for the pose on LS Task {self.id}")
                return annotations

            assert category is not None
            assert image_width is not None
            assert image_height is not None

            sum_annotation = IRPoseAnnotation.from_points(
                category=category,
                points=points,
                state=NormalizationState.NORMALIZED,
                image_width=image_width,
                image_height=image_height,
            )
            if bbox is not None:
                sum_annotation.width = bbox.width
                sum_annotation.height = bbox.height
                sum_annotation.top = bbox.top
                sum_annotation.left = bbox.left

            poses.append(sum_annotation)

        logger.debug(f"Consolidated {len(poses)} pose annotations for LS Task {self.id}")

        if len(poses) == 0:
            return annotations

        annotations = list(filter(lambda ann: ann.imported_id not in annotations_to_remove, annotations))
        annotations.extend(poses)
        return annotations

    def add_ir_annotation(self, ann: IRAnnotationBase):
        ls_ann_type = ir_annotation_lookup.get(type(ann))

        if ls_ann_type is None:
            raise ValueError(f"Unsupported IR annotation type: {type(ann)}")

        ls_anns = ls_ann_type.from_ir_annotation(ann)
        self.add_annotations(ls_anns)

        # For pose: log additional metadata
        if isinstance(ann, IRPoseAnnotation):
            bbox = cast(RectangleLabelsAnnotation, ls_anns[0])
            keypoints = cast(list[KeyPointLabelsAnnotation], ls_anns[1:])
            self.log_pose_metadata(bbox, keypoints)

    def add_ir_annotations(self, anns: Sequence[IRAnnotationBase]):
        for ann in anns:
            self.add_ir_annotation(ann)


def parse_ls_task(task: Union[str, bytes]) -> LabelStudioTask:
    return LabelStudioTask.model_validate_json(task)