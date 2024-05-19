import logging
from typing import TYPE_CHECKING, List, Optional, Union, TypeGuard

from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationProject,
    AnnotatedFile,
    AnnotationABC,
    BBoxAnnotation,
    PoseAnnotation,
    KeyPoint,
    NormalizationState,
    Category,
)
from dagshub_annotation_converter.schema.label_studio.abc import ImageAnnotationResultABC, AnnotationResultABC
from dagshub_annotation_converter.schema.label_studio.task import (
    LabelStudioTask,
    PosePointsLookupKey,
    PoseBBoxLookupKey,
)

if TYPE_CHECKING:
    from dagshub.data_engine.model.datasource import Datasource, QueryResult, Datapoint

logger = logging.getLogger(__name__)


class DagshubDatasourceImporter:
    def __init__(
        self,
        datasource_or_queryresult: Union["Datasource", "QueryResult"],
        annotation_fields: Optional[Union[str, List[str]]] = None,
    ):
        """

        :param datasource:  Datasource or QueryResult to import from.
            If it's a datasource, calls ``datasource.all()`` to load all datapoints
        :param annotation_fields: List of names of fields to get annotations from.
            If ``None``, loads all annotation fields.
            Loading multiple fields concatenates the annotations together.
        """
        from dagshub.data_engine.model.datasource import Datasource

        self.datasource: "Datasource"
        self._query_result: Optional["QueryResult"] = None
        if isinstance(datasource_or_queryresult, Datasource):
            self.datasource = datasource_or_queryresult
        else:
            self.datasource = datasource_or_queryresult.datasource
            self._query_result = datasource_or_queryresult

        if isinstance(annotation_fields, str):
            annotation_fields = [annotation_fields]
        self.annotation_fields: List[str] = (
            annotation_fields if annotation_fields else self.datasource.annotation_fields
        )

        if len(self.annotation_fields) == 0:
            raise RuntimeError("Datasource doesn't have any annotation fields")

    @property
    def query_result(self) -> "QueryResult":
        if self._query_result is None:
            self._query_result = self.datasource.all()
        return self._query_result

    def parse(self) -> AnnotationProject:
        project = AnnotationProject()

        self.query_result.get_blob_fields(
            *self.annotation_fields,
            load_into_memory=True,
        )

        for dp in self.query_result:
            self.parse_datapoint(project, dp)

        return project

    def parse_datapoint(self, project: AnnotationProject, dp: "Datapoint"):
        ann_file = AnnotatedFile(file=dp.path)

        for annotation_field in self.annotation_fields:
            if annotation_field not in dp.metadata or not isinstance(dp.metadata[annotation_field], bytes):
                continue

            task = LabelStudioTask.model_validate_json(dp.metadata[annotation_field])
            self.convert_ls_task_to_ir(project, ann_file, task)

        # Add only if the datapoint had annotations assigned
        if len(ann_file.annotations) > 0:
            project.files.append(ann_file)

    @staticmethod
    def is_all_image_annotations(val: list[AnnotationResultABC]) -> TypeGuard[list[ImageAnnotationResultABC]]:
        return all(isinstance(x, ImageAnnotationResultABC) for x in val)

    def convert_ls_task_to_ir(self, project: AnnotationProject, f: AnnotatedFile, task: LabelStudioTask):
        if len(task.annotations) == 0:
            return

        for annotations in task.annotations:
            annotations_obj = annotations.result
            # Narrow the type to the image abc
            assert self.is_all_image_annotations(annotations_obj)
            if len(annotations_obj) == 0:
                continue

            for annotation in annotations_obj:
                f.annotations.extend(self.convert_ls_annotation_to_ir(project, f, annotation))

        self._consolidate_poses(f, task)

    @staticmethod
    def _consolidate_poses(f: AnnotatedFile, task: LabelStudioTask):
        # For keypoints - try to rebuild the exact poses using the metadata that may be in the data of the task
        if PosePointsLookupKey not in task.data or PoseBBoxLookupKey not in task.data:
            return

        # Build a dictionary of all annotation indexes in the task by id
        # Keep the indexes instead of annotations so we can pop them for convenience
        annotation_lookup = {ann.imported_id: ann for ann in f.annotations if ann.imported_id is not None}
        pose_bboxes: list[str] = task.data[PoseBBoxLookupKey]
        pose_points: list[list[str]] = task.data[PosePointsLookupKey]

        annotations_to_remove: set[str] = set()
        poses: list[PoseAnnotation] = []

        for bbox_id, point_ids in zip(pose_bboxes, pose_points):
            # Fetch the bbox of the pose
            maybe_bbox = annotation_lookup.get(bbox_id)
            bbox: Optional[BBoxAnnotation] = None
            category: Optional[Category] = None
            if maybe_bbox is None:
                logger.warning(
                    f"Bounding box of pose with annotation ID {bbox_id} "
                    f"does not exist in the task but exists in metadata"
                )
            elif not isinstance(maybe_bbox, BBoxAnnotation):
                logger.warning(f"Bounding box of pose with annotation ID {bbox_id} is not a bounding box annotation")
            else:
                bbox = maybe_bbox
                category = bbox.category
                annotations_to_remove.add(bbox_id)
            # Fetch the points
            points: list[KeyPoint] = []
            for point_id in point_ids:
                maybe_point = annotation_lookup.get(point_id)
                if maybe_point is None:
                    logger.warning(
                        f"Point of pose with annotation ID {bbox_id} "
                        f"does not exist in the task but exists in metadata"
                    )
                    continue
                elif not isinstance(maybe_point, PoseAnnotation):
                    logger.warning(f"Point of pose with annotation ID {point_id} is not a point annotation")
                    continue
                else:
                    if category is None:
                        category = maybe_point.category
                    points.extend(maybe_point.points)
                    annotations_to_remove.add(point_id)

            if len(points) == 0:
                logger.warning(f"No points found for the pose on file {f.file}")
                return

            assert category is not None
            sum_annotation = PoseAnnotation.from_points(
                category=category, points=points, state=NormalizationState.NORMALIZED
            )
            if bbox is not None:
                sum_annotation.width = bbox.width
                sum_annotation.height = bbox.height
                sum_annotation.top = bbox.top
                sum_annotation.left = bbox.left

            poses.append(sum_annotation)

        logger.debug(f"Consolidated {len(poses)} pose annotations for file {f.file}")

        if len(poses) == 0:
            return

        f.annotations = list(filter(lambda ann: ann.imported_id not in annotations_to_remove, f.annotations))
        f.annotations.extend(poses)

    @staticmethod
    def convert_ls_annotation_to_ir(
        project: AnnotationProject, f: AnnotatedFile, annotation: ImageAnnotationResultABC
    ) -> list[AnnotationABC]:
        # Set the image dimensions if they weren't set already
        if f.image_width is None or f.image_height is None:
            f.image_width = annotation.original_width
            f.image_height = annotation.original_height

        return annotation.to_ir_annotation(project)
