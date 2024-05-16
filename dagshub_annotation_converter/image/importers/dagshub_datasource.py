import itertools
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union, TypeGuard

from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationProject,
    AnnotatedFile,
    AnnotationABC,
    PoseAnnotation,
)
from dagshub_annotation_converter.schema.label_studio.abc import ImageAnnotationResultABC, AnnotationResultABC
from dagshub_annotation_converter.schema.label_studio.task import LabelStudioTask

if TYPE_CHECKING:
    from dagshub.data_engine.model.datasource import Datasource, QueryResult, Datapoint


class DagshubDatasourceImporter:
    def __init__(
        self,
        datasource_or_queryresult: Union["Datasource", "QueryResult"],
        annotation_fields: Optional[Union[str, List[str]]] = None,
        download_path: Optional[Union[str, Path]] = None,
    ):
        """

        :param datasource:  Datasource or QueryResult to import from.
            If it's a datasource, calls ``datasource.all()`` to load all datapoints
        :param annotation_fields: List of names of fields to get annotations from.
            If ``None``, loads all annotation fields.
            Loading multiple fields concatenates the annotations together.
        :param download_path: Where to download the datapoints. If None, they won't be downloaded.
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
        # TODO: handle self.download_path

        project = AnnotationProject()

        self.query_result.get_blob_fields(
            *self.annotation_fields,
            load_into_memory=True,
            # cache_on_disk=False   # FIXME: turn on for prod
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

        # For keypoints - since we're getting them one-by-one, we actually need to aggregate them.
        # Sadly I don't know how to do it better than making a single pose out of them

        pose_annotations: list[PoseAnnotation] = list(
            filter(lambda ann: isinstance(ann, PoseAnnotation), f.annotations)
        )
        non_pose_annotations = list(filter(lambda ann: not isinstance(ann, PoseAnnotation), f.annotations))

        aggregated_pose_annotations = []

        pose_annotations.sort(key=lambda ann: ann.category.name)  # Needed for groupby to work
        for cat, poses_iter in itertools.groupby(pose_annotations, key=lambda ann: ann.category):
            poses: list[PoseAnnotation] = list(poses_iter)
            all_points = []
            for pose in poses:
                all_points.extend(pose.points)
            aggregated_annotation = PoseAnnotation.from_points(category=cat, points=all_points, state=poses[0].state)
            aggregated_pose_annotations.append(aggregated_annotation)

        f.annotations = non_pose_annotations
        f.annotations.extend(aggregated_pose_annotations)

    @staticmethod
    def convert_ls_annotation_to_ir(
        project: AnnotationProject, f: AnnotatedFile, annotation: ImageAnnotationResultABC
    ) -> list[AnnotationABC]:
        # Set the image dimensions if they weren't set already
        if f.image_width is None or f.image_height is None:
            f.image_width = annotation.original_width
            f.image_height = annotation.original_height

        return annotation.to_ir_annotation(project)
