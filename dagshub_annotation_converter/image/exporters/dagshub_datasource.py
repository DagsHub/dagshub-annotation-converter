import logging
import urllib.parse
from functools import cached_property
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, List, Tuple, Sequence

from dagshub.common.api import UserAPI
from dagshub.data_engine import dtypes

from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationProject,
    AnnotatedFile,
    AnnotationABC,
    SegmentationAnnotation,
    BBoxAnnotation,
    PoseAnnotation,
)

import pandas as pd

from dagshub_annotation_converter.schema.label_studio.abc import AnnotationResultABC
from dagshub_annotation_converter.schema.label_studio.keypointlabels import (
    KeyPointLabelsAnnotation,
    KeyPointLabelsAnnotationValue,
)
from dagshub_annotation_converter.schema.label_studio.polygonlabels import (
    PolygonLabelsAnnotation,
    PolygonLabelsAnnotationValue,
)
from dagshub_annotation_converter.schema.label_studio.rectanglelabels import (
    RectangleLabelsAnnotationValue,
    RectangleLabelsAnnotation,
)
from dagshub_annotation_converter.schema.label_studio.task import LabelStudioTask

if TYPE_CHECKING:
    from dagshub.data_engine.model.datasource import Datasource

logger = logging.getLogger(__name__)


class DagshubDatasourceExporter:
    def __init__(
        self,
        datasource: "Datasource",
        annotation_field="exported_annotation",
        upload_classes=True,
        relativize_paths=True,
    ):
        """

        :param datasource: Datasource to upload to
        :param annotation_field: Annotation field to fill
        :param upload_classes: If True, adds the classes to a <annotation_field>_classes field
        :param relativize_paths: If True, tries to relativize the paths of the datapoints relative to their path in repo
            If False, the paths are inserted as is.
            Toggle this off only if you're 100% sure that the paths of the annotations
            are the same as the paths in the datasource
        """

        self.ds = datasource
        self.annotation_field = annotation_field
        self.upload_classes = upload_classes
        self.relativize_paths = relativize_paths

    @property
    def _annotation_classes_field(self):
        return f"{self.annotation_field}_classes"

    @cached_property
    def _current_user(self) -> UserAPI:
        return UserAPI.get_current_user(host=self.ds.source.repoApi.host)

    def export(self, project: AnnotationProject):
        res: list[tuple] = []

        for f in project.files:
            # TODO: make sure this works with:
            # - absolute paths
            # - bucket datasources
            fpath = PurePosixPath(f.file)
            if self.relativize_paths:
                try:
                    fpath = fpath.relative_to(self.ds.source.source_prefix)
                except:
                    logger.warning(f"File {fpath} is not part of the datasource, skipping")
                    continue
            task = self.convert_annotated_file(f)
            download_url = self.ds.source.raw_path(str(fpath))
            download_path = urllib.parse.urlparse(download_url).path
            task.data["image"] = download_path  # Required for correctly loading the dp image
            data = [str(fpath), task.model_dump_json().encode()]
            if self.upload_classes:
                data.append(", ".join([cat.name for cat in f.categories]))
            res.append(tuple(data))
        columns = ["path", self.annotation_field]
        if self.upload_classes:
            columns.append(self._annotation_classes_field)

        df = pd.DataFrame(res, columns=columns)
        self.ds.upload_metadata_from_dataframe(df)
        self.ds.metadata_field(self.annotation_field).set_type(dtypes.Blob).set_annotation().apply()

        logger.warning(
            f"Uploaded annotations to datasource [{self.ds.source.name}]"
            f" of repo [{self.ds.source.repoApi.full_name}]"
        )

    def convert_annotated_file(self, f: AnnotatedFile) -> LabelStudioTask:
        task = LabelStudioTask(user_id=self._current_user.user_id)
        for ann in f.annotations:
            task.add_annotations(self.convert_annotation(f, ann))
        return task

    def convert_annotation(self, f: AnnotatedFile, annotation: AnnotationABC) -> Sequence[AnnotationResultABC]:
        # Todo: dynamic dispatch
        if isinstance(annotation, SegmentationAnnotation):
            return [self.convert_segmentation(f, annotation)]
        if isinstance(annotation, BBoxAnnotation):
            return [self.convert_bbox(f, annotation)]
        if isinstance(annotation, PoseAnnotation):
            bbox, keypoints = self.convert_pose(f, annotation)
            return [bbox, *keypoints]
        raise RuntimeError(f"Unknown type: {type(annotation)}")

    @staticmethod
    def convert_segmentation(f: AnnotatedFile, annotation: SegmentationAnnotation) -> PolygonLabelsAnnotation:
        assert f.image_width is not None
        assert f.image_height is not None

        annotation = annotation.normalized(f.image_width, f.image_height)
        points = [[p.x * 100, p.y * 100] for p in annotation.points]
        value = PolygonLabelsAnnotationValue(points=points, polygonlabels=[annotation.category.name])
        res = PolygonLabelsAnnotation(
            original_width=f.image_width,
            original_height=f.image_height,
            image_rotation=0.0,
            value=value,
        )
        return res

    @staticmethod
    def convert_bbox(f: AnnotatedFile, annotation: BBoxAnnotation) -> RectangleLabelsAnnotation:
        assert f.image_width is not None
        assert f.image_height is not None

        annotation = annotation.normalized(f.image_width, f.image_height)
        value = RectangleLabelsAnnotationValue(
            x=annotation.left * 100,
            y=annotation.top * 100,
            width=annotation.width * 100,
            height=annotation.height * 100,
            rectanglelabels=[annotation.category.name],
        )
        res = RectangleLabelsAnnotation(
            original_width=f.image_width,
            original_height=f.image_height,
            image_rotation=0.0,
            value=value,
        )
        return res

    @staticmethod
    def convert_pose(
        f: AnnotatedFile, annotation: PoseAnnotation
    ) -> Tuple[RectangleLabelsAnnotation, List[KeyPointLabelsAnnotation]]:
        assert f.image_width is not None
        assert f.image_height is not None

        annotation = annotation.normalized(f.image_width, f.image_height)
        value = RectangleLabelsAnnotationValue(
            x=annotation.left * 100,
            y=annotation.top * 100,
            width=annotation.width * 100,
            height=annotation.height * 100,
            rectanglelabels=[annotation.category.name],
        )
        bbox = RectangleLabelsAnnotation(
            original_width=f.image_width,
            original_height=f.image_height,
            image_rotation=0.0,
            value=value,
        )

        keypoints: List[KeyPointLabelsAnnotation] = []
        for kp in annotation.points:
            # Ignore explicitly invisible points
            if kp.is_visible is not None and not kp.is_visible:
                continue

            kp_val = KeyPointLabelsAnnotationValue(
                x=kp.x * 100, y=kp.y * 100, width=1.0, keypointlabels=[annotation.category.name]
            )
            ann = KeyPointLabelsAnnotation(
                original_width=f.image_width,
                original_height=f.image_height,
                image_rotation=0.0,
                value=kp_val,
            )
            keypoints.append(ann)

        return bbox, keypoints
