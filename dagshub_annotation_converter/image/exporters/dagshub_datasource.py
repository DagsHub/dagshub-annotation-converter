import logging
import urllib.parse
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from dagshub_annotation_converter.image.ir.annotation_ir import (
    AnnotationProject,
    AnnotatedFile,
    AnnotationABC,
    SegmentationAnnotation,
    BBoxAnnotation,
)

import pandas as pd

from dagshub_annotation_converter.schema.label_studio.abc import AnnotationResultABC
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
        self, datasource: "Datasource", annotation_field="exported_annotation"
    ):
        self.ds = datasource
        self.annotation_field = annotation_field

    def export(self, project: AnnotationProject):
        res: list[tuple[str, bytes]] = []

        for f in project.files:
            # TODO: make sure this works with:
            # - absolute paths
            # - bucket datasources
            fpath = PurePosixPath(f.file)
            try:
                relpath = fpath.relative_to(self.ds.source.source_prefix)
            except:
                logger.warning(f"File {fpath} is not part of the datasource, skipping")
                continue
            task = self.convert_annotated_file(f)
            download_url = self.ds.source.raw_path(str(relpath))
            download_path = urllib.parse.urlparse(download_url).path
            task.data[
                "image"
            ] = download_path  # Required for correctly loading the dp image
            res.append((str(relpath), task.model_dump_json().encode()))

        # print(res)
        df = pd.DataFrame(res, columns=["path", self.annotation_field])
        self.ds.upload_metadata_from_dataframe(df)
        self.ds.metadata_field(self.annotation_field).set_annotation().apply()

    def convert_annotated_file(self, f: AnnotatedFile) -> LabelStudioTask:
        task = LabelStudioTask()
        for ann in f.annotations:
            task.add_annotation(self.convert_annotation(f, ann))
        return task

    def convert_annotation(
        self, f: AnnotatedFile, annotation: AnnotationABC
    ) -> AnnotationResultABC:
        # Todo: dynamic dispatch
        if isinstance(annotation, SegmentationAnnotation):
            return self.convert_segmentation(f, annotation)
        if isinstance(annotation, BBoxAnnotation):
            return self.convert_bbox(f, annotation)
        raise RuntimeError(f"Unknown type: {type(annotation)}")

    @staticmethod
    def convert_segmentation(
        f: AnnotatedFile, annotation: SegmentationAnnotation
    ) -> PolygonLabelsAnnotation:
        assert f.image_width is not None
        assert f.image_height is not None

        annotation = annotation.normalized(f.image_width, f.image_height)
        points = [[p.x * 100, p.y * 100] for p in annotation.points]
        value = PolygonLabelsAnnotationValue(
            points=points, polygonlabels=[annotation.category.name]
        )
        res = PolygonLabelsAnnotation(
            original_width=f.image_width,
            original_height=f.image_height,
            image_rotation=0.0,
            value=value,
        )
        return res

    @staticmethod
    def convert_bbox(
        f: AnnotatedFile, annotation: BBoxAnnotation
    ) -> RectangleLabelsAnnotation:
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
