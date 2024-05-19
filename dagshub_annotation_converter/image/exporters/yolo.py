import datetime
import logging
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Union, Literal

import yaml

from dagshub_annotation_converter.image.util.path_util import yolo_img_path_to_label_path
from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationProject,
    AnnotatedFile,
    SegmentationAnnotation,
    BBoxAnnotation,
    PoseAnnotation,
)

YoloAnnotationType = Literal["bbox", "segmentation", "pose"]

logger = logging.getLogger(__name__)


class YoloExporterStrategy:
    @abstractmethod
    def get_yolo_yaml(self, project: AnnotationProject) -> str:
        """Gets the contents of the YOLO metadata yaml file"""
        ...

    @abstractmethod
    def convert_file(self, project: AnnotationProject, f: AnnotatedFile) -> str:
        """Converts annotations for a single file into a yolo annotation and returns the content"""
        ...


class BBoxExporterStrategy(YoloExporterStrategy):
    def get_yolo_yaml(self, project: AnnotationProject) -> str:
        yaml_structure = {"names": {cat.id: cat.name for cat in project.categories.categories}}
        return yaml.dump(yaml_structure)

    def convert_file(self, project: AnnotationProject, f: AnnotatedFile) -> str:
        wrong_annotation_counter = 0
        res = ""
        for ann in f.annotations:
            assert f.image_width is not None and f.image_height is not None
            ann = ann.normalized(f.image_width, f.image_height)
            if not isinstance(ann, BBoxAnnotation):
                wrong_annotation_counter += 1
                continue
            middle_x = ann.left + (ann.width / 2)
            middle_y = ann.top + (ann.height / 2)
            res += f"{ann.category.id} {middle_x} {middle_y} {ann.width} {ann.height}\n"
        if wrong_annotation_counter != 0:
            logger.warning(f"Skipped {wrong_annotation_counter} non-bounding box annotation(s) for file {f.file}")
        return res


class SegmentationExporterStrategy(YoloExporterStrategy):
    def get_yolo_yaml(self, project: AnnotationProject) -> str:
        yaml_structure = {"names": {cat.id: cat.name for cat in project.categories.categories}}
        return yaml.dump(yaml_structure)

    def convert_file(self, project: AnnotationProject, f: AnnotatedFile) -> str:
        wrong_annotation_counter = 0
        res = ""
        for ann in f.annotations:
            assert f.image_width is not None and f.image_height is not None
            ann = ann.normalized(f.image_width, f.image_height)
            if not isinstance(ann, SegmentationAnnotation):
                wrong_annotation_counter += 1
                continue
            res += f"{ann.category.id} "
            res += " ".join([f"{p.x} {p.y}" for p in ann.points])
            res += "\n"
        if wrong_annotation_counter != 0:
            logger.warning(f"Skipped {wrong_annotation_counter} non-segment annotation(s) for file {f.file}")
        return res


class PoseExporterStrategy(YoloExporterStrategy):
    def get_yolo_yaml(self, project: AnnotationProject) -> str:
        yaml_structure = {"names": {cat.id: cat.name for cat in project.categories.categories}}
        yaml_structure["kpt_shape"] = [keypoints_n, 3]
        return yaml.dump(yaml_structure)

    def convert_file(self, project: AnnotationProject, f: AnnotatedFile):
        # For pose: validate that the amount of points is equal across all annotations, otherwise don't import
        # Need poses to be consistent
        wrong_annotation_counter = 0
        res = ""
        for ann in f.annotations:
            assert f.image_width is not None and f.image_height is not None
            ann = ann.normalized(f.image_width, f.image_height)
            if not isinstance(ann, PoseAnnotation):
                wrong_annotation_counter += 1
                continue
            middle_x = ann.left + (ann.width / 2)
            middle_y = ann.top + (ann.height / 2)
            res += f"{ann.category.id} {middle_x} {middle_y} {ann.width} {ann.height} "
            res += " ".join([f"{p.x} {p.y} {1 if p.is_visible or p.is_visible is None else 0}" for p in ann.points])
        if wrong_annotation_counter != 0:
            logger.warning(f"Skipped {wrong_annotation_counter} non-pose annotation(s) for file {f.file}")
        return res


class YoloExporter:
    def __init__(
        self,
        data_dir: Union[str, PathLike],
        annotation_type: YoloAnnotationType,
        image_dir_name: str = "images",
        label_dir_name: str = "labels",
        label_extension: str = ".txt",
        meta_file: Union[str, PathLike] = "annotations.yaml",
    ):
        self.data_dir = data_dir
        self.image_dir_name = image_dir_name
        self.label_dir_name = label_dir_name
        self.meta_file = meta_file
        self.label_extension = label_extension
        if not self.label_extension.startswith("."):
            self.label_extension = "." + self.label_extension

        self.strategy = self.determine_export_strategy(annotation_type)

    @staticmethod
    def determine_export_strategy(annotation_type: YoloAnnotationType) -> YoloExporterStrategy:
        # TODO: try to guess the type from the annotations in the project
        if annotation_type == "bbox":
            return BBoxExporterStrategy()
        elif annotation_type == "segmentation":
            return SegmentationExporterStrategy()
        elif annotation_type == "pose":
            return PoseExporterStrategy()
        else:
            raise ValueError(
                f"Unknown yolo annotation type: {annotation_type}. Allowed types are: {YoloAnnotationType}"
            )

    def export(self, project: AnnotationProject):
        # Write the metadata file
        with open(self.meta_file, "w") as f:
            dt_now = datetime.datetime.now()
            f.write(
                f"# This YOLO dataset was autogenerated by DagsHub annotation converter on {dt_now.isoformat()}\n\n"
            )
            f.write(self.strategy.get_yolo_yaml(project))

        # Write the annotations
        for annotated in project.files:
            converted = self.strategy.convert_file(project, annotated)
            if not converted:
                logger.warning(f"No annotations of fitting type found for file {annotated.file}")
                continue
            annotation_file_path = yolo_img_path_to_label_path(
                Path(annotated.file), self.image_dir_name, self.label_dir_name, self.label_extension
            )
            annotation_file_path = self.data_dir / annotation_file_path
            annotation_file_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_file_path.write_text(converted)
