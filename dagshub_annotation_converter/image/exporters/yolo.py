import datetime
import logging
import os
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
    KeyPoint,
)

YoloAnnotationType = Literal["bbox", "segmentation", "pose"]

logger = logging.getLogger(__name__)


class YoloExporterStrategy:
    @abstractmethod
    def get_yolo_yaml(self, project: AnnotationProject, data_dir: Path, image_dir_name: str) -> str:
        """Gets the contents of the YOLO metadata yaml file"""
        ...

    @abstractmethod
    def convert_file(self, project: AnnotationProject, f: AnnotatedFile) -> str:
        """Converts annotations for a single file into a yolo annotation and returns the content"""
        ...

    def generate_generic_yaml_content(self, project: AnnotationProject, data_dir: Path, image_dir_name: str) -> dict:
        train, val = self.determine_test_and_val_folders(data_dir, image_dir_name)
        yaml_structure = {
            "path": str(data_dir.absolute()),
            "names": {cat.id: cat.name for cat in project.categories.categories},
            "nc": len(project.categories),
            "train": train,
            "val": val,
        }
        return yaml_structure

    def determine_test_and_val_folders(self, data_dir: Path, image_dir_name: str) -> tuple[str, str]:
        """
        Tries to find test and val folder.
        Current logic:
        1) Drill down in data_dir until an image directory is found
        2) Then look at the subfolders of the image dir
            If there's 0/1/more than 2 dirs - make the data dir both the test and val dir
            If there are two dirs and one contains train/one contains val, the other gets assigned accordingly

        """
        img_dir = None
        for root, dirs, files in os.walk(data_dir):
            if image_dir_name in dirs:
                img_dir = Path(root) / image_dir_name
        if img_dir is None:
            raise RuntimeError(f'Could not find the image dir named "{image_dir_name}" in the data directory')
        subdirs = [img_dir / p for p in os.listdir(img_dir)]
        subdirs = [p for p in subdirs if p.is_dir()]

        img_dir_relative = img_dir.relative_to(data_dir)

        if len(subdirs) != 2:
            return str(img_dir_relative), str(img_dir_relative)

        p1 = subdirs[0]
        p2 = subdirs[1]

        if "train" in p1.name or "val" in p2.name:
            return str(img_dir_relative / p1.name), str(img_dir_relative / p2.name)
        elif "val" in p1.name or "train" in p1.name:
            return str(img_dir_relative / p2.name), str(img_dir_relative / p1.name)
        return str(img_dir_relative), str(img_dir_relative)


class BBoxExporterStrategy(YoloExporterStrategy):
    def get_yolo_yaml(self, project: AnnotationProject, data_dir: Path, image_dir_name: str) -> str:
        return yaml.dump(self.generate_generic_yaml_content(project, data_dir, image_dir_name))

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
    def get_yolo_yaml(self, project: AnnotationProject, data_dir: Path, image_dir_name: str) -> str:
        return yaml.dump(self.generate_generic_yaml_content(project, data_dir, image_dir_name))

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
    def get_yolo_yaml(self, project: AnnotationProject, data_dir: Path, image_dir_name: str) -> str:
        yaml_dict = self.generate_generic_yaml_content(project, data_dir, image_dir_name)
        if len(project.pose_config.pose_points) == 0:
            project.regenerate_pose_points()

        max_points = project.pose_config.bind_pose_points_to_max()

        yaml_dict["kpt_shape"] = [max_points, 3]
        return yaml.dump(yaml_dict)

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
            points = ann.points
            # Fill out the remaining until we hit the max
            category_n_points = project.pose_config.pose_points.get(ann.category, len(points))
            if len(points) < category_n_points:
                delta = category_n_points - len(points)
                points.extend([KeyPoint(x=0.0, y=0.0, is_visible=False)] * delta)
            res += " ".join([f"{p.x} {p.y} {1 if p.is_visible or p.is_visible is None else 0}" for p in points])
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
        self.data_dir = Path(data_dir)
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

        # Write the metadata file
        Path(self.meta_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_file, "w") as f:
            dt_now = datetime.datetime.now()
            f.write(
                f"# This YOLO dataset was autogenerated by DagsHub annotation converter on {dt_now.isoformat()}\n\n"
            )
            f.write(self.strategy.get_yolo_yaml(project, self.data_dir, self.image_dir_name))
