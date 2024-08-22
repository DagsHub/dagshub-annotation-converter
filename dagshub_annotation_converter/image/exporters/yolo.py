import datetime
import logging
import os
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Union, Literal, Tuple

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
    def get_yolo_yaml(
        self,
        project: AnnotationProject,
        data_dir: Path,
        image_dir_name: str,
        label_dir_name: str,
        **kwargs,
    ) -> str:
        """Gets the contents of the YOLO metadata yaml file"""
        ...

    @abstractmethod
    def convert_file(self, project: AnnotationProject, f: AnnotatedFile, **kwargs) -> str:
        """Converts annotations for a single file into a yolo annotation and returns the content"""
        ...

    def generate_generic_yaml_content(
        self,
        project: AnnotationProject,
        data_dir: Path,
        image_dir_name: str,
        label_dir_name: str,
    ) -> dict:
        train, val = self.determine_test_and_val_folders(data_dir, image_dir_name, label_dir_name)
        yaml_structure = {
            "path": str(data_dir.absolute()),
            "names": {cat.id: cat.name for cat in project.categories.categories},
            "nc": len(project.categories),
            "train": train,
            "val": val,
        }
        return yaml_structure

    @staticmethod
    def determine_test_and_val_folders(data_dir: Path, image_dir_name: str, label_dir_name: str) -> Tuple[str, str]:
        """
        Tries to find test and val folder.
        Current logic:
        1) Drill down in data_dir until a label directory is found
        2) Then look at the subfolders of the label dir
            If there's 0/1/more than 2 dirs - make the data dir both the test and val dir
            If there are two dirs and one contains train/one contains val, the other gets assigned accordingly

        """
        label_dir = None
        for root, dirs, files in os.walk(data_dir):
            if label_dir_name in dirs:
                label_dir = Path(root) / label_dir_name
        if label_dir is None:
            raise RuntimeError(f'Could not find the label dir named "{label_dir_name}" in the data directory')
        subdirs = [label_dir / p for p in os.listdir(label_dir)]
        subdirs = [p for p in subdirs if p.is_dir()]

        # Replace the label -> image
        img_dir_parts = list(label_dir.parts)
        for i, part in enumerate(reversed(label_dir.parts)):
            if part == label_dir_name:
                img_dir_parts[len(img_dir_parts) - i - 1] = image_dir_name
                break

        img_dir = Path(*img_dir_parts)
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
    def get_yolo_yaml(
        self, project: AnnotationProject, data_dir: Path, image_dir_name: str, label_dir_name: str, **kwargs
    ) -> str:
        return yaml.dump(self.generate_generic_yaml_content(project, data_dir, image_dir_name, label_dir_name))

    def convert_file(self, project: AnnotationProject, f: AnnotatedFile, **kwargs) -> str:
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
    def get_yolo_yaml(
        self,
        project: AnnotationProject,
        data_dir: Path,
        image_dir_name: str,
        label_dir_name: str,
        **kwargs,
    ) -> str:
        return yaml.dump(self.generate_generic_yaml_content(project, data_dir, image_dir_name, label_dir_name))

    def convert_file(self, project: AnnotationProject, f: AnnotatedFile, **kwargs) -> str:
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
    def get_yolo_yaml(
        self, project: AnnotationProject, data_dir: Path, image_dir_name: str, label_dir_name: str, **kwargs
    ) -> str:
        yaml_dict = self.generate_generic_yaml_content(project, data_dir, image_dir_name, label_dir_name)
        if len(project.pose_config.pose_points) == 0:
            project.regenerate_pose_points()

        max_points = project.pose_config.bind_pose_points_to_max()

        yaml_dict["kpt_shape"] = [max_points, kwargs.get("keypoint_dim", 3)]
        return yaml.dump(yaml_dict)

    def convert_file(self, project: AnnotationProject, f: AnnotatedFile, **kwargs):
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
            points_output = ""
            for p in points:
                points_output += f" {p.x} {p.y} "
                if kwargs.get("keypoint_dim", 3) == 3:
                    points_output += "1 " if p.is_visible or p.is_visible is None else "0 "
                points_output = points_output.strip()
            res += points_output
            res += "\n"
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
        overwrite_existing_meta_file: bool = True,
        pose_keypoint_dim: Literal[2, 3] = 3,
    ):
        """
        Export annotations to YOLO format
        :param data_dir: path to the directory where to store the labels
        :param annotation_type: Type of annotations to export. Either bbox, segmentation, or pose
        :param image_dir_name: Name of the image directory. Default is images
        :param label_dir_name: Name of the label directory. Default is labels
        :param label_extension: Extension to use on the label files. Default is .txt
        :param meta_file: Path of the YOLO .yaml file that will be written
        :param overwrite_existing_meta_file: If this is set to True, existing meta file wouldn't be overwritten.
            Be cautious, this might cause category mismatches, if the data has changed!
        :param pose_keypoint_dim: For pose annotations - dimensions of a keypoint. Either 2 or 3, 3 by default
        """
        self.data_dir = Path(data_dir)
        self.image_dir_name = image_dir_name
        self.label_dir_name = label_dir_name
        self.meta_file = meta_file
        self.label_extension = label_extension
        self.overwrite_existing_meta_file = overwrite_existing_meta_file
        self.pose_keypoint_dim = pose_keypoint_dim
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
            converted = self.strategy.convert_file(project, annotated, keypoint_dim=self.pose_keypoint_dim)
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
        meta_file_path = Path(self.meta_file)
        meta_file_path.parent.mkdir(parents=True, exist_ok=True)
        if meta_file_path.exists() and not self.overwrite_existing_meta_file:
            logger.warning(f"Not overwriting existing YOLO config {self.meta_file}")
        else:
            with open(self.meta_file, "w") as f:
                dt_now = datetime.datetime.now()
                f.write(
                    f"# This YOLO dataset was autogenerated by DagsHub annotation converter on {dt_now.isoformat()}\n\n"
                )
                f.write(
                    self.strategy.get_yolo_yaml(
                        project,
                        self.data_dir,
                        self.image_dir_name,
                        self.label_dir_name,
                        keypoint_dim=self.pose_keypoint_dim,
                    )
                )
                logger.warning(f"Written out the YOLO config to {self.meta_file}")
