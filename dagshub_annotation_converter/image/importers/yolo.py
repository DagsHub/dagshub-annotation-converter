import logging
import os
from os import PathLike
from pathlib import Path
from typing import Union, Literal, Tuple, Dict, Optional

import yaml
from PIL import Image

from dagshub_annotation_converter.image.util.path_util import yolo_img_path_to_label_path
from dagshub_annotation_converter.schema.ir.annotation_ir import (
    AnnotationProject,
    Categories,
    AnnotatedFile,
    SegmentationAnnotation,
    BBoxAnnotation,
    NormalizationState,
    AnnotationABC,
    PoseAnnotation,
)
from dagshub_annotation_converter.image.util import is_image, get_extension

logger = logging.getLogger(__name__)


class YoloImporter:
    def __init__(
        self,
        data_dir: Union[str, PathLike],
        annotation_type: Literal["bbox", "segmentation", "pose"],
        image_dir_name: str = "images",
        label_dir_name: str = "labels",
        label_extension: str = ".txt",
        meta_file: Union[str, PathLike] = "annotations.yaml",
    ):
        # TODO: handle colocated annotations (in the same dir)
        self.data_dir = data_dir
        self.image_dir_name = image_dir_name
        self.label_dir_name = label_dir_name
        self.meta_file = meta_file
        self.label_extension = label_extension
        self.annotation_type = annotation_type
        if not self.label_extension.startswith("."):
            self.label_extension = "." + self.label_extension

    def parse(self) -> AnnotationProject:
        project = AnnotationProject()
        self._parse_yolo_meta(project)
        self._parse_images(project)
        logger.warning(f"Imported {len(project.files)} YOLO annotations.")
        return project

    def _parse_yolo_meta(self, project: AnnotationProject):
        with open(self.meta_file) as f:
            meta_dict = yaml.safe_load(f)
        project.categories = self._parse_categories(meta_dict)
        if self.annotation_type == "pose":
            project.additional_metadata["yolo_keypoint_shape"] = meta_dict["kpt_shape"]
            project.additional_metadata["yolo_flip_idx"] = meta_dict.get("flip_idx")

    def _parse_categories(self, yolo_meta: Dict) -> Categories:
        categories = Categories()
        for cat_id, cat_name in yolo_meta["names"].items():
            categories.add(cat_name, cat_id)
        return categories

    def _parse_images(self, project: AnnotationProject):
        for dirpath, subdirs, files in os.walk(self.data_dir):
            if self.image_dir_name not in dirpath.split("/"):
                logger.debug(f"{dirpath} is not an image dir, skipping")
                continue
            for filename in files:
                img = Path(os.path.join(dirpath, filename))
                if not is_image(img):
                    logger.debug(f"Skipping {img} because it's not an image")
                    continue
                annotation = yolo_img_path_to_label_path(
                    img, self.image_dir_name, self.label_dir_name, self.label_extension
                )
                if not annotation.exists():
                    logger.warning(f"Couldn't find annotation file [{annotation}] for image file [{img}]")
                    continue
                project.files.append(self._parse_annotation(img, annotation, project))

    def _get_annotation_file(self, img: Path) -> Path:
        new_parts = list(img.parts)
        # Replace last occurrence of image_dir_name to label_dir_name
        for i, part in enumerate(reversed(img.parts)):
            if part == self.image_dir_name:
                new_parts[len(new_parts) - i - 1] = self.label_dir_name

        # Replace the extension
        new_parts[-1] = new_parts[-1].replace(get_extension(img), self.label_extension)
        return Path(*new_parts)

    def _parse_annotation(self, img: Path, annotation: Path, project: AnnotationProject) -> AnnotatedFile:
        res = AnnotatedFile(file=img)
        res.image_width, res.image_height = self._get_image_dimensions(img)

        with open(annotation) as ann_file:
            for line in ann_file.readlines():
                ann: AnnotationABC
                if self.annotation_type == "segmentation":
                    ann = self._parse_segment(line, project.categories)
                elif self.annotation_type == "bbox":
                    ann = self._parse_bbox(line, project.categories)
                elif self.annotation_type == "pose":
                    # dimensions is either 2 or 3, [x, y, (optional) visibility]
                    keypoint_dimensions = project.additional_metadata["yolo_keypoint_shape"][1]
                    ann = self._parse_pose(line, project.categories, keypoint_dimensions)
                else:
                    raise RuntimeError(f"Unknown annotation type [{self.annotation_type}]")
                res.annotations.append(ann.denormalized(res.image_width, res.image_height))
        return res

    @staticmethod
    def _parse_segment(line: str, categories: Categories) -> SegmentationAnnotation:
        vals = line.split()
        category = categories.get(int(vals[0]))
        if category is None:
            raise RuntimeError(f"Unknown category {category}. Imported categories from the .yaml: {categories}")
        res = SegmentationAnnotation(category=category, state=NormalizationState.NORMALIZED)
        for i in range(1, len(vals) - 1, 2):
            x = float(vals[i])
            y = float(vals[i + 1])
            res.add_point(x, y)
        return res

    @staticmethod
    def _parse_bbox(line: str, categories: Categories) -> BBoxAnnotation:
        vals = line.split()
        category = categories.get(int(vals[0]))
        if category is None:
            raise RuntimeError(f"Unknown category {category}. Imported categories from the .yaml: {categories}")
        middle_x = float(vals[1])
        middle_y = float(vals[2])
        width = float(vals[3])
        height = float(vals[4])

        top, left, width, height = YoloImporter._convert_bbox_from_middle_to_top_left(middle_x, middle_y, width, height)

        res = BBoxAnnotation(
            top=top,
            left=left,
            width=width,
            height=height,
            category=category,
            state=NormalizationState.NORMALIZED,
        )
        return res

    @staticmethod
    def _parse_pose(line: str, categories: Categories, keypoint_dimensions: int) -> PoseAnnotation:
        vals = line.split()
        category = categories.get(int(vals[0]))
        if category is None:
            raise RuntimeError(f"Unknown category {category}. Imported categories from the .yaml: {categories}")
        middle_x = float(vals[1])
        middle_y = float(vals[2])
        width = float(vals[3])
        height = float(vals[4])

        top, left, width, height = YoloImporter._convert_bbox_from_middle_to_top_left(middle_x, middle_y, width, height)

        res = PoseAnnotation(
            category=category,
            top=top,
            left=left,
            width=width,
            height=height,
            state=NormalizationState.NORMALIZED,
        )

        for i in range(5, len(vals) - 1, keypoint_dimensions):
            x = float(vals[i])
            y = float(vals[i + 1])
            is_visible: Optional[bool] = None
            if keypoint_dimensions == 3:
                is_visible = float(vals[i + 2]) > 0
            res.add_point(x, y, is_visible)

        return res

    @staticmethod
    def _get_image_dimensions(filepath: Path) -> Tuple[int, int]:
        with Image.open(filepath) as img:
            return img.width, img.height

    @staticmethod
    def _convert_bbox_from_middle_to_top_left(
        middle_x: float, middle_y: float, width: float, height: float
    ) -> Tuple[float, float, float, float]:
        """
        Converts the YOLO bbox format which has the point in the middle, to a bbox with the point in the top-left
        Returns:
             top, left, width, height
        """
        top = middle_y - height / 2.0
        left = middle_x - width / 2.0

        return top, left, width, height


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    os.chdir("/Users/kirillbolashev/temp/COCO_1K")
    importer = YoloImporter(data_dir="data", annotation_type="segmentation", meta_file="custom_coco.yaml")
    importer.parse()
