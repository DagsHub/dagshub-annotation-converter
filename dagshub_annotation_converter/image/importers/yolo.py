import logging
import os
from os import PathLike
from pathlib import Path
from typing import Union, Literal, Tuple

import yaml
from PIL import Image

from dagshub_annotation_converter.image.ir.annotation_ir import (
    AnnotationProject,
    Categories,
    AnnotatedFile,
    SegmentationAnnotation,
    BBoxAnnotation,
    NormalizationState, AnnotationABC,
)
from dagshub_annotation_converter.image.util import is_image, get_extension

logger = logging.getLogger(__name__)


class YoloImporter:
    def __init__(
        self,
        data_dir: Union[str, PathLike],
        annotation_type: Literal["bbox", "segmentation"],
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
        project.categories = self._parse_categories()
        self._parse_images(project)
        return project

    def _parse_categories(self) -> Categories:
        with open(self.meta_file) as f:
            meta_dict = yaml.safe_load(f)
        categories = Categories()
        for cat_id, cat_name in meta_dict["names"].items():
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
                annotation = self._get_annotation_file(img)
                if not annotation.exists():
                    logger.warning(
                        f"Couldn't find annotation file [{annotation}] for image file [{img}]"
                    )
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

    def _parse_annotation(
        self, img: Path, annotation: Path, project: AnnotationProject
    ) -> AnnotatedFile:
        res = AnnotatedFile(file=img)
        res.image_width, res.image_height = self._get_image_dimensions(img)

        with open(annotation) as ann_file:
            for line in ann_file.readlines():
                ann: AnnotationABC
                if self.annotation_type == "segmentation":
                    ann = self._parse_segment(line, project.categories)
                elif self.annotation_type == "bbox":
                    ann = self._parse_bbox(line, project.categories)
                else:
                    raise RuntimeError(
                        f"Unknown annotation type [{self.annotation_type}]"
                    )
                res.annotations.append(
                    ann.denormalized(res.image_width, res.image_height)
                )
        return res

    @staticmethod
    def _parse_segment(line: str, categories: Categories) -> SegmentationAnnotation:
        vals = line.split()
        category = categories.get(int(vals[0]))
        if category is None:
            raise RuntimeError(
                f"Unknown category {category}. Imported categories from the .yaml: {categories}"
            )
        res = SegmentationAnnotation(
            category=category, state=NormalizationState.NORMALIZED
        )
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
            raise RuntimeError(
                f"Unknown category {category}. Imported categories from the .yaml: {categories}"
            )
        middle_x = float(vals[1])
        middle_y = float(vals[2])
        width = float(vals[3])
        height = float(vals[4])

        top = middle_y - height / 2.0
        left = middle_x - width / 2.0

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
    def _get_image_dimensions(filepath: Path) -> Tuple[int, int]:
        with Image.open(filepath) as img:
            return img.width, img.height


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    os.chdir("/Users/kirillbolashev/temp/COCO_1K")
    importer = YoloImporter(
        data_dir="data", annotation_type="segmentation", meta_file="custom_coco.yaml"
    )
    importer.parse()
