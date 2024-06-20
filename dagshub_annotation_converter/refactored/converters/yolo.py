import logging
import os
from pathlib import Path
from typing import Union, Sequence, List, Dict, Optional

import PIL.Image

from dagshub_annotation_converter.refactored.formats.yolo.bbox import import_bbox_from_string
from dagshub_annotation_converter.refactored.formats.yolo.context import (
    YoloContext,
    YoloAnnotationTypes,
    YoloConverterFunction,
)
from dagshub_annotation_converter.refactored.formats.yolo.pose import import_pose_from_string
from dagshub_annotation_converter.refactored.formats.yolo.segmentation import import_segment_from_string
from dagshub_annotation_converter.refactored.ir.image import IRAnnotationBase
from dagshub_annotation_converter.refactored.util import is_image, yolo_img_path_to_label_path

logger = logging.getLogger(__name__)


def load_yolo_from_fs_with_context(
    context: YoloContext,
) -> dict[str, Sequence[IRAnnotationBase]]:
    assert context.path is not None

    annotations: dict[str, Sequence[IRAnnotationBase]] = {}

    for dirpath, subdirs, files in os.walk(context.path):
        if context.image_dir_name not in dirpath.split("/"):
            logger.debug(f"{dirpath} is not an image dir, skipping")
            continue
        for filename in files:
            fullpath = os.path.join(dirpath, filename)
            img = Path(fullpath)
            relpath = img.relative_to(context.path)
            if not is_image(img):
                logger.debug(f"Skipping {img} because it's not an image")
                continue
            annotation = yolo_img_path_to_label_path(
                img, context.image_dir_name, context.label_dir_name, context.label_extension
            )
            if not annotation.exists():
                logger.warning(f"Couldn't find annotation file [{annotation}] for image file [{img}]")
                continue
            annotations[str(relpath)] = parse_annotation(context, context.path, img, annotation)

    return annotations


def parse_annotation(
    context: YoloContext, base_path: Path, img_path: Path, annotation_path: Path
) -> Sequence[IRAnnotationBase]:
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size

    annotation_strings = annotation_path.read_text().strip().split("\n")

    convert_funcs: Dict[str, YoloConverterFunction] = {
        "bbox": import_bbox_from_string,
        "segmentation": import_segment_from_string,
        "pose": import_pose_from_string,
    }

    assert context.annotation_type is not None

    convert_func = convert_funcs[context.annotation_type]

    res: List[IRAnnotationBase] = []
    rel_path = str(img_path.relative_to(base_path))

    for ann in annotation_strings:
        res.append(convert_func(ann, context, img_width, img_height, img).with_filename(rel_path))

    return res


def load_yolo_from_fs(
    annotation_type: YoloAnnotationTypes,
    meta_file: Union[str, Path] = "annotations.yaml",
    path: Optional[Union[str, Path]] = None,
    image_dir_name: str = "images",
    label_dir_name: str = "labels",
) -> tuple[dict[str, Sequence[IRAnnotationBase]], YoloContext]:
    context = YoloContext.from_yaml_file(meta_file)
    context.image_dir_name = image_dir_name
    context.label_dir_name = label_dir_name
    context.annotation_type = annotation_type

    if path is not None:
        context.path = Path(path)

    return load_yolo_from_fs_with_context(context), context
