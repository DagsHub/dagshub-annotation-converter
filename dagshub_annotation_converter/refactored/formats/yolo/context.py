import os.path
from pathlib import Path
from typing import Dict, Union, Optional, Literal, Callable

import yaml
from pydantic import BaseModel

from dagshub_annotation_converter.refactored.ir.image import Categories


from dagshub_annotation_converter.refactored.formats.common import ImageType
from dagshub_annotation_converter.refactored.ir.image import IRAnnotationBase

YoloConverterFunction = Callable[
    [str, "YoloContext", Optional[int], Optional[int], Optional[ImageType]], IRAnnotationBase
]

YoloAnnotationTypes = Literal["bbox", "segmentation", "pose"]


class YoloContext(BaseModel):
    annotation_type: YoloAnnotationTypes
    """Type of annotations associated with this Yolo Context"""
    categories: Categories = Categories()
    """List of categories"""
    label_dir_name: str = "labels"
    """Name of the directory containing label files"""
    image_dir_name: str = "image"
    """Name of the directory containing image files"""
    keypoint_dim: Literal[2, 3] = 3
    """[For pose annotations] 
    Dimension of the annotation: 2 - x, y; 3 - x, y, visibility"""
    keypoints_in_annotation: Optional[int] = None
    """[For pose annotations]
    Number of keypoints in each annotation"""
    label_extension: str = ".txt"
    """Extension of the annotation files"""
    path: Optional[Path] = None
    """Base path to the data"""
    test_path: Optional[Path] = None
    """Path to the test data, relative to the base path"""
    val_path: Optional[Path] = None
    """Path to the validation data, relative to the base path"""

    @staticmethod
    def from_yaml_file(file_path: Union[str, Path], annotation_type: YoloAnnotationTypes) -> "YoloContext":
        res = YoloContext(annotation_type=annotation_type)
        file_path = Path(file_path)
        with open(file_path) as f:
            meta_dict = yaml.safe_load(f)
        res.categories = YoloContext._parse_categories(meta_dict)

        if "kpt_shape" in meta_dict:
            res.keypoints_in_annotation = meta_dict["kpt_shape"][0]
            res.keypoint_dim = meta_dict["kpt_shape"][1]

        if "path" in meta_dict:
            if os.path.isabs(meta_dict["path"]):
                res.path = Path(meta_dict["path"])
            else:
                res.path = file_path.parent / meta_dict["path"]

        return res

    @staticmethod
    def _parse_categories(yolo_meta: Dict) -> Categories:
        categories = Categories()
        for cat_id, cat_name in yolo_meta["names"].items():
            categories.add(cat_name, cat_id)
        return categories
