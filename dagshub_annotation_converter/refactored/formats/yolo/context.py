from pathlib import Path
from typing import Dict, Union, Optional, Literal

import yaml
from pydantic import BaseModel

from dagshub_annotation_converter.refactored.ir.image import Categories


class YoloContext(BaseModel):
    categories: Categories = Categories()
    """List of categories"""
    label_dir_name: str = "labels"
    """Name of the directory containing label files"""
    image_dir_name: str = "image"
    """Name of the directory containing image files"""
    keypoint_dim: Literal[2, 3] = 3
    """2 - x, y; 3 - x, y, visibility"""
    keypoints_in_annotation: Optional[int] = None
    """Number of keypoints in each annotation"""

    @staticmethod
    def from_yaml_file(file_path: Union[str, Path]) -> "YoloContext":
        res = YoloContext()
        with open(file_path) as f:
            meta_dict = yaml.safe_load(f)
        res.categories = YoloContext._parse_categories(meta_dict)

        if "kpt_shape" in meta_dict:
            res.keypoints_in_annotation = meta_dict["kpt_shape"][0]
            res.keypoint_dim = meta_dict["kpt_shape"][1]
        return res

    @staticmethod
    def _parse_categories(yolo_meta: Dict) -> Categories:
        categories = Categories()
        for cat_id, cat_name in yolo_meta["names"].items():
            categories.add(cat_name, cat_id)
        return categories
