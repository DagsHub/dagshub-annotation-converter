from typing import Callable, Mapping, Any

from dagshub_annotation_converter.formats.yolo.bbox import export_bbox
from dagshub_annotation_converter.formats.yolo.context import YoloContext, YoloAnnotationTypes
from dagshub_annotation_converter.formats.yolo.pose import export_pose
from dagshub_annotation_converter.formats.yolo.segmentation import export_segmentation
from dagshub_annotation_converter.ir.image import (
    IRBBoxAnnotation,
    IRSegmentationAnnotation,
    IRPoseAnnotation,
)

# Type actually has to be IRAnnotationBase, but it messes up MyPy
YoloExportFunctionType = Callable[[Any, YoloContext], str]

export_lookup: Mapping[YoloAnnotationTypes, YoloExportFunctionType] = {
    "bbox": export_bbox,
    "segmentation": export_segmentation,
    "pose": export_pose,
}

allowed_annotation_types = {
    "bbox": IRBBoxAnnotation,
    "segmentation": IRSegmentationAnnotation,
    "pose": IRPoseAnnotation,
}
