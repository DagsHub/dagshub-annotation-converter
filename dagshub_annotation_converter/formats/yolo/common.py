from typing import Mapping, Any, Callable

from .context import YoloAnnotationTypes, YoloContext
from .bbox import export_bbox
from .segmentation import export_segmentation
from .pose import export_pose

from dagshub_annotation_converter.ir.image import (
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
    IRPoseImageAnnotation,
)

# Type actually has to be IRAnnotationBase, but it messes up MyPy
YoloExportFunctionType = Callable[[Any, YoloContext], str]

export_lookup: Mapping[YoloAnnotationTypes, YoloExportFunctionType] = {
    "bbox": export_bbox,
    "segmentation": export_segmentation,
    "pose": export_pose,
}

allowed_annotation_types = {
    "bbox": IRBBoxImageAnnotation,
    "segmentation": IRSegmentationImageAnnotation,
    "pose": IRPoseImageAnnotation,
}
