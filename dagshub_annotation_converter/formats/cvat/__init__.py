from typing import Callable, Dict

from lxml.etree import ElementBase

from .box import parse_box
from .ellipse import parse_ellipse
from .polygon import parse_polygon
from .points import parse_points
from .skeleton import parse_skeleton
from .video import (
    parse_video_track,
    parse_video_meta,
    export_video_track_to_xml,
    build_cvat_video_xml,
    cvat_video_xml_to_string,
)
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase

CVATParserFunction = Callable[[ElementBase, ElementBase], IRImageAnnotationBase]

annotation_parsers: Dict[str, CVATParserFunction] = {
    "box": parse_box,
    "polygon": parse_polygon,
    "points": parse_points,
    "skeleton": parse_skeleton,
    "ellipse": parse_ellipse,
}

__all__ = [
    "annotation_parsers",
    parse_box.__name__,
    parse_ellipse.__name__,
    parse_polygon.__name__,
    parse_points.__name__,
    parse_skeleton.__name__,
    parse_video_track.__name__,
    parse_video_meta.__name__,
    export_video_track_to_xml.__name__,
    build_cvat_video_xml.__name__,
    cvat_video_xml_to_string.__name__,
]
