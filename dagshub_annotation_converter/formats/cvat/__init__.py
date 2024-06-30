from typing import Callable

from lxml.etree import ElementBase

from .box import parse_box
from .polygon import parse_polygon
from .points import parse_points
from .skeleton import parse_skeleton
from dagshub_annotation_converter.ir.image import IRAnnotationBase

CVATParserFunction = Callable[[ElementBase, ElementBase], IRAnnotationBase]

annotation_parsers: dict[str, CVATParserFunction] = {
    "box": parse_box,
    "polygon": parse_polygon,
    "points": parse_points,
    "skeleton": parse_skeleton,
}