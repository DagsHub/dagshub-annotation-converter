from typing import Callable

from lxml.etree import ElementBase

from .box import parse_box
from .polygon import parse_polygon
from .points import parse_points
from .skeleton import parse_skeleton
from .context import CVATContext
from dagshub_annotation_converter.refactored.ir.image import IRAnnotationBase

CVATParserFunction = Callable[[CVATContext, ElementBase, ElementBase], IRAnnotationBase]

annotation_parsers: dict[str, CVATParserFunction] = {
    "box": parse_box,
    "polygon": parse_polygon,
    "points": parse_points,
    "skeleton": parse_skeleton,
}
