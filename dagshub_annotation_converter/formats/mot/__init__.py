"""
MOT (Multiple Object Tracking) format support.

Uses CVAT MOT 1.1 format (9 columns):
frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility

Example: 1,1,1363,569,103,241,1,1,0.86014
"""

from dagshub_annotation_converter.formats.mot.context import MOTContext
from dagshub_annotation_converter.formats.mot.bbox import import_bbox_from_line, export_bbox_to_line

__all__ = [
    "MOTContext",
    "import_bbox_from_line",
    "export_bbox_to_line",
]
