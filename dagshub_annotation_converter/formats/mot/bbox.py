import logging
from typing import Optional, Tuple

from dagshub_annotation_converter.formats.mot.context import MOTContext
from dagshub_annotation_converter.ir.video import CoordinateStyle, IRVideoBBoxFrameAnnotation

logger = logging.getLogger(__name__)


def import_bbox_from_line(line: str, context: MOTContext) -> Optional[Tuple[int, IRVideoBBoxFrameAnnotation]]:
    """
    Parse a single MOT line into an IRVideoBBoxAnnotation.

    CVAT MOT 1.1 ground truth format (9 columns):
    ``frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility``

    Example: ``1,1,1363,569,103,241,1,1,0.86014``

    MOT uses 1-based frame numbering; IR uses 0-based.
    """
    parts = line.strip().split(",")

    frame_id = int(parts[0])
    track_id = int(parts[1])
    x = float(parts[2])
    y = float(parts[3])
    w = float(parts[4])
    h = float(parts[5])
    not_ignored = int(parts[6])
    class_id = int(parts[7])
    visibility = float(parts[8])

    category_name = context.categories[class_id].name

    if not_ignored == 0:
        logger.warning(f"Skipping ignored annotation in frame {frame_id} track {track_id} category {category_name}")
        return None

    ann = IRVideoBBoxFrameAnnotation(
        frame_number=frame_id - 1,
        keyframe=False,
        left=x,
        top=y,
        width=w,
        height=h,
        video_width=context.video_width,
        video_height=context.video_height,
        categories={category_name: 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        visibility=visibility,
    )
    return track_id, ann


def _export_bbox_to_line(ann: IRVideoBBoxFrameAnnotation, track_id: int, context: MOTContext) -> str:
    """
    Export an IRVideoBBoxAnnotation to a MOT line.

    CVAT MOT 1.1 format (9 columns):
    ``frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility``

    MOT uses 1-based frame numbering; IR uses 0-based.
    """
    if ann.coordinate_style != CoordinateStyle.DENORMALIZED:
        raise ValueError("_export_bbox_to_line expects a denormalized video annotation")

    category_name = ann.ensure_has_one_category()
    class_id = context.categories[category_name].id
    not_ignored = 1

    x = int(ann.left) if ann.left == int(ann.left) else ann.left
    y = int(ann.top) if ann.top == int(ann.top) else ann.top
    w = int(ann.width) if ann.width == int(ann.width) else ann.width
    h = int(ann.height) if ann.height == int(ann.height) else ann.height

    mot_frame_id = ann.frame_number + 1

    return f"{mot_frame_id},{track_id},{x},{y},{w},{h},{not_ignored},{class_id},{ann.visibility}"
