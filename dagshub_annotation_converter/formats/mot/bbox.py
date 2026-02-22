from dagshub_annotation_converter.formats.mot.context import MOTContext
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle


def import_bbox_from_line(line: str, context: MOTContext) -> IRVideoBBoxAnnotation:
    """
    Parse a single MOT line into an IRVideoBBoxAnnotation.

    CVAT MOT 1.1 format (9 columns):
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

    category_name = context.get_category_name(class_id)

    meta = {"source_format": "mot"}
    if not_ignored == 0:
        meta["ignored"] = True
    if visibility <= 0.0:
        meta["outside"] = True

    return IRVideoBBoxAnnotation(
        track_id=track_id,
        frame_number=frame_id - 1,
        keyframe=True,
        left=x,
        top=y,
        width=w,
        height=h,
        image_width=context.image_width,
        image_height=context.image_height,
        categories={category_name: 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        visibility=visibility,
        meta=meta,
    )


def export_bbox_to_line(ann: IRVideoBBoxAnnotation, context: MOTContext) -> str:
    """
    Export an IRVideoBBoxAnnotation to a MOT line.

    CVAT MOT 1.1 format (9 columns):
    ``frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility``

    MOT uses 1-based frame numbering; IR uses 0-based.
    """
    if ann.coordinate_style == CoordinateStyle.NORMALIZED:
        if (
            (ann.image_width is None and context.image_width is not None)
            or (ann.image_height is None and context.image_height is not None)
        ):
            ann = ann.model_copy()
            if ann.image_width is None and context.image_width is not None:
                ann.image_width = context.image_width
            if ann.image_height is None and context.image_height is not None:
                ann.image_height = context.image_height
        ann = ann.denormalized()

    category_name = ann.ensure_has_one_category()
    class_id = context.get_class_id(category_name)
    not_ignored = 0 if ann.meta.get("ignored", False) else 1

    x = int(ann.left) if ann.left == int(ann.left) else ann.left
    y = int(ann.top) if ann.top == int(ann.top) else ann.top
    w = int(ann.width) if ann.width == int(ann.width) else ann.width
    h = int(ann.height) if ann.height == int(ann.height) else ann.height

    mot_frame_id = ann.frame_number + 1

    return f"{mot_frame_id},{ann.track_id},{x},{y},{w},{h},{not_ignored},{class_id},{ann.visibility}"
