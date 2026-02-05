"""MOT bounding box import/export functions."""

from dagshub_annotation_converter.formats.mot.context import MOTContext
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle


def import_bbox_from_line(line: str, context: MOTContext) -> IRVideoBBoxAnnotation:
    """
    Parse a single MOT line into an IRVideoBBoxAnnotation.
    
    CVAT MOT 1.1 format (9 columns):
    frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    
    Example: 1,1,1363,569,103,241,1,1,0.86014
    
    Note: MOT uses 1-based frame numbering, IR uses 0-based.
    
    Args:
        line: Single line from gt.txt file
        context: MOT context with categories and image dimensions
        
    Returns:
        IRVideoBBoxAnnotation with parsed values (0-based frame number)
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
    
    # Get category name from context
    category_name = context.get_category_name(class_id)
    
    # Build metadata
    meta = {}
    if not_ignored == 0:
        meta["ignored"] = True
    
    return IRVideoBBoxAnnotation(
        track_id=track_id,
        frame_number=frame_id - 1,  # Convert 1-based MOT to 0-based IR
        left=x,
        top=y,
        width=w,
        height=h,
        image_width=context.image_width or 0,
        image_height=context.image_height or 0,
        categories={category_name: 1.0},
        coordinate_style=CoordinateStyle.DENORMALIZED,
        visibility=visibility,
        confidence=1.0,  # Ground truth confidence
        meta=meta,
    )


def export_bbox_to_line(ann: IRVideoBBoxAnnotation, context: MOTContext) -> str:
    """
    Export an IRVideoBBoxAnnotation to a MOT line.
    
    CVAT MOT 1.1 format (9 columns):
    frame_id, track_id, x, y, w, h, "not ignored", class_id, visibility
    
    Note: MOT uses 1-based frame numbering, IR uses 0-based.
    
    Args:
        ann: Video bounding box annotation (0-based frame number)
        context: MOT context with categories mapping
        
    Returns:
        MOT format line string (1-based frame number)
    """
    # Ensure denormalized coordinates
    if ann.coordinate_style == CoordinateStyle.NORMALIZED:
        ann = ann.denormalized()
    
    # Get class_id from category
    category_name = ann.ensure_has_one_category()
    class_id = context.get_class_id(category_name)
    
    # Check if annotation is ignored
    not_ignored = 0 if ann.meta.get("ignored", False) else 1
    
    # Format values - use integers for coordinates if they're whole numbers
    x = int(ann.left) if ann.left == int(ann.left) else ann.left
    y = int(ann.top) if ann.top == int(ann.top) else ann.top
    w = int(ann.width) if ann.width == int(ann.width) else ann.width
    h = int(ann.height) if ann.height == int(ann.height) else ann.height
    
    # Convert 0-based IR frame to 1-based MOT frame
    mot_frame_id = ann.frame_number + 1
    
    return f"{mot_frame_id},{ann.track_id},{x},{y},{w},{h},{not_ignored},{class_id},{ann.visibility}"
