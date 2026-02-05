"""CVAT Video format parsing and export for video object tracking."""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from lxml import etree
from lxml.etree import ElementBase

from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle


def parse_video_track(
    track_elem: ElementBase,
    image_width: int,
    image_height: int,
) -> List[IRVideoBBoxAnnotation]:
    """
    Parse a CVAT video track element into a list of video bbox annotations.
    
    CVAT video XML track structure:
    <track id="0" label="person" source="manual">
      <box frame="0" outside="0" occluded="0" keyframe="1"
           xtl="100" ytl="150" xbr="150" ybr="270" z_order="0"/>
      ...
    </track>
    
    Args:
        track_elem: CVAT <track> XML element
        image_width: Frame width in pixels
        image_height: Frame height in pixels
        
    Returns:
        List of IRVideoBBoxAnnotation for each frame in the track
    """
    track_id = int(track_elem.attrib["id"])
    label = track_elem.attrib["label"]
    
    annotations = []
    
    for box_elem in track_elem.findall("box"):
        frame_number = int(box_elem.attrib["frame"])
        outside = int(box_elem.attrib.get("outside", 0))
        
        # Skip frames where object is outside (not visible)
        if outside == 1:
            continue
        
        occluded = int(box_elem.attrib.get("occluded", 0))
        keyframe = int(box_elem.attrib.get("keyframe", 0))
        
        # Parse bounding box coordinates
        xtl = float(box_elem.attrib["xtl"])
        ytl = float(box_elem.attrib["ytl"])
        xbr = float(box_elem.attrib["xbr"])
        ybr = float(box_elem.attrib["ybr"])
        
        # Convert to left, top, width, height
        left = xtl
        top = ytl
        width = xbr - xtl
        height = ybr - ytl
        
        # Visibility: 1.0 if not occluded, < 1.0 if occluded
        visibility = 0.5 if occluded == 1 else 1.0
        
        # Build metadata
        meta = {
            "keyframe": keyframe == 1,
            "z_order": int(box_elem.attrib.get("z_order", 0)),
        }
        
        ann = IRVideoBBoxAnnotation(
            track_id=track_id,
            frame_number=frame_number,
            left=left,
            top=top,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            categories={label: 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=visibility,
            confidence=1.0,
            meta=meta,
        )
        
        annotations.append(ann)
    
    return annotations


def parse_video_meta(meta_elem: ElementBase) -> tuple:
    """
    Parse CVAT video XML meta element for frame dimensions.
    
    Args:
        meta_elem: CVAT <meta> XML element
        
    Returns:
        Tuple of (width, height, seq_length) or (None, None, None) if not found
    """
    width = None
    height = None
    seq_length = None
    
    task_elem = meta_elem.find(".//task")
    if task_elem is not None:
        size_elem = task_elem.find("size")
        if size_elem is not None and size_elem.text:
            seq_length = int(size_elem.text)
        
        original_size = task_elem.find("original_size")
        if original_size is not None:
            width_elem = original_size.find("width")
            height_elem = original_size.find("height")
            if width_elem is not None and width_elem.text:
                width = int(width_elem.text)
            if height_elem is not None and height_elem.text:
                height = int(height_elem.text)
    
    return width, height, seq_length


def export_video_track_to_xml(
    track_id: int,
    annotations: Sequence[IRVideoBBoxAnnotation],
) -> ElementBase:
    """
    Export a track's annotations to a CVAT video XML track element.
    
    Args:
        track_id: Track ID
        annotations: List of annotations for this track (must all have same track_id)
        
    Returns:
        lxml Element representing the track
    """
    if not annotations:
        raise ValueError("Cannot create track from empty annotations list")
    
    # Get label from first annotation
    first_ann = annotations[0]
    label = first_ann.ensure_has_one_category()
    
    # Create track element
    track_elem = etree.Element("track")
    track_elem.set("id", str(track_id))
    track_elem.set("label", label)
    track_elem.set("source", "manual")
    
    # Sort annotations by frame number
    sorted_anns = sorted(annotations, key=lambda a: a.frame_number)
    
    for ann in sorted_anns:
        # Ensure denormalized coordinates
        if ann.coordinate_style == CoordinateStyle.NORMALIZED:
            ann = ann.denormalized()
        
        # Calculate box corners
        xtl = ann.left
        ytl = ann.top
        xbr = ann.left + ann.width
        ybr = ann.top + ann.height
        
        # Determine occluded and keyframe from meta
        occluded = 0
        if ann.visibility < 1.0:
            occluded = 1
        keyframe = 1 if ann.meta.get("keyframe", True) else 0
        z_order = ann.meta.get("z_order", 0)
        
        # Create box element
        box_elem = etree.SubElement(track_elem, "box")
        box_elem.set("frame", str(ann.frame_number))
        box_elem.set("outside", "0")
        box_elem.set("occluded", str(occluded))
        box_elem.set("keyframe", str(keyframe))
        box_elem.set("xtl", f"{xtl:.2f}")
        box_elem.set("ytl", f"{ytl:.2f}")
        box_elem.set("xbr", f"{xbr:.2f}")
        box_elem.set("ybr", f"{ybr:.2f}")
        box_elem.set("z_order", str(z_order))
    
    return track_elem


def build_cvat_video_xml(
    annotations: Sequence[IRVideoBBoxAnnotation],
    video_name: str = "video.mp4",
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
) -> ElementBase:
    """
    Build a complete CVAT video XML document from annotations.
    
    Args:
        annotations: List of video bbox annotations
        video_name: Name of the video file
        image_width: Frame width (inferred from annotations if not provided)
        image_height: Frame height (inferred from annotations if not provided)
        seq_length: Total number of frames (inferred from annotations if not provided)
        
    Returns:
        lxml Element representing the complete CVAT annotations document
    """
    # Infer dimensions and length from annotations if not provided
    if annotations:
        if image_width is None:
            image_width = annotations[0].image_width
        if image_height is None:
            image_height = annotations[0].image_height
        if seq_length is None:
            seq_length = max(ann.frame_number for ann in annotations) + 1
    else:
        image_width = image_width or 1920
        image_height = image_height or 1080
        seq_length = seq_length or 1
    
    # Create root element
    root = etree.Element("annotations")
    
    # Add version
    version_elem = etree.SubElement(root, "version")
    version_elem.text = "1.1"
    
    # Add meta section
    meta_elem = etree.SubElement(root, "meta")
    task_elem = etree.SubElement(meta_elem, "task")
    
    # Task mode
    mode_elem = etree.SubElement(task_elem, "mode")
    mode_elem.text = "interpolation"
    
    # Task size (number of frames)
    size_elem = etree.SubElement(task_elem, "size")
    size_elem.text = str(seq_length)
    
    # Original size
    orig_size_elem = etree.SubElement(task_elem, "original_size")
    width_elem = etree.SubElement(orig_size_elem, "width")
    width_elem.text = str(image_width)
    height_elem = etree.SubElement(orig_size_elem, "height")
    height_elem.text = str(image_height)
    
    # Collect unique labels
    labels: Dict[str, None] = {}
    for ann in annotations:
        label = ann.ensure_has_one_category()
        labels[label] = None
    
    # Add labels section
    labels_elem = etree.SubElement(task_elem, "labels")
    for label_name in labels:
        label_elem = etree.SubElement(labels_elem, "label")
        name_elem = etree.SubElement(label_elem, "name")
        name_elem.text = label_name
        type_elem = etree.SubElement(label_elem, "type")
        type_elem.text = "rectangle"
    
    # Add source info
    source_elem = etree.SubElement(task_elem, "source")
    source_elem.text = video_name
    
    # Group annotations by track_id
    tracks: Dict[int, List[IRVideoBBoxAnnotation]] = defaultdict(list)
    for ann in annotations:
        tracks[ann.track_id].append(ann)
    
    # Add track elements
    for track_id in sorted(tracks.keys()):
        track_anns = tracks[track_id]
        track_elem = export_video_track_to_xml(track_id, track_anns)
        root.append(track_elem)
    
    return root


def cvat_video_xml_to_string(
    annotations: Sequence[IRVideoBBoxAnnotation],
    video_name: str = "video.mp4",
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
) -> bytes:
    """
    Convert annotations to CVAT video XML string.
    
    Args:
        annotations: List of video bbox annotations
        video_name: Name of the video file
        image_width: Frame width (inferred from annotations if not provided)
        image_height: Frame height (inferred from annotations if not provided)
        seq_length: Total number of frames (inferred from annotations if not provided)
        
    Returns:
        XML content as bytes
    """
    root = build_cvat_video_xml(annotations, video_name, image_width, image_height, seq_length)
    return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="utf-8")
