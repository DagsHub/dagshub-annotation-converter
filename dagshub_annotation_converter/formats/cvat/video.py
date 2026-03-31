from typing import List, Optional, Tuple, Set

from lxml import etree
from lxml.etree import ElementBase

from dagshub_annotation_converter.ir.video import (
    CoordinateStyle,
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
    IRVideoSequence,
)


def _canonicalize_track_annotations(
    annotations: List[IRVideoBBoxFrameAnnotation],
) -> List[IRVideoBBoxFrameAnnotation]:
    canonical: List[IRVideoBBoxFrameAnnotation] = []

    for idx, ann in enumerate(annotations):
        prev_ann = annotations[idx - 1] if idx > 0 else None
        next_ann = annotations[idx + 1] if idx + 1 < len(annotations) else None

        if not ann.is_visible:
            continue

        keep = ann.keyframe or ann.visibility < 1.0
        if prev_ann is None or not prev_ann.is_visible:
            keep = True
        if next_ann is not None and not next_ann.is_visible:
            keep = True

        if not keep:
            continue

        canonical_ann = ann.model_copy(deep=True)
        if next_ann is not None and not next_ann.is_visible:
            canonical_ann.keyframe = False
        elif (
            ann.keyframe
            and next_ann is not None
            and next_ann.keyframe
            and next_ann.is_visible
            and next_ann.frame_number == ann.frame_number + 1
        ):
            canonical_ann.keyframe = False
        elif canonical_ann.visibility < 1.0:
            canonical_ann.keyframe = True

        canonical.append(canonical_ann)

    return canonical

def parse_video_track(
    track_elem: ElementBase,
    image_width: int,
    image_height: int,
) -> IRVideoAnnotationTrack:
    """
    Parse a CVAT video track element into video bbox annotations.

    CVAT video XML track structure::

        <track id="0" label="person" source="manual">
          <box frame="0" outside="0" occluded="0" keyframe="1"
               xtl="100" ytl="150" xbr="150" ybr="270" z_order="0"/>
          ...
        </track>
    """
    track_id = track_elem.attrib["id"]
    label = track_elem.attrib["label"]

    annotations = []
    box_elems = track_elem.findall("box")
    for box_elem in box_elems:
        frame_number = int(box_elem.attrib["frame"])
        outside = int(box_elem.attrib.get("outside", 0))
        occluded = int(box_elem.attrib.get("occluded", 0))
        keyframe = int(box_elem.attrib.get("keyframe", 0))

        xtl = float(box_elem.attrib["xtl"])
        ytl = float(box_elem.attrib["ytl"])
        xbr = float(box_elem.attrib["xbr"])
        ybr = float(box_elem.attrib["ybr"])

        if outside == 1:
            visibility = 0.0
        elif occluded == 1:
            visibility = 0.5
        else:
            visibility = 1.0

        meta = {"z_order": int(box_elem.attrib.get("z_order", 0))}

        ann = IRVideoBBoxFrameAnnotation(
            frame_number=frame_number,
            keyframe=keyframe == 1,
            left=xtl,
            top=ytl,
            width=xbr - xtl,
            height=ybr - ytl,
            video_width=image_width,
            video_height=image_height,
            categories={label: 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            visibility=visibility,
            meta=meta,
        )

        annotations.append(ann)

    canonical_annotations = _canonicalize_track_annotations(annotations)
    return IRVideoAnnotationTrack.from_annotations(canonical_annotations, track_id=track_id)


def parse_video_meta(meta_elem: ElementBase) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse CVAT video XML meta element for frame dimensions and sequence length."""
    width: Optional[int] = None
    height: Optional[int] = None
    seq_length: Optional[int] = None

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
    track: IRVideoAnnotationTrack,
    seq_length: Optional[int] = None,
    video_width: Optional[int] = None,
    video_height: Optional[int] = None,
) -> ElementBase:
    """Export a track's annotations to a CVAT video XML track element."""
    if not track.annotations:
        raise ValueError("Cannot create track from empty annotations list")

    track = track.denormalized(video_width=video_width, video_height=video_height)
    first_ann = track.annotations[0]
    label = first_ann.ensure_has_one_category()

    track_elem = etree.Element("track")
    track_elem.set("id", track.track_id)
    track_elem.set("label", label)
    track_elem.set("source", "manual")
    sorted_anns = sorted(
        (ann for ann in track.annotations if isinstance(ann, IRVideoBBoxFrameAnnotation)), key=lambda a: a.frame_number
    )
    for idx, ann in enumerate(sorted_anns):
        xtl = ann.left
        ytl = ann.top
        xbr = ann.left + ann.width
        ybr = ann.top + ann.height

        outside = 1 if ann.visibility == 0.0 else 0
        occluded = 0
        if not outside and ann.visibility < 1.0:
            occluded = 1
        keyframe = 1
        z_order = ann.meta.get("z_order", 0)

        box_elem = etree.SubElement(track_elem, "box")
        box_elem.set("frame", str(ann.frame_number))
        box_elem.set("outside", str(outside))
        box_elem.set("occluded", str(occluded))
        box_elem.set("keyframe", str(keyframe))
        box_elem.set("xtl", f"{xtl:.2f}")
        box_elem.set("ytl", f"{ytl:.2f}")
        box_elem.set("xbr", f"{xbr:.2f}")
        box_elem.set("ybr", f"{ybr:.2f}")
        box_elem.set("z_order", str(z_order))

        boundary_frame = ann.frame_number + 1
        max_frame = seq_length - 1 if isinstance(seq_length, int) and seq_length > 0 else None
        has_future_annotation = idx < len(sorted_anns) + 1
        has_known_room_for_boundary = max_frame is not None and boundary_frame <= max_frame
        should_add_stop_boundary = (
            not ann.keyframe and outside == 0 and (has_future_annotation or has_known_room_for_boundary)
        )

        if should_add_stop_boundary:
            has_next_on_boundary = idx + 1 < len(sorted_anns) and sorted_anns[idx + 1].frame_number == boundary_frame
            can_add_boundary = max_frame is None or boundary_frame <= max_frame
            if not has_next_on_boundary and can_add_boundary:
                boundary_elem = etree.SubElement(track_elem, "box")
                boundary_elem.set("frame", str(boundary_frame))
                boundary_elem.set("outside", "1")
                boundary_elem.set("occluded", "0")
                boundary_elem.set("keyframe", "1")
                boundary_elem.set("xtl", f"{xtl:.2f}")
                boundary_elem.set("ytl", f"{ytl:.2f}")
                boundary_elem.set("xbr", f"{xbr:.2f}")
                boundary_elem.set("ybr", f"{ybr:.2f}")
                boundary_elem.set("z_order", str(z_order))

    return track_elem


def build_cvat_video_xml(
    sequence: IRVideoSequence,
    video_name: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
) -> ElementBase:
    """Build a complete CVAT video XML document from annotations."""
    if sequence.tracks:
        if image_width is None:
            image_width = sequence.resolved_video_width()
        if image_height is None:
            image_height = sequence.resolved_video_height()
        if seq_length is None:
            seq_length = sequence.resolved_sequence_length()
    else:
        image_width = image_width or 1920
        image_height = image_height or 1080
        seq_length = seq_length or 1

    root = etree.Element("annotations")

    version_elem = etree.SubElement(root, "version")
    version_elem.text = "1.1"

    meta_elem = etree.SubElement(root, "meta")
    task_elem = etree.SubElement(meta_elem, "task")

    mode_elem = etree.SubElement(task_elem, "mode")
    mode_elem.text = "interpolation"

    size_elem = etree.SubElement(task_elem, "size")
    size_elem.text = str(seq_length)

    orig_size_elem = etree.SubElement(task_elem, "original_size")
    width_elem = etree.SubElement(orig_size_elem, "width")
    width_elem.text = str(image_width)
    height_elem = etree.SubElement(orig_size_elem, "height")
    height_elem.text = str(image_height)

    labels: Set[str] = set()
    for _, ann in sequence.iter_track_annotations():
        label = ann.ensure_has_one_category()
        labels.add(label)

    labels_elem = etree.SubElement(task_elem, "labels")
    for label_name in labels:
        label_elem = etree.SubElement(labels_elem, "label")
        name_elem = etree.SubElement(label_elem, "name")
        name_elem.text = label_name
        type_elem = etree.SubElement(label_elem, "type")
        type_elem.text = "rectangle"

    source_elem = etree.SubElement(task_elem, "source")
    source_elem.text = video_name

    for track in sequence.tracks:
        track_elem = export_video_track_to_xml(
            track,
            seq_length=seq_length,
            video_width=image_width,
            video_height=image_height,
        )
        root.append(track_elem)

    return root


def cvat_video_xml_to_string(
    sequence: IRVideoSequence,
    video_name: str,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
) -> bytes:
    root = build_cvat_video_xml(sequence, video_name, image_width, image_height, seq_length)
    return etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="utf-8")
