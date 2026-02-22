import logging
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Sequence, List, Dict, Union, Optional
from zipfile import ZipFile

import lxml.etree

from dagshub_annotation_converter.formats.cvat import annotation_parsers
from dagshub_annotation_converter.formats.cvat.context import parse_image_tag
from dagshub_annotation_converter.formats.cvat.video import (
    parse_video_track,
    parse_video_meta,
    cvat_video_xml_to_string,
)
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase, IRBBoxImageAnnotation, IRPoseImageAnnotation
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation
from dagshub_annotation_converter.util.video import get_video_dimensions, get_video_frame_count
from dagshub_annotation_converter.features import ConverterFeatures

logger = logging.getLogger(__name__)

# Type aliases for clarity
CVATImageAnnotations = Dict[str, Sequence[IRImageAnnotationBase]]
CVATVideoAnnotations = Dict[int, Sequence[IRVideoBBoxAnnotation]]
CVATAnnotations = Union[CVATImageAnnotations, CVATVideoAnnotations]


def _group_annotations_by_video_name(
    annotations: Sequence[IRVideoBBoxAnnotation],
    default_video_name: str,
) -> Dict[str, List[IRVideoBBoxAnnotation]]:
    grouped: Dict[str, List[IRVideoBBoxAnnotation]] = defaultdict(list)
    for ann in annotations:
        if ann.filename:
            grouped[Path(ann.filename).name].append(ann)
        else:
            grouped[default_video_name].append(ann)
    return grouped


def _resolve_video_name_for_export(
    annotations: Sequence[IRVideoBBoxAnnotation],
    video_name: str,
) -> str:
    source_names = sorted({Path(ann.filename).name for ann in annotations if ann.filename})
    if len(source_names) > 1:
        raise ValueError(
            "CVAT video export supports a single source video per export. "
            f"Found multiple annotation filenames: {', '.join(source_names)}"
        )
    if video_name == "video.mp4" and len(source_names) == 1:
        return source_names[0]
    return video_name


def parse_image_annotations(img: lxml.etree.ElementBase) -> Sequence[IRImageAnnotationBase]:
    annotations: List[IRImageAnnotationBase] = []
    for annotation_elem in img:
        annotation_type = annotation_elem.tag
        if annotation_type not in annotation_parsers:
            logger.warning(f"Unknown CVAT annotation type {annotation_type}")
            continue
        annotations.append(annotation_parsers[annotation_type](annotation_elem, img))

    annotations = _maybe_group_poses(annotations)

    return annotations


def _maybe_group_poses(annotations: List[IRImageAnnotationBase]) -> List[IRImageAnnotationBase]:
    if not ConverterFeatures.cvat_pose_grouping_by_group_id_enabled():
        return annotations
    res = []
    annotation_groups: Dict[str, List[IRImageAnnotationBase]] = defaultdict(list)
    for annotation in annotations:
        group_id = annotation.meta.get("group_id")
        if group_id is None:
            res.append(annotation)
        else:
            annotation_groups[group_id].append(annotation)

    for group_id, group_annotations in annotation_groups.items():
        if len(group_annotations) == 1:
            res.extend(group_annotations)
            continue

        bbox_count = sum((isinstance(ann, IRBBoxImageAnnotation) for ann in group_annotations))
        point_count = sum((isinstance(ann, IRPoseImageAnnotation) for ann in group_annotations))

        # If we have more than one bbox or point annotation in the group, don't bother trying to group
        if bbox_count != 1 or point_count != 1:
            res.extend(group_annotations)
            continue

        group_res = []
        bbox_ann: Optional[IRBBoxImageAnnotation] = None
        pose_ann: Optional[IRPoseImageAnnotation] = None

        for ann in group_annotations:
            if isinstance(ann, IRBBoxImageAnnotation):
                bbox_ann = ann
            elif isinstance(ann, IRPoseImageAnnotation):
                pose_ann = ann
            else:
                group_res.append(ann)

        assert bbox_ann is not None and pose_ann is not None

        # If there's somehow multiple labels (shouldn't be happening in CVAT), don't group
        if not (bbox_ann.has_one_category() and pose_ann.has_one_category()):
            res.extend(group_annotations)
            continue

        # Different categories - don't group
        if bbox_ann.ensure_has_one_category() != pose_ann.ensure_has_one_category():
            res.extend(group_annotations)
            continue

        pose_ann.width = bbox_ann.width
        pose_ann.height = bbox_ann.height
        pose_ann.top = bbox_ann.top
        pose_ann.left = bbox_ann.left

        group_res.append(pose_ann)
        res.extend(group_res)

    return res


def _detect_cvat_mode(root_elem: lxml.etree.ElementBase) -> str:
    """Detect CVAT annotation mode: returns ``"image"`` or ``"video"``."""
    mode_elem = root_elem.find(".//meta/task/mode")
    if mode_elem is not None and mode_elem.text:
        if mode_elem.text == "interpolation":
            return "video"
        elif mode_elem.text == "annotation":
            return "image"

    has_tracks = len(root_elem.findall(".//track")) > 0
    has_images = len(root_elem.findall(".//image")) > 0

    if has_tracks and not has_images:
        return "video"
    elif has_images and not has_tracks:
        return "image"
    elif has_tracks and has_images:
        logger.warning("CVAT XML contains both <track> and <image> elements, treating as video mode")
        return "video"
    else:
        return "image"


def _parse_image_mode(root_elem: lxml.etree.ElementBase) -> CVATImageAnnotations:
    annotations: CVATImageAnnotations = {}
    for image_node in root_elem.xpath("//image"):
        image_info = parse_image_tag(image_node)
        annotations[image_info.name] = parse_image_annotations(image_node)
    return annotations


def _parse_video_mode(
    root_elem: lxml.etree.ElementBase,
    image_width: Optional[int],
    image_height: Optional[int],
) -> CVATVideoAnnotations:
    if image_width is None or image_height is None:
        meta_elem = root_elem.find("meta")
        if meta_elem is not None:
            meta_width, meta_height, _ = parse_video_meta(meta_elem)
            if image_width is None:
                image_width = meta_width
            if image_height is None:
                image_height = meta_height

    if image_width is None or image_height is None:
        missing = []
        if image_width is None:
            missing.append("image_width")
        if image_height is None:
            missing.append("image_height")
        raise ValueError(
            f"Cannot determine frame dimensions for CVAT video annotations. "
            f"Missing: {', '.join(missing)}. "
            f"Provide {', '.join(missing)} explicitly."
        )

    all_annotations: Dict[int, List[IRVideoBBoxAnnotation]] = {}
    for track_elem in root_elem.findall(".//track"):
        track_annotations = parse_video_track(track_elem, image_width, image_height)
        for ann in track_annotations:
            if ann.frame_number not in all_annotations:
                all_annotations[ann.frame_number] = []
            all_annotations[ann.frame_number].append(ann)

    return all_annotations


def load_cvat_from_xml_string(
    xml_text: bytes,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> CVATAnnotations:
    """Load CVAT annotations from XML string, auto-detecting image or video mode."""
    root_elem = lxml.etree.XML(xml_text)
    mode = _detect_cvat_mode(root_elem)

    if mode == "video":
        return _parse_video_mode(root_elem, image_width, image_height)
    else:
        return _parse_image_mode(root_elem)


def load_cvat_from_xml_file(
    xml_file: Union[str, PathLike],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> CVATAnnotations:
    """Load CVAT annotations from XML file, auto-detecting image or video mode."""
    with open(xml_file, "rb") as f:
        return load_cvat_from_xml_string(f.read(), image_width, image_height)


def load_cvat_from_zip(
    zip_path: Union[str, PathLike],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> CVATAnnotations:
    """Load CVAT annotations from ZIP archive, auto-detecting image or video mode."""
    with ZipFile(zip_path) as proj_zip:
        with proj_zip.open("annotations.xml") as f:
            return load_cvat_from_xml_string(f.read(), image_width, image_height)


def load_cvat_from_fs(
    import_dir: Union[str, PathLike],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Dict[str, CVATAnnotations]:
    """Load CVAT annotations from all XML/ZIP files under a directory."""
    import_dir = Path(import_dir)
    results: Dict[str, CVATAnnotations] = {}

    for xml_path in sorted(import_dir.rglob("*.xml")):
        rel = str(xml_path.relative_to(import_dir))
        results[rel] = load_cvat_from_xml_file(xml_path, image_width, image_height)

    for zip_path in sorted(import_dir.rglob("*.zip")):
        rel = str(zip_path.relative_to(import_dir))
        results[rel] = load_cvat_from_zip(zip_path, image_width, image_height)

    return results


def export_cvat_video_to_xml_string(
    annotations: Sequence[IRVideoBBoxAnnotation],
    video_name: str = "video.mp4",
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_file: Optional[Union[str, PathLike]] = None,
) -> bytes:
    """Export video annotations to CVAT XML bytes, resolving dimensions from args/annotations/video_file."""
    resolved_video_name = _resolve_video_name_for_export(annotations, video_name)

    resolved_width = image_width if image_width is not None and image_width > 0 else None
    resolved_height = image_height if image_height is not None and image_height > 0 else None

    if resolved_width is None or resolved_height is None:
        for ann in annotations:
            if resolved_width is None and ann.image_width is not None and ann.image_width > 0:
                resolved_width = ann.image_width
            if resolved_height is None and ann.image_height is not None and ann.image_height > 0:
                resolved_height = ann.image_height
            if resolved_width is not None and resolved_height is not None:
                break

    if (resolved_width is None or resolved_height is None) and video_file is not None:
        probed_width, probed_height, _ = get_video_dimensions(Path(video_file))
        if resolved_width is None:
            resolved_width = probed_width
        if resolved_height is None:
            resolved_height = probed_height

    if seq_length is None and video_file is not None:
        frame_count = get_video_frame_count(Path(video_file))
        if frame_count is not None and frame_count > 0:
            seq_length = frame_count

    if resolved_width is None or resolved_height is None:
        raise ValueError(
            "Cannot determine frame dimensions for CVAT video export. "
            "Provide image_width/image_height, use annotations with valid dimensions, "
            "or provide video_file for probing."
        )

    prepared_annotations = []
    for ann in annotations:
        prepared = ann
        if prepared.image_width is None or prepared.image_width <= 0:
            prepared = prepared.model_copy()
            prepared.image_width = resolved_width
        if prepared.image_height is None or prepared.image_height <= 0:
            if prepared is ann:
                prepared = prepared.model_copy()
            prepared.image_height = resolved_height
        prepared_annotations.append(prepared)

    return cvat_video_xml_to_string(prepared_annotations, resolved_video_name, resolved_width, resolved_height, seq_length)


def export_cvat_video_to_file(
    annotations: Sequence[IRVideoBBoxAnnotation],
    output_path: Union[str, PathLike],
    video_name: str = "video.mp4",
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_file: Optional[Union[str, PathLike]] = None,
) -> Path:
    """Export video annotations to a CVAT XML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_content = export_cvat_video_to_xml_string(
        annotations, video_name, image_width, image_height, seq_length, video_file
    )

    with open(output_path, "wb") as f:
        f.write(xml_content)

    logger.info(f"Exported {len(annotations)} CVAT video annotations to {output_path}")
    return output_path


def export_cvat_video_to_zip(
    annotations: Sequence[IRVideoBBoxAnnotation],
    output_path: Union[str, PathLike],
    video_name: str = "video.mp4",
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_file: Optional[Union[str, PathLike]] = None,
) -> Path:
    """Export video annotations to a CVAT-compatible ZIP containing ``annotations.xml``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w") as z:
        grouped = _group_annotations_by_video_name(annotations, video_name)

        # Single-video behavior remains identical and fully CVAT-compatible.
        if len(grouped) == 1:
            xml_content = export_cvat_video_to_xml_string(
                annotations, video_name, image_width, image_height, seq_length, video_file
            )
            z.writestr("annotations.xml", xml_content)
        else:
            if video_file is not None:
                raise ValueError(
                    "video_file is ambiguous for multi-video CVAT zip export. "
                    "Provide explicit dimensions or export per video."
                )
            for group_video_name, group_annotations in sorted(grouped.items()):
                xml_content = export_cvat_video_to_xml_string(
                    group_annotations,
                    group_video_name,
                    image_width,
                    image_height,
                    seq_length,
                    None,
                )
                # Store one XML per source video under its own folder.
                z.writestr(f"{group_video_name}/annotations.xml", xml_content)

    logger.info(f"Exported CVAT video annotations to {output_path}")
    return output_path


def export_cvat_videos_to_zips(
    annotations: Sequence[IRVideoBBoxAnnotation],
    output_dir: Union[str, PathLike],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_files: Optional[Dict[str, Union[str, PathLike]]] = None,
) -> List[Path]:
    """Export grouped video annotations to one CVAT zip per source video."""
    def resolve_video_file(video_name: str) -> Optional[Union[str, PathLike]]:
        if video_files is None:
            return None
        if video_name in video_files:
            return video_files[video_name]
        video_stem = Path(video_name).stem
        if video_stem in video_files:
            return video_files[video_stem]
        video_basename = Path(video_name).name
        if video_basename in video_files:
            return video_files[video_basename]
        for key, value in video_files.items():
            key_path = Path(key)
            if key_path.name == video_name or key_path.name == video_basename:
                return value
            if key_path.stem == video_stem:
                return value
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _group_annotations_by_video_name(annotations, "video.mp4")
    outputs: List[Path] = []
    for video_name, group_annotations in sorted(grouped.items()):
        zip_name = f"{video_name}.zip"
        output_path = output_dir / zip_name
        export_cvat_video_to_zip(
            group_annotations,
            output_path,
            video_name=video_name,
            image_width=image_width,
            image_height=image_height,
            seq_length=seq_length,
            video_file=resolve_video_file(video_name),
        )
        outputs.append(output_path)
    return outputs
