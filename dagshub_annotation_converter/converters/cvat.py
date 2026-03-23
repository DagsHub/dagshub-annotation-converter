import logging
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from zipfile import ZipFile

import lxml.etree

from dagshub_annotation_converter.features import ConverterFeatures
from dagshub_annotation_converter.formats.cvat import annotation_parsers
from dagshub_annotation_converter.formats.cvat.context import parse_image_tag
from dagshub_annotation_converter.formats.cvat.video import (
    cvat_video_xml_to_string,
    parse_video_meta,
    parse_video_track,
)
from dagshub_annotation_converter.ir.image import IRBBoxImageAnnotation, IRImageAnnotationBase, IRPoseImageAnnotation
from dagshub_annotation_converter.ir.video import IRVideoAnnotationTrack, IRVideoSequence
from dagshub_annotation_converter.util.video import get_video_dimensions, get_video_frame_count

logger = logging.getLogger(__name__)

CVATImageAnnotations = Dict[str, Sequence[IRImageAnnotationBase]]
CVATVideoAnnotations = IRVideoSequence
CVATAnnotations = Union[CVATImageAnnotations, CVATVideoAnnotations]


def _resolve_video_name_for_export(
    sequence: IRVideoSequence,
    video_name: Optional[str],
    video_file: Optional[Union[str, PathLike]],
) -> str:
    if video_name:
        return video_name
    if sequence.filename:
        return Path(sequence.filename).name
    if video_file is not None:
        return Path(video_file).name
    raise ValueError(
        "Cannot determine video name for CVAT video export. "
        "Provide video_name explicitly, set sequence.filename, or provide video_file."
    )


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

        if not (bbox_ann.has_one_category() and pose_ann.has_one_category()):
            res.extend(group_annotations)
            continue

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
        if mode_elem.text == "annotation":
            return "image"

    has_tracks = len(root_elem.findall(".//track")) > 0
    has_images = len(root_elem.findall(".//image")) > 0

    if has_tracks and not has_images:
        return "video"
    if has_images and not has_tracks:
        return "image"
    if has_tracks and has_images:
        logger.warning("CVAT XML contains both <track> and <image> elements, treating as video mode")
        return "video"
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
    seq_length: Optional[int] = None
    source_name: Optional[str] = None
    meta_elem = root_elem.find("meta")
    if meta_elem is not None:
        meta_width, meta_height, seq_length = parse_video_meta(meta_elem)
        if image_width is None:
            image_width = meta_width
        if image_height is None:
            image_height = meta_height
        source_elem = meta_elem.find(".//task/source")
        if source_elem is not None and source_elem.text:
            source_name = source_elem.text

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

    tracks: List[IRVideoAnnotationTrack] = []
    for track_elem in root_elem.findall(".//track"):
        tracks.append(parse_video_track(track_elem, image_width, image_height))

    return IRVideoSequence(
        tracks=tracks,
        filename=source_name,
        sequence_length=seq_length,
        video_width=image_width,
        video_height=image_height,
    )


def load_cvat_from_xml_bytes(
    xml_bytes: bytes,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> CVATAnnotations:
    """Load CVAT annotations from XML string, auto-detecting image or video mode."""
    root_elem = lxml.etree.XML(xml_bytes)
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
        return load_cvat_from_xml_bytes(f.read(), image_width, image_height)


def load_cvat_from_zip(
    zip_path: Union[str, PathLike],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> CVATAnnotations:
    """Load CVAT annotations from ZIP archive, auto-detecting image or video mode."""
    with ZipFile(zip_path) as proj_zip:
        with proj_zip.open("annotations.xml") as f:
            return load_cvat_from_xml_bytes(f.read(), image_width, image_height)


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


def export_cvat_video_to_xml_bytes(
    sequence: IRVideoSequence,
    video_name: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_file: Optional[Union[str, PathLike]] = None,
) -> bytes:
    """Export video annotations to CVAT XML bytes, resolving dimensions from args/sequence/video_file."""
    resolved_video_name = _resolve_video_name_for_export(sequence, video_name, video_file)

    resolved_width = image_width if image_width is not None and image_width > 0 else None
    resolved_height = image_height if image_height is not None and image_height > 0 else None

    if resolved_width is None or resolved_height is None:
        for _, ann in sequence.iter_track_annotations():
            if resolved_width is None and ann.video_width is not None and ann.video_width > 0:
                resolved_width = ann.video_width
            if resolved_height is None and ann.video_height is not None and ann.video_height > 0:
                resolved_height = ann.video_height
            if resolved_width is not None and resolved_height is not None:
                break

    if (resolved_width is None or resolved_height is None) and video_file is not None:
        try:
            probed_width, probed_height, _ = get_video_dimensions(Path(video_file))
            if resolved_width is None:
                resolved_width = probed_width
            if resolved_height is None:
                resolved_height = probed_height
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not probe video dimensions from {video_file}: {e}")

    if seq_length is None and video_file is not None:
        try:
            frame_count = get_video_frame_count(Path(video_file))
            if frame_count is not None and frame_count > 0:
                seq_length = frame_count
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not probe video frame count from {video_file}: {e}")

    if resolved_width is None or resolved_height is None:
        raise ValueError(
            "Cannot determine frame dimensions for CVAT video export. "
            "Provide image_width/image_height, use annotations with valid dimensions, "
            "or provide video_file for probing."
        )

    if seq_length is None:
        seq_length = sequence.resolved_sequence_length()

    prepared_sequence = sequence.model_copy(deep=True)
    prepared_sequence.video_width = resolved_width
    prepared_sequence.video_height = resolved_height
    if seq_length is not None and seq_length > 0:
        prepared_sequence.sequence_length = seq_length

    for _, ann in prepared_sequence.iter_track_annotations():
        if ann.video_width is None or ann.video_width <= 0:
            ann.video_width = resolved_width
        if ann.video_height is None or ann.video_height <= 0:
            ann.video_height = resolved_height

    return cvat_video_xml_to_string(
        prepared_sequence,
        resolved_video_name,
        resolved_width,
        resolved_height,
        seq_length,
    )


def export_cvat_video_to_file(
    sequence: IRVideoSequence,
    output_path: Union[str, PathLike],
    video_name: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_file: Optional[Union[str, PathLike]] = None,
) -> Path:
    """Export video annotations to a CVAT XML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_content = export_cvat_video_to_xml_bytes(
        sequence,
        video_name,
        image_width,
        image_height,
        seq_length,
        video_file,
    )

    with open(output_path, "wb") as f:
        f.write(xml_content)

    logger.info(f"Exported {len(sequence.tracks)} CVAT video tracks to {output_path}")
    return output_path


def export_cvat_video_to_zip(
    sequence: IRVideoSequence,
    output_path: Union[str, PathLike],
    video_name: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_file: Optional[Union[str, PathLike]] = None,
) -> Path:
    """Export video annotations to a CVAT-compatible ZIP containing ``annotations.xml``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, "w") as z:
        xml_content = export_cvat_video_to_xml_bytes(
            sequence,
            video_name,
            image_width,
            image_height,
            seq_length,
            video_file,
        )
        z.writestr("annotations.xml", xml_content)

    logger.info(f"Exported CVAT video annotations to {output_path}")
    return output_path


def export_cvat_videos_to_zips(
    sequences: Sequence[IRVideoSequence],
    output_dir: Union[str, PathLike],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    seq_length: Optional[int] = None,
    video_files: Optional[Dict[str, Union[str, PathLike]]] = None,
) -> List[Path]:
    """Export one CVAT zip per video sequence."""

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

    outputs: List[Path] = []
    for sequence in sequences:
        if not sequence.filename:
            raise ValueError(
                "Cannot determine video name for CVAT multi-export. "
                "Each sequence must have sequence.filename set."
            )
        video_name = Path(sequence.filename).name
        zip_name = f"{video_name}.zip"
        output_path = output_dir / zip_name
        export_cvat_video_to_zip(
            sequence,
            output_path,
            video_name=video_name,
            image_width=image_width,
            image_height=image_height,
            seq_length=seq_length,
            video_file=resolve_video_file(video_name),
        )
        outputs.append(output_path)
    return outputs
