import logging
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

from dagshub_annotation_converter.formats.mot import MOTContext, import_bbox_from_line, export_bbox_to_line
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation, CoordinateStyle
from dagshub_annotation_converter.util.video import (
    find_video_sibling,
    get_video_dimensions,
    get_video_frame_count,
)

logger = logging.getLogger(__name__)


def _is_safe_zip_path(name: str) -> bool:
    """Reject path traversal (Zip Slip)."""
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def _find_mot_prefix_in_zip(z: ZipFile) -> str:
    """Find the MOT root prefix (e.g. '' or 'seqname/') from zip entries."""
    names = [n for n in z.namelist() if _is_safe_zip_path(n)]
    if "gt/gt.txt" in names:
        return ""
    for name in names:
        parts = name.split("/")
        if len(parts) >= 3 and parts[-2] == "gt" and parts[-1] == "gt.txt":
            prefix = "/".join(parts[:-2]) + "/"
            return prefix
    raise FileNotFoundError("Could not find gt/gt.txt in zip")


def load_mot_from_file(
    gt_path: Union[str, Path],
    context: MOTContext,
) -> Dict[int, Sequence[IRVideoBBoxAnnotation]]:
    """Load MOT annotations from a gt.txt file."""
    gt_path = Path(gt_path)
    annotations: Dict[int, List[IRVideoBBoxAnnotation]] = {}
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ann = import_bbox_from_line(line, context)
            if ann.frame_number not in annotations:
                annotations[ann.frame_number] = []
            annotations[ann.frame_number].append(ann)
    return annotations


def _try_fill_dimensions_from_video(
    context: MOTContext,
    source_path: Path,
    video_file: Optional[Union[str, Path]],
) -> None:
    """Try to fill missing context dimensions from a video file.

    Looks for a video at *video_file* (if given), then falls back to a
    sibling of *source_path* with the same stem and a common video extension.
    """
    if context.image_width is not None and context.image_height is not None:
        return

    candidates: List[Path] = []
    if video_file is not None:
        vf = Path(video_file)
        if vf.is_file():
            candidates.append(vf)
        else:
            sibling = source_path.parent / vf.name if not vf.is_absolute() else vf
            if sibling.is_file():
                candidates.append(sibling)

    sibling = find_video_sibling(source_path)
    if sibling is not None:
        candidates.append(sibling)

    for candidate in candidates:
        try:
            width, height, fps = get_video_dimensions(candidate)
            if context.image_width is None:
                context.image_width = width
            if context.image_height is None:
                context.image_height = height
            if context.frame_rate == 30.0 and fps > 0:
                context.frame_rate = fps
            if context.seq_length is None:
                frame_count = get_video_frame_count(candidate)
                if frame_count is not None and frame_count > 0:
                    context.seq_length = frame_count
            logger.info(f"Inferred dimensions {width}x{height} from {candidate}")
            return
        except (ImportError, ValueError) as e:
            logger.debug(f"Could not read video {candidate}: {e}")


def _validate_context_dimensions(context: MOTContext, source: str) -> None:
    if context.image_width is None or context.image_height is None:
        missing = []
        if context.image_width is None:
            missing.append("image_width")
        if context.image_height is None:
            missing.append("image_height")
        raise ValueError(
            f"MOT annotations from {source} require frame dimensions, but "
            f"{', '.join(missing)} could not be determined. "
            f"Provide {', '.join(missing)} explicitly, or place a video file "
            f"with the same name next to the annotation source."
        )


def _to_denormalized_for_mot(ann: IRVideoBBoxAnnotation, context: MOTContext) -> IRVideoBBoxAnnotation:
    if ann.coordinate_style == CoordinateStyle.NORMALIZED:
        if (
            (ann.video_width is None and context.image_width is not None)
            or (ann.video_height is None and context.image_height is not None)
        ):
            ann = ann.model_copy()
            if ann.video_width is None and context.image_width is not None:
                ann.video_width = context.image_width
            if ann.video_height is None and context.image_height is not None:
                ann.video_height = context.image_height
        ann = ann.denormalized()
    return ann


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _is_outside(ann: IRVideoBBoxAnnotation) -> bool:
    return bool(ann.meta.get("outside", False))


def _interpolation_enabled(ann: IRVideoBBoxAnnotation) -> bool:
    return bool(ann.meta.get("ls_enabled", True))


def _interpolate_track_for_mot(
    track_annotations: Sequence[IRVideoBBoxAnnotation],
    context: MOTContext,
    end_frame: Optional[int] = None,
) -> List[IRVideoBBoxAnnotation]:
    if not track_annotations:
        return []

    by_frame: Dict[int, IRVideoBBoxAnnotation] = {}
    for ann in sorted(track_annotations, key=lambda a: a.frame_number):
        by_frame[ann.frame_number] = _to_denormalized_for_mot(ann, context)

    ordered = [by_frame[frame] for frame in sorted(by_frame)]
    dense: List[IRVideoBBoxAnnotation] = []

    for idx, curr in enumerate(ordered):
        curr_outside = _is_outside(curr)
        dense.append(curr)

        if idx == len(ordered) - 1:
            continue

        nxt = ordered[idx + 1]
        gap = nxt.frame_number - curr.frame_number - 1
        if gap <= 0 or curr_outside or not _interpolation_enabled(curr):
            continue

        curr_ignored = bool(curr.meta.get("ignored", False))
        nxt_ignored = bool(nxt.meta.get("ignored", False))

        for step in range(1, gap + 1):
            t = step / (gap + 1)
            interpolated = curr.model_copy(deep=True)
            interpolated.frame_number = curr.frame_number + step
            interpolated.keyframe = False
            interpolated.left = _lerp(curr.left, nxt.left, t)
            interpolated.top = _lerp(curr.top, nxt.top, t)
            interpolated.width = _lerp(curr.width, nxt.width, t)
            interpolated.height = _lerp(curr.height, nxt.height, t)
            interpolated.rotation = _lerp(curr.rotation, nxt.rotation, t)
            interpolated.visibility = _lerp(curr.visibility, nxt.visibility, t)
            if curr.timestamp is not None and nxt.timestamp is not None:
                interpolated.timestamp = _lerp(curr.timestamp, nxt.timestamp, t)
            else:
                interpolated.timestamp = None
            interpolated.meta.pop("outside", None)
            if curr_ignored and nxt_ignored:
                interpolated.meta["ignored"] = True
            else:
                interpolated.meta.pop("ignored", None)
            dense.append(interpolated)

    if end_frame is not None:
        last = ordered[-1]
        if not _is_outside(last) and _interpolation_enabled(last) and last.frame_number < end_frame:
            for frame_number in range(last.frame_number + 1, end_frame + 1):
                extrapolated = last.model_copy(deep=True)
                extrapolated.frame_number = frame_number
                extrapolated.keyframe = False
                extrapolated.meta.pop("outside", None)
                dense.append(extrapolated)

    return dense


def load_mot_from_dir(
    mot_dir: Union[str, Path],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    video_file: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[int, Sequence[IRVideoBBoxAnnotation]], MOTContext]:
    """
    Load MOT annotations from a directory structure.

    Expected structure::

        mot_dir/
          gt/
            gt.txt
            labels.txt (optional)
          seqinfo.ini (optional)

    If dimensions are missing from seqinfo.ini and not provided explicitly,
    falls back to probing *video_file* (if given) or a video with the same
    name as *mot_dir* located next to it.
    """
    mot_dir = Path(mot_dir)
    gt_dir = mot_dir / "gt"

    seqinfo_path = mot_dir / "seqinfo.ini"
    if seqinfo_path.exists():
        context = MOTContext.from_seqinfo_string(seqinfo_path.read_text(encoding="utf-8"))
    else:
        context = MOTContext()
        logger.warning(f"seqinfo.ini not found at {seqinfo_path}, using default context")

    if image_width is not None:
        context.image_width = image_width
    if image_height is not None:
        context.image_height = image_height

    labels_path = gt_dir / "labels.txt"
    if labels_path.exists():
        context.categories = MOTContext.load_labels(labels_path)

    gt_path = gt_dir / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Could not find gt.txt in {gt_dir}")

    _try_fill_dimensions_from_video(context, mot_dir, video_file)
    _validate_context_dimensions(context, str(mot_dir))
    annotations = load_mot_from_file(gt_path, context)
    return annotations, context


def _load_mot_from_gt_content(gt_content: str, context: MOTContext) -> Dict[int, Sequence[IRVideoBBoxAnnotation]]:
    annotations: Dict[int, List[IRVideoBBoxAnnotation]] = {}
    for line in gt_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ann = import_bbox_from_line(line, context)
        if ann.frame_number not in annotations:
            annotations[ann.frame_number] = []
        annotations[ann.frame_number].append(ann)
    return annotations


def load_mot_from_zip(
    zip_path: Union[str, Path],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    video_file: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[int, Sequence[IRVideoBBoxAnnotation]], MOTContext]:
    """
    Load MOT annotations from a ZIP archive (no extraction, avoids Zip Slip).

    Expected ZIP structure::

        gt/
          gt.txt
          labels.txt (optional)
        seqinfo.ini (optional)

    Or nested: ``seqname/gt/gt.txt``, ``seqname/seqinfo.ini``, etc.

    If dimensions are missing and not provided explicitly, falls back to
    probing *video_file* (if given) or a video with the same stem as the
    zip located in the same directory.
    """
    zip_path = Path(zip_path)
    with ZipFile(zip_path) as z:
        prefix = _find_mot_prefix_in_zip(z)
        gt_key = f"{prefix}gt/gt.txt"
        labels_key = f"{prefix}gt/labels.txt"
        seqinfo_key = f"{prefix}seqinfo.ini"

        if gt_key not in z.namelist():
            raise FileNotFoundError(f"Could not find gt/gt.txt in {zip_path}")

        if seqinfo_key in z.namelist():
            with z.open(seqinfo_key) as f:
                context = MOTContext.from_seqinfo_string(f.read().decode("utf-8"))
        else:
            context = MOTContext()
            logger.warning("seqinfo.ini not found in zip, using default context")

        if image_width is not None:
            context.image_width = image_width
        if image_height is not None:
            context.image_height = image_height

        if labels_key in z.namelist():
            with z.open(labels_key) as f:
                context.categories = MOTContext.load_labels_from_string(f.read().decode("utf-8"))

        _try_fill_dimensions_from_video(context, zip_path, video_file)
        _validate_context_dimensions(context, str(zip_path))

        with z.open(gt_key) as f:
            gt_content = f.read().decode("utf-8")
        annotations = _load_mot_from_gt_content(gt_content, context)

    return annotations, context


def load_mot_from_fs(
    import_dir: Union[str, Path],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    video_files: Optional[Dict[str, Union[str, Path]]] = None,
    datasource_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Tuple[Dict[int, Sequence[IRVideoBBoxAnnotation]], MOTContext]]:
    """Load MOT annotations from all sequence directories/ZIPs under a directory."""
    import_dir = Path(import_dir)
    results: Dict[str, Tuple[Dict[int, Sequence[IRVideoBBoxAnnotation]], MOTContext]] = {}
    normalized_datasource_path: Optional[Path] = None
    if datasource_path is not None:
        normalized_datasource_path = Path(str(datasource_path).lstrip("/"))

    def _sequence_lookup_keys(sequence_key: str) -> List[str]:
        seq_path = Path(sequence_key)
        keys = [sequence_key, seq_path.as_posix(), seq_path.name, seq_path.stem]
        second_stem = Path(seq_path.stem).stem
        if second_stem not in keys:
            keys.append(second_stem)
        return [key for key in keys if key]

    def _sequence_video_filename(sequence_key: str) -> str:
        name = Path(sequence_key).name
        if name.lower().endswith(".zip"):
            return Path(name).stem
        return name

    def _resolve_video_file(sequence_key: str) -> Optional[Path]:
        lookup_keys = _sequence_lookup_keys(sequence_key)

        if video_files is not None:
            for key in lookup_keys:
                if key in video_files:
                    return Path(video_files[key])

            for raw_key, raw_value in video_files.items():
                key_path = Path(str(raw_key))
                candidates = [
                    str(raw_key),
                    key_path.as_posix(),
                    key_path.name,
                    key_path.stem,
                    Path(key_path.stem).stem,
                ]
                if any(candidate in lookup_keys for candidate in candidates if candidate):
                    return Path(raw_value)

        if normalized_datasource_path is not None:
            video_filename = _sequence_video_filename(sequence_key)
            candidate = import_dir.parent / "data" / normalized_datasource_path / video_filename
            if candidate.is_file():
                return candidate

        return None

    seq_roots = {
        gt_path.parent.parent
        for gt_path in import_dir.rglob("gt.txt")
        if gt_path.parent.name == "gt"
    }
    for seq_root in sorted(seq_roots):
        key = str(seq_root.relative_to(import_dir))
        video_file = _resolve_video_file(key)
        results[key] = load_mot_from_dir(seq_root, image_width, image_height, video_file)

    for zip_path in sorted(import_dir.rglob("*.zip")):
        key = str(zip_path.relative_to(import_dir))
        video_file = _resolve_video_file(key)
        results[key] = load_mot_from_zip(zip_path, image_width, image_height, video_file)

    return results


def export_to_mot(
    annotations: List[IRVideoBBoxAnnotation],
    context: MOTContext,
    output_path: Union[str, Path],
    video_file: Optional[Union[str, Path]] = None,
) -> Path:
    """Export annotations to MOT gt.txt format, resolving missing dimensions from annotations/video_file."""
    if context.image_width is None or context.image_height is None:
        for ann in annotations:
            if context.image_width is None and ann.video_width is not None and ann.video_width > 0:
                context.image_width = ann.video_width
            if context.image_height is None and ann.video_height is not None and ann.video_height > 0:
                context.image_height = ann.video_height
            if context.image_width is not None and context.image_height is not None:
                break

    if (context.image_width is None or context.image_height is None) and video_file is not None:
        try:
            width, height, fps = get_video_dimensions(Path(video_file))
            if context.image_width is None:
                context.image_width = width
            if context.image_height is None:
                context.image_height = height
            if context.frame_rate == 30.0 and fps > 0:
                context.frame_rate = fps
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not probe video dimensions from {video_file}: {e}")

    if context.seq_length is None and video_file is not None:
        try:
            frame_count = get_video_frame_count(Path(video_file))
            if frame_count is not None and frame_count > 0:
                context.seq_length = frame_count
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not probe video frame count from {video_file}: {e}")

    if context.image_width is None or context.image_height is None:
        raise ValueError(
            "Cannot determine frame dimensions for MOT export. "
            "Provide context.image_width/context.image_height, use annotations with valid dimensions, "
            "or provide video_file for probing."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracks: Dict[int, List[IRVideoBBoxAnnotation]] = defaultdict(list)
    for ann in annotations:
        tracks[ann.track_id].append(ann)

    context_end_frame = context.seq_length - 1 if context.seq_length is not None and context.seq_length > 0 else None
    expanded_annotations: List[IRVideoBBoxAnnotation] = []
    for _, track_annotations in sorted(tracks.items()):
        track_end_candidates: List[int] = []
        if context_end_frame is not None:
            track_end_candidates.append(context_end_frame)
        for ann in track_annotations:
            frames_count = ann.meta.get("ls_frames_count")
            if isinstance(frames_count, int) and frames_count > 0:
                track_end_candidates.append(frames_count - 1)

        track_end_frame = max(track_end_candidates) if track_end_candidates else None
        expanded_annotations.extend(_interpolate_track_for_mot(track_annotations, context, end_frame=track_end_frame))

    sorted_anns = sorted(expanded_annotations, key=lambda a: (a.frame_number, a.track_id))
    if sorted_anns:
        exported_seq_length = max(ann.frame_number for ann in sorted_anns) + 1
        if context.seq_length is None or context.seq_length < exported_seq_length:
            context.seq_length = exported_seq_length
    lines = [export_bbox_to_line(ann, context) for ann in sorted_anns]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    logger.info(f"Exported {len(lines)} MOT annotations to {output_path}")
    return output_path


def export_mot_to_dir(
    annotations: List[IRVideoBBoxAnnotation],
    context: MOTContext,
    output_dir: Union[str, Path],
    video_file: Optional[Union[str, Path]] = None,
    create_seqinfo: bool = False,
) -> Path:
    """
    Export annotations to MOT directory structure.

    Creates::

        output_dir/
          gt/
            gt.txt
            labels.txt
          seqinfo.ini (optional)

    Missing dimensions are resolved with the same fallback as ``export_to_mot``.
    """
    output_dir = Path(output_dir)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    seqinfo_path = output_dir / "seqinfo.ini"

    export_to_mot(annotations, context, gt_dir / "gt.txt", video_file=video_file)

    if create_seqinfo:
        if context.seq_length is None and annotations:
            context.seq_length = max(ann.frame_number for ann in annotations) + 1
        context.write_seqinfo(seqinfo_path)
    elif seqinfo_path.exists():
        seqinfo_path.unlink()
    context.write_labels(gt_dir / "labels.txt")

    logger.info(f"Exported MOT sequence to {output_dir}")
    return output_dir


def export_mot_to_zip(
    annotations: List[IRVideoBBoxAnnotation],
    context: MOTContext,
    output_path: Union[str, Path],
    video_file: Optional[Union[str, Path]] = None,
    create_seqinfo: bool = False,
) -> Path:
    """Export annotations to a MOT zip with gt/gt.txt, gt/labels.txt, and optional seqinfo.ini."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp) / "sequence"
        export_mot_to_dir(
            annotations,
            context,
            tmp_dir,
            video_file=video_file,
            create_seqinfo=create_seqinfo,
        )
        with ZipFile(output_path, "w") as z:
            for file_path in sorted(tmp_dir.rglob("*")):
                if file_path.is_file():
                    z.write(file_path, arcname=str(file_path.relative_to(tmp_dir)))

    logger.info(f"Exported MOT sequence zip to {output_path}")
    return output_path


def export_mot_sequences_to_dirs(
    annotations: List[IRVideoBBoxAnnotation],
    context: MOTContext,
    output_dir: Union[str, Path],
    video_files: Optional[Dict[str, Union[str, Path]]] = None,
    create_seqinfo: bool = False,
) -> Dict[str, Path]:
    """Export multiple MOT sequences to one zip per source filename."""
    def resolve_video_file(sequence_name: str) -> Optional[Union[str, Path]]:
        if video_files is None:
            return None
        if sequence_name in video_files:
            return video_files[sequence_name]
        sequence_stem = Path(sequence_name).stem
        if sequence_stem in video_files:
            return video_files[sequence_stem]
        sequence_basename = Path(sequence_name).name
        if sequence_basename in video_files:
            return video_files[sequence_basename]
        return None

    grouped: Dict[str, List[IRVideoBBoxAnnotation]] = defaultdict(list)
    for ann in annotations:
        if ann.filename:
            sequence_name = Path(ann.filename).name
        else:
            sequence_name = context.seq_name or "sequence"
        grouped[sequence_name].append(ann)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for sequence_name, seq_annotations in sorted(grouped.items()):
        seq_context = context.model_copy(deep=True)
        seq_context.seq_name = sequence_name
        outputs[sequence_name] = export_mot_to_zip(
            seq_annotations,
            seq_context,
            output_dir / f"{sequence_name}.zip",
            video_file=resolve_video_file(sequence_name),
            create_seqinfo=create_seqinfo,
        )
    return outputs
