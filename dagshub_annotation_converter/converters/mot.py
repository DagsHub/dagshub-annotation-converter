import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

from dagshub_annotation_converter.formats.mot import MOTContext, import_bbox_from_line, export_bbox_to_line
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation
from dagshub_annotation_converter.util.video import find_video_sibling, get_video_dimensions

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
        context = MOTContext.from_seqinfo(seqinfo_path)
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

        if gt_key not in z.namelist() or not _is_safe_zip_path(gt_key):
            raise FileNotFoundError(f"Could not find gt/gt.txt in {zip_path}")

        if seqinfo_key in z.namelist() and _is_safe_zip_path(seqinfo_key):
            with z.open(seqinfo_key) as f:
                context = MOTContext.from_seqinfo_string(f.read().decode("utf-8"))
        else:
            context = MOTContext()
            logger.warning("seqinfo.ini not found in zip, using default context")

        if image_width is not None:
            context.image_width = image_width
        if image_height is not None:
            context.image_height = image_height

        if labels_key in z.namelist() and _is_safe_zip_path(labels_key):
            with z.open(labels_key) as f:
                context.categories = MOTContext.load_labels_from_string(f.read().decode("utf-8"))

        _try_fill_dimensions_from_video(context, zip_path, video_file)
        _validate_context_dimensions(context, str(zip_path))

        with z.open(gt_key) as f:
            gt_content = f.read().decode("utf-8")
        annotations = _load_mot_from_gt_content(gt_content, context)

    return annotations, context


def export_to_mot(
    annotations: List[IRVideoBBoxAnnotation],
    context: MOTContext,
    output_path: Union[str, Path],
) -> Path:
    """Export annotations to MOT gt.txt format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_anns = sorted(annotations, key=lambda a: (a.frame_number, a.track_id))
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
) -> Path:
    """
    Export annotations to MOT directory structure.

    Creates::

        output_dir/
          gt/
            gt.txt
            labels.txt
          seqinfo.ini
    """
    output_dir = Path(output_dir)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    export_to_mot(annotations, context, gt_dir / "gt.txt")

    if context.seq_length is None and annotations:
        context.seq_length = max(ann.frame_number for ann in annotations) + 1
    context.write_seqinfo(output_dir / "seqinfo.ini")
    context.write_labels(gt_dir / "labels.txt")

    logger.info(f"Exported MOT sequence to {output_dir}")
    return output_dir
