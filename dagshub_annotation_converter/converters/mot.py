import logging
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

from dagshub_annotation_converter.formats.mot import MOTContext, import_bbox_from_line
from dagshub_annotation_converter.formats.mot.bbox import _export_bbox_to_line
from dagshub_annotation_converter.ir.video import (
    IRVideoAnnotationTrack,
    IRVideoBBoxFrameAnnotation,
    IRVideoSequence,
)
from dagshub_annotation_converter.util.video import (
    find_video_sibling,
    probe_video,
)

logger = logging.getLogger(__name__)

_NUMERIC_TRACK_ID_RE = re.compile(r"^(?:track_)?(?P<track_id>\d+)$")


def _mot_track_id_from_identifier(identifier: str) -> int:
    match = _NUMERIC_TRACK_ID_RE.fullmatch(identifier.strip())
    if match is not None:
        return int(match.group("track_id"))
    return int(hashlib.md5(identifier.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


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


def _apply_sequence_filename(sequence: IRVideoSequence, filename: Optional[str]) -> IRVideoSequence:
    sequence.filename = filename
    if filename is None:
        return sequence
    for _, ann in sequence.iter_track_annotations():
        if ann.filename is None:
            ann.filename = filename
    return sequence


def load_mot_from_file(
    gt_path: Union[str, Path],
    context: MOTContext,
) -> IRVideoSequence:
    """Load MOT annotations from a gt.txt file."""
    gt_path = Path(gt_path)
    with open(gt_path, "r") as f:
        sequence = _load_mot_from_gt_content(f.read(), context)
    return _apply_sequence_filename(sequence, context.sequence_name)


def _try_fill_dimensions_from_video(
    context: MOTContext,
    source_path: Path,
    video_file: Optional[Union[str, Path]],
) -> None:
    """Try to fill missing context dimensions from a video file.

    Looks for a video at *video_file* (if given), then falls back to a
    sibling of *source_path* with the same stem and a common video extension.
    When *source_path* is a directory (e.g. the MOT sequence folder ``/data/seq1``),
    the sibling video is expected at the same level as that directory
    (e.g. ``/data/seq1.mp4``), which is the standard MOT dataset layout.
    """
    if (
        context.video_width is not None
        and context.video_height is not None
        and context.frame_rate is not None
        and context.sequence_length is not None
    ):
        return

    candidates: List[Path] = []
    if video_file is not None:
        vf = Path(video_file)
        if vf.is_file():
            candidates.append(vf)
        elif not vf.is_absolute():
            sibling = source_path.parent / vf.name
            if sibling.is_file():
                candidates.append(sibling)

    sibling = find_video_sibling(source_path)
    if sibling is not None:
        candidates.append(sibling)

    for candidate in candidates:
        try:
            probe = probe_video(candidate)
            if context.video_width is None:
                context.video_width = probe.width
            if context.video_height is None:
                context.video_height = probe.height
            if probe.fps > 0:
                if not probe.fps.is_integer():
                    logger.warning(f"Non-integer frame rate {probe.fps} in video {candidate}, using rounded value")
                context.frame_rate = int(round(probe.fps))
            if context.sequence_length is None and probe.frame_count > 0:
                context.sequence_length = probe.frame_count
            logger.info(f"Inferred dimensions {probe.width}x{probe.height} from {candidate}")
            return
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not read video {candidate}: {e}")


def _validate_context_dimensions(context: MOTContext, source: str) -> None:
    if context.video_width is None or context.video_height is None:
        missing = []
        if context.video_width is None:
            missing.append("image_width")
        if context.video_height is None:
            missing.append("image_height")
        raise ValueError(
            f"MOT annotations from {source} require frame dimensions, but "
            f"{', '.join(missing)} could not be determined. "
            f"Provide {', '.join(missing)} explicitly, or place a video file "
            f"with the same name next to the annotation source."
        )


def _interpolate_track_for_mot(
    track: IRVideoAnnotationTrack,
    context: MOTContext,
    end_frame: Optional[int] = None,
) -> List[IRVideoBBoxFrameAnnotation]:
    if not track.annotations:
        return []

    by_frame: Dict[int, IRVideoBBoxFrameAnnotation] = {}
    denormalized_track = track.denormalized(
        video_width=context.video_width,
        video_height=context.video_height,
    )
    for ann in sorted(denormalized_track.annotations, key=lambda a: a.frame_number):
        if isinstance(ann, IRVideoBBoxFrameAnnotation):
            by_frame[ann.frame_number] = ann

    ordered = [by_frame[frame] for frame in sorted(by_frame)]
    dense: List[IRVideoBBoxFrameAnnotation] = []

    for idx, curr in enumerate(ordered):
        curr_visible = curr.is_visible
        dense.append(curr)

        if idx == len(ordered) - 1:
            continue

        nxt = ordered[idx + 1]
        gap = nxt.frame_number - curr.frame_number - 1
        if gap <= 0 or not curr_visible or not curr.interpolation_enabled:
            continue

        for step in range(1, gap + 1):
            t = step / (gap + 1)
            interpolated = curr.interpolate(nxt, t)
            dense.append(interpolated)

    if end_frame is not None:
        last = ordered[-1]
        if last.is_visible and last.interpolation_enabled and last.frame_number < end_frame:
            # MOT expects dense rows through the known sequence end. When the
            # final visible keyframe keeps interpolation enabled, freeze that
            # last box forward rather than trying to extrapolate new motion.
            for frame_number in range(last.frame_number + 1, end_frame + 1):
                frozen = last.model_copy(deep=True)
                frozen.frame_number = frame_number
                frozen.keyframe = False
                dense.append(frozen)

    return dense


def load_mot_from_dir(
    mot_dir: Union[str, Path],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    video_file: Optional[Union[str, Path]] = None,
) -> Tuple[IRVideoSequence, MOTContext]:
    """
    Load MOT annotations from a directory structure.

    Expected structure::

        mot_dir/
          gt/
            gt.txt
            labels.txt
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
        context.video_width = image_width
    if image_height is not None:
        context.video_height = image_height

    labels_path = gt_dir / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Could not find labels.txt in {gt_dir}")
    context.categories = MOTContext.load_labels(labels_path)

    gt_path = gt_dir / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Could not find gt.txt in {gt_dir}")

    _try_fill_dimensions_from_video(context, mot_dir, video_file)
    _validate_context_dimensions(context, str(mot_dir))
    sequence = load_mot_from_file(gt_path, context)
    return _apply_sequence_filename(sequence, context.sequence_name or mot_dir.name), context


def _load_mot_from_gt_content(gt_content: str, context: MOTContext) -> IRVideoSequence:
    tracks: Dict[int, List[IRVideoBBoxFrameAnnotation]] = defaultdict(list)
    for line in gt_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = import_bbox_from_line(line, context)
        if parsed is None:
            continue
        track_id, ann = parsed
        tracks[track_id].append(ann)
    return IRVideoSequence(
        tracks=[
            IRVideoAnnotationTrack.from_annotations(track_annotations, object_id=str(track_id))
            for track_id, track_annotations in tracks.items()
        ],
        sequence_length=context.sequence_length,
        video_width=context.video_width,
        video_height=context.video_height,
    )


def load_mot_from_zip(
    zip_path: Union[str, Path],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    video_file: Optional[Union[str, Path]] = None,
) -> Tuple[IRVideoSequence, MOTContext]:
    """
    Load MOT annotations from a ZIP archive (no extraction, avoids Zip Slip).

    Expected ZIP structure::

        gt/
          gt.txt
          labels.txt
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
            logger.warning("seqinfo.ini not found in zip, using default context")
            context = MOTContext()

        if image_width is not None:
            context.video_width = image_width
        if image_height is not None:
            context.video_height = image_height

        if labels_key not in z.namelist():
            raise FileNotFoundError(f"Could not find {labels_key} in {zip_path}")
        with z.open(labels_key) as f:
            context.categories = MOTContext.load_labels_from_string(f.read().decode("utf-8"))

        _try_fill_dimensions_from_video(context, zip_path, video_file)
        _validate_context_dimensions(context, str(zip_path))

        with z.open(gt_key) as f:
            gt_content = f.read().decode("utf-8")
        sequence = _load_mot_from_gt_content(gt_content, context)

    return _apply_sequence_filename(sequence, context.sequence_name or zip_path.stem), context


def load_mot_from_fs(
    import_dir: Union[str, Path],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    video_dir_name: str = "videos",
    label_dir_name: str = "labels",
) -> Dict[Path, Tuple[IRVideoSequence, MOTContext]]:
    """Load MOT annotations from a dataset root containing video and label directories.

    Expected layout::

        data/
          videos/
            video1.mp4
            nested/video2.mp4
            nested/video3.mp4
          labels/
            video1.mp4.zip
            nested/video2.mp4.zip
            nested/video3/
              gt/gt.txt
              gt/labels.txt

    By default, the function expects ``videos/`` and ``labels/`` under
    *import_dir*, but both directory names are configurable.

    It mirrors label paths back to videos:

    - ``labels/<path>/<video>.ext.zip`` maps to ``videos/<path>/<video>.ext``
    - ``labels/<path>/<video-stem>/`` maps to the video in ``videos/<path>/``
      with the same stem and any supported video extension

    Args:
        import_dir: Dataset root containing the video and label directories.
        image_width: Override frame width for all sequences instead of probing it.
        image_height: Override frame height for all sequences instead of probing it.
        video_dir_name: Name of the directory under *import_dir* that contains videos.
        label_dir_name: Name of the directory under *import_dir* that contains MOT labels.

    Returns a dict keyed by each matched video path relative to
    ``import_dir / video_dir_name``.
    Each imported sequence also gets that same relative video path in
    ``sequence.filename`` and on its frame annotations.
    """
    import_dir = Path(import_dir)
    video_dir = import_dir / video_dir_name
    labels_dir = import_dir / label_dir_name
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Could not find video directory {video_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Could not find label directory {labels_dir}")

    results: Dict[Path, Tuple[IRVideoSequence, MOTContext]] = {}

    def _resolve_video_for_annotation_source(annotation_source: Path) -> Tuple[Path, Path]:
        relative_annotation_path = annotation_source.relative_to(labels_dir)

        if annotation_source.is_file():
            relative_video_path = relative_annotation_path.with_suffix("")
            video_file = video_dir / relative_video_path
            if not video_file.is_file():
                raise FileNotFoundError(
                    f"Could not find video {relative_video_path} for annotation archive {annotation_source}"
                )
            return relative_video_path, video_file

        # Directory annotations intentionally drop the video extension
        # (e.g. ``labels/foo/bar/earth.mp4.zip`` vs ``labels/foo/bar/earth/``).
        # Resolve the real video file by looking for a sibling in ``videos/``
        # with the same stem and any supported video extension.
        video_file = find_video_sibling(video_dir / relative_annotation_path)
        if video_file is None:
            raise FileNotFoundError(
                f"Could not find a video matching annotation directory {annotation_source} under {video_dir}"
            )
        return video_file.relative_to(video_dir), video_file

    def _store_result(relative_video_path: Path, loaded: Tuple[IRVideoSequence, MOTContext]) -> None:
        if relative_video_path in results:
            raise ValueError(f"Found multiple MOT annotation sources for {relative_video_path}")

        sequence, context = loaded
        display_path = relative_video_path.as_posix()
        results[relative_video_path] = (_apply_sequence_filename(sequence, display_path), context)

    seq_roots = {gt_path.parent.parent for gt_path in labels_dir.rglob("gt.txt") if gt_path.parent.name == "gt"}
    for seq_root in sorted(seq_roots):
        relative_video_path, video_file = _resolve_video_for_annotation_source(seq_root)
        _store_result(relative_video_path, load_mot_from_dir(seq_root, image_width, image_height, video_file))

    for zip_path in sorted(labels_dir.rglob("*.zip")):
        relative_video_path, video_file = _resolve_video_for_annotation_source(zip_path)
        _store_result(relative_video_path, load_mot_from_zip(zip_path, image_width, image_height, video_file))

    return results


def export_to_mot(
    sequence: IRVideoSequence,
    context: MOTContext,
    output_path: Union[str, Path],
    video_file: Optional[Union[str, Path]] = None,
) -> Path:
    """Export annotations to MOT gt.txt format, resolving missing dimensions from annotations/video_file."""
    if context.video_width is None:
        context.video_width = sequence.resolved_video_width()
    if context.video_height is None:
        context.video_height = sequence.resolved_video_height()

    if video_file is not None and (
        context.video_width is None
        or context.video_height is None
        or context.sequence_length is None
    ):
        try:
            probe = probe_video(Path(video_file))
            if context.video_width is None:
                context.video_width = probe.width
            if context.video_height is None:
                context.video_height = probe.height
            if probe.fps > 0:
                if not probe.fps.is_integer():
                    logger.warning(f"Non-integer frame rate {probe.fps} in video {video_file}, using rounded value")
                context.frame_rate = int(round(probe.fps))
            if context.sequence_length is None and probe.frame_count > 0:
                context.sequence_length = probe.frame_count
        except (ImportError, ValueError) as e:
            logger.warning(f"Could not probe video from {video_file}: {e}")

    if context.video_width is None or context.video_height is None:
        raise ValueError(
            "Cannot determine frame dimensions for MOT export. "
            "Provide context.video_width/context.video_height, use annotations with valid dimensions, "
            "or provide video_file for probing."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if context.sequence_length is None:
        context.sequence_length = sequence.resolved_sequence_length()

    context_end_frame = (
        context.sequence_length - 1 if context.sequence_length is not None and context.sequence_length > 0 else None
    )
    expanded_annotations: List[Tuple[int, IRVideoBBoxFrameAnnotation]] = []
    for track in sequence.tracks:
        mot_track_id = _mot_track_id_from_identifier(track.object_id)
        for ann in _interpolate_track_for_mot(track, context, end_frame=context_end_frame):
            expanded_annotations.append((mot_track_id, ann))

    sorted_anns = sorted(expanded_annotations, key=lambda item: (item[1].frame_number, item[0]))
    if sorted_anns:
        exported_seq_length = max(ann.frame_number for _, ann in sorted_anns) + 1
        # non-empty sorted_anns implies non-empty tracks, so resolved_sequence_length() is non-None
        assert context.sequence_length is not None
        if context.sequence_length < exported_seq_length:
            context.sequence_length = exported_seq_length
    for _, ann in sorted_anns:
        category_name = ann.ensure_has_one_category()
        context.categories.get_or_create(category_name)
    lines = [_export_bbox_to_line(ann, track_id, context) for track_id, ann in sorted_anns]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")

    logger.info(f"Exported {len(lines)} MOT annotations to {output_path}")
    return output_path


def export_mot_to_dir(
    sequence: IRVideoSequence,
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

    export_to_mot(sequence, context, gt_dir / "gt.txt", video_file=video_file)

    if create_seqinfo:
        if context.sequence_length is None:
            context.sequence_length = sequence.resolved_sequence_length()
        context.write_seqinfo(seqinfo_path)
    context.write_labels(gt_dir / "labels.txt")

    logger.info(f"Exported MOT sequence to {output_dir}")
    return output_dir


def export_mot_to_zip(
    sequence: IRVideoSequence,
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
            sequence,
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
    sequences: Sequence[IRVideoSequence],
    context: MOTContext,
    output_dir: Union[str, Path],
    video_files: Optional[Dict[str, Union[str, Path]]] = None,
    create_seqinfo: bool = False,
    video_dir_name: str = "videos",
    label_dir_name: str = "labels",
) -> Dict[str, Path]:
    """Export multiple MOT sequences to a dataset layout compatible with ``load_mot_from_fs``."""

    def resolve_video_file(relative_video_path: Path) -> Optional[Union[str, Path]]:
        dataset_video = output_dir / video_dir_name / relative_video_path
        if dataset_video.is_file():
            return dataset_video
        if video_files is None:
            return None
        sequence_name = relative_video_path.as_posix()
        if sequence_name in video_files:
            return video_files[sequence_name]
        sequence_stem = Path(sequence_name).stem
        if sequence_stem in video_files:
            return video_files[sequence_stem]
        sequence_basename = Path(sequence_name).name
        if sequence_basename in video_files:
            return video_files[sequence_basename]
        return None

    def resolve_relative_video_path(sequence: IRVideoSequence) -> Path:
        if sequence.filename:
            filename_path = Path(sequence.filename)
            if filename_path.is_absolute():
                return Path(filename_path.name)
            return filename_path
        return Path(context.sequence_name or "sequence")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / video_dir_name).mkdir(parents=True, exist_ok=True)
    labels_root = output_dir / label_dir_name
    labels_root.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for sequence in sequences:
        relative_video_path = resolve_relative_video_path(sequence)
        sequence_name = relative_video_path.name
        seq_context = context.model_copy(deep=True)
        seq_context.sequence_name = sequence_name
        # Mirror the video path under the labels directory so load_mot_from_fs()
        # can map ``labels/<relative-video>.zip`` back to ``videos/<relative-video>``.
        output_path = labels_root / relative_video_path.parent / f"{relative_video_path.name}.zip"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs[relative_video_path.as_posix()] = export_mot_to_zip(
            sequence,
            seq_context,
            output_path,
            video_file=resolve_video_file(relative_video_path),
            create_seqinfo=create_seqinfo,
        )
    return outputs
