"""
MOT format converter.

Supports CVAT MOT 1.1 format for video object tracking.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

from dagshub_annotation_converter.formats.mot import MOTContext, import_bbox_from_line, export_bbox_to_line
from dagshub_annotation_converter.ir.video import IRVideoBBoxAnnotation

logger = logging.getLogger(__name__)


def _is_safe_zip_path(name: str) -> bool:
    """Reject path traversal (Zip Slip)."""
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def _find_mot_prefix_in_zip(z: ZipFile) -> str:
    """Find the MOT root prefix (e.g. '' or 'seqname/') from zip entries."""
    names = [n for n in z.namelist() if _is_safe_zip_path(n)]
    # Prefer gt/gt.txt at root
    if "gt/gt.txt" in names:
        return ""
    # Look for X/gt/gt.txt
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
    """
    Load MOT annotations from a gt.txt file.
    
    Args:
        gt_path: Path to gt.txt file
        context: MOT context with categories and image dimensions
        
    Returns:
        Dict mapping frame_number to list of annotations for that frame
    """
    gt_path = Path(gt_path)
    
    annotations: Dict[int, List[IRVideoBBoxAnnotation]] = {}
    
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            ann = import_bbox_from_line(line, context)
            
            if ann.frame_number not in annotations:
                annotations[ann.frame_number] = []
            annotations[ann.frame_number].append(ann)
    
    return annotations


def load_mot_from_dir(
    mot_dir: Union[str, Path],
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Tuple[Dict[int, Sequence[IRVideoBBoxAnnotation]], MOTContext]:
    """
    Load MOT annotations from a directory structure.
    
    Expected structure:
    mot_dir/
      gt/
        gt.txt
        labels.txt (optional)
      seqinfo.ini (optional)
    
    Args:
        mot_dir: Path to MOT sequence directory
        image_width: Frame width (overrides seqinfo.ini if provided)
        image_height: Frame height (overrides seqinfo.ini if provided)
        
    Returns:
        Tuple of (annotations by frame, MOT context)
    """
    mot_dir = Path(mot_dir)
    gt_dir = mot_dir / "gt"
    
    # Load context from seqinfo.ini (at root level)
    seqinfo_path = mot_dir / "seqinfo.ini"
    if seqinfo_path.exists():
        context = MOTContext.from_seqinfo(seqinfo_path)
    else:
        context = MOTContext()
        logger.warning(f"seqinfo.ini not found at {seqinfo_path}, using default context")
    
    # Override dimensions if provided
    if image_width is not None:
        context.image_width = image_width
    if image_height is not None:
        context.image_height = image_height
    
    # Load labels if available (inside gt/ folder)
    labels_path = gt_dir / "labels.txt"
    if labels_path.exists():
        context.categories = MOTContext.load_labels(labels_path)
    
    # Find gt.txt file
    gt_path = gt_dir / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Could not find gt.txt in {gt_dir}")
    
    annotations = load_mot_from_file(gt_path, context)
    
    return annotations, context


def _load_mot_from_gt_content(gt_content: str, context: MOTContext) -> Dict[int, Sequence[IRVideoBBoxAnnotation]]:
    """Parse gt.txt content into annotations by frame."""
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
) -> Tuple[Dict[int, Sequence[IRVideoBBoxAnnotation]], MOTContext]:
    """
    Load MOT annotations from a ZIP archive.
    
    Reads files directly from the zip (no extraction) to avoid Zip Slip.
    
    Expected ZIP structure:
      gt/
        gt.txt
        labels.txt (optional)
      seqinfo.ini (optional)
    
    Or nested: seqname/gt/gt.txt, seqname/gt/labels.txt, seqname/seqinfo.ini
    
    Args:
        zip_path: Path to ZIP archive containing MOT data
        image_width: Frame width (overrides seqinfo.ini if provided)
        image_height: Frame height (overrides seqinfo.ini if provided)
        
    Returns:
        Tuple of (annotations by frame, MOT context)
    """
    zip_path = Path(zip_path)
    
    with ZipFile(zip_path) as z:
        prefix = _find_mot_prefix_in_zip(z)
        gt_key = f"{prefix}gt/gt.txt"
        labels_key = f"{prefix}gt/labels.txt"
        seqinfo_key = f"{prefix}seqinfo.ini"
        
        if gt_key not in z.namelist() or not _is_safe_zip_path(gt_key):
            raise FileNotFoundError(f"Could not find gt/gt.txt in {zip_path}")
        
        # Build context
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
        
        # Load gt.txt
        with z.open(gt_key) as f:
            gt_content = f.read().decode("utf-8")
        annotations = _load_mot_from_gt_content(gt_content, context)
    
    return annotations, context


def export_to_mot(
    annotations: List[IRVideoBBoxAnnotation],
    context: MOTContext,
    output_path: Union[str, Path],
) -> Path:
    """
    Export annotations to MOT format.
    
    Args:
        annotations: List of video bbox annotations
        context: MOT context with categories
        output_path: Path to output gt.txt file
        
    Returns:
        Path to the written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort annotations by frame_number, then by track_id
    sorted_anns = sorted(annotations, key=lambda a: (a.frame_number, a.track_id))
    
    lines = []
    for ann in sorted_anns:
        line = export_bbox_to_line(ann, context)
        lines.append(line)
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")  # Trailing newline
    
    logger.info(f"Exported {len(lines)} MOT annotations to {output_path}")
    
    return output_path


def export_mot_to_dir(
    annotations: List[IRVideoBBoxAnnotation],
    context: MOTContext,
    output_dir: Union[str, Path],
) -> Path:
    """
    Export annotations to MOT directory structure.
    
    Creates:
    output_dir/
      gt/
        gt.txt
        labels.txt
      seqinfo.ini
    
    Args:
        annotations: List of video bbox annotations
        context: MOT context
        output_dir: Output directory path
        
    Returns:
        Path to the output directory
    """
    output_dir = Path(output_dir)
    gt_dir = output_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Export annotations
    gt_path = gt_dir / "gt.txt"
    export_to_mot(annotations, context, gt_path)
    
    # Write seqinfo.ini (at root level)
    seqinfo_path = output_dir / "seqinfo.ini"
    
    # Infer seq_length from annotations if not set (IR frames are 0-based)
    if context.seq_length is None and annotations:
        context.seq_length = max(ann.frame_number for ann in annotations) + 1
    
    context.write_seqinfo(seqinfo_path)
    
    # Write labels.txt (inside gt/ folder)
    labels_path = gt_dir / "labels.txt"
    context.write_labels(labels_path)
    
    logger.info(f"Exported MOT sequence to {output_dir}")
    
    return output_dir
