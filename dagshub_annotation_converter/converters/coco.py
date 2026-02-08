import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from dagshub_annotation_converter.converters.common import group_annotations_by_filename
from dagshub_annotation_converter.formats.coco import (
    CocoContext,
    import_bbox,
    import_segmentation,
    export_bbox,
    export_segmentation,
)
from dagshub_annotation_converter.ir.image import (
    IRImageAnnotationBase,
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
)

logger = logging.getLogger(__name__)


def _load_coco_dict(coco: Dict[str, Any]) -> Tuple[Dict[str, Sequence[IRImageAnnotationBase]], CocoContext]:
    context = CocoContext()
    for category in coco.get("categories", []):
        context.categories[int(category["id"])] = str(category["name"])

    image_lookup: Dict[int, Dict[str, Any]] = {}
    for image in coco.get("images", []):
        image_lookup[int(image["id"])] = image

    grouped: Dict[str, List[IRImageAnnotationBase]] = {}
    for raw_annotation in coco.get("annotations", []):
        image_id = int(raw_annotation["image_id"])
        image = image_lookup.get(image_id)
        if image is None:
            logger.warning("Skipping COCO annotation id=%s with unknown image_id=%s", raw_annotation.get("id"), image_id)
            continue

        filename = str(image["file_name"])
        if filename not in grouped:
            grouped[filename] = []

        if "bbox" in raw_annotation and raw_annotation["bbox"] is not None:
            grouped[filename].append(import_bbox(raw_annotation, image, context).with_filename(filename))

        if "segmentation" in raw_annotation and raw_annotation["segmentation"] is not None:
            segmentation_annotations = import_segmentation(raw_annotation, image, context)
            grouped[filename].extend([ann.with_filename(filename) for ann in segmentation_annotations])

    return grouped, context


def load_coco_from_file(path: Union[str, Path]) -> Tuple[Dict[str, Sequence[IRImageAnnotationBase]], CocoContext]:
    with open(path, "r") as f:
        coco = json.load(f)
    return _load_coco_dict(coco)


def load_coco_from_json_string(json_str: str) -> Tuple[Dict[str, Sequence[IRImageAnnotationBase]], CocoContext]:
    return _load_coco_dict(json.loads(json_str))


def _build_coco_dict(
    annotations: Sequence[IRImageAnnotationBase],
    context: Optional[CocoContext] = None,
) -> Dict[str, Any]:
    export_context = context.model_copy(deep=True) if context is not None else CocoContext()
    grouped = group_annotations_by_filename(annotations)

    images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    annotation_id = 1

    for image_id, (filename, anns) in enumerate(grouped.items(), start=1):
        first = anns[0]
        images.append(
            {
                "id": image_id,
                "width": first.image_width,
                "height": first.image_height,
                "file_name": filename,
            }
        )

        for ann in anns:
            if isinstance(ann, IRBBoxImageAnnotation):
                coco_annotations.append(export_bbox(ann, export_context, image_id, annotation_id))
                annotation_id += 1
            elif isinstance(ann, IRSegmentationImageAnnotation):
                coco_annotations.append(export_segmentation(ann, export_context, image_id, annotation_id))
                annotation_id += 1
            else:
                logger.warning(
                    "Skipping unsupported annotation type for COCO export: %s (file=%s)",
                    type(ann).__name__,
                    filename,
                )

    categories = [{"id": category_id, "name": name} for category_id, name in sorted(export_context.categories.items())]

    return {
        "categories": categories,
        "images": images,
        "annotations": coco_annotations,
    }


def export_to_coco_file(
    annotations: Sequence[IRImageAnnotationBase],
    output_path: Union[str, Path],
    context: Optional[CocoContext] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coco = _build_coco_dict(annotations, context=context)

    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return output_path

