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
    export_segmentation_group,
)
from dagshub_annotation_converter.ir.image import (
    IRImageAnnotationBase,
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
)

logger = logging.getLogger(__name__)


def _consume_annotation_id(
    imported_id: Optional[str],
    used_ids: set,
    next_annotation_id: int,
) -> Tuple[int, int]:
    if imported_id is not None:
        try:
            parsed_id = int(imported_id)
            if parsed_id > 0 and parsed_id not in used_ids:
                used_ids.add(parsed_id)
                return parsed_id, max(next_annotation_id, parsed_id + 1)
        except ValueError:
            pass

    while next_annotation_id in used_ids:
        next_annotation_id += 1
    assigned_id = next_annotation_id
    used_ids.add(assigned_id)
    return assigned_id, next_annotation_id + 1


def _load_coco_dict(coco: Dict[str, Any]) -> Tuple[Dict[str, List[IRImageAnnotationBase]], CocoContext]:
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
            logger.warning(
                "Skipping COCO annotation id=%s with unknown image_id=%s",
                raw_annotation.get("id"), image_id)
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


def load_coco_from_file(path: Union[str, Path]) -> Tuple[Dict[str, List[IRImageAnnotationBase]], CocoContext]:
    with open(path, "r") as f:
        coco = json.load(f)
    return _load_coco_dict(coco)


def load_coco_from_json_string(json_str: str) -> Tuple[Dict[str, List[IRImageAnnotationBase]], CocoContext]:
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
    used_annotation_ids = set()

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

        segmentation_groups: List[List[IRSegmentationImageAnnotation]] = []
        segmentation_group_lookup: Dict[Tuple[str, str], int] = {}

        for ann in anns:
            if isinstance(ann, IRBBoxImageAnnotation):
                ann_id, annotation_id = _consume_annotation_id(ann.imported_id, used_annotation_ids, annotation_id)
                coco_annotations.append(export_bbox(ann, export_context, image_id, ann_id))
                continue

            if isinstance(ann, IRSegmentationImageAnnotation):
                if ann.imported_id is None:
                    segmentation_groups.append([ann])
                    continue

                category_name = ann.ensure_has_one_category()
                key = (ann.imported_id, category_name)
                if key not in segmentation_group_lookup:
                    segmentation_group_lookup[key] = len(segmentation_groups)
                    segmentation_groups.append([ann])
                else:
                    segmentation_groups[segmentation_group_lookup[key]].append(ann)
                continue

            logger.warning(
                "Skipping unsupported annotation type for COCO export: %s (file=%s)",
                type(ann).__name__,
                filename,
            )

        for group in segmentation_groups:
            group_imported_id = group[0].imported_id
            ann_id, annotation_id = _consume_annotation_id(group_imported_id, used_annotation_ids, annotation_id)
            coco_annotations.append(export_segmentation_group(group, export_context, image_id, ann_id))

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
