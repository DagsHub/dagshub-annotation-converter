import json
from pathlib import Path

from dagshub_annotation_converter.converters.coco import export_to_coco_file, load_coco_from_file
from dagshub_annotation_converter.ir.image import (
    CoordinateStyle,
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
    IRSegmentationPoint,
)


def test_coco_export(tmp_path):
    annotations = [
        IRBBoxImageAnnotation(
            filename="images/a.jpg",
            categories={"cat": 1.0},
            top=20.0,
            left=10.0,
            width=30.0,
            height=40.0,
            image_width=100,
            image_height=80,
            coordinate_style=CoordinateStyle.DENORMALIZED,
        ),
        IRSegmentationImageAnnotation(
            filename="images/a.jpg",
            categories={"dog": 1.0},
            image_width=100,
            image_height=80,
            coordinate_style=CoordinateStyle.DENORMALIZED,
            points=[
                IRSegmentationPoint(x=5.0, y=10.0),
                IRSegmentationPoint(x=15.0, y=10.0),
                IRSegmentationPoint(x=15.0, y=30.0),
                IRSegmentationPoint(x=5.0, y=30.0),
            ],
        ),
    ]

    output_file = tmp_path / "annotations.json"
    exported = export_to_coco_file(annotations, output_file)

    assert exported == output_file
    assert output_file.exists()

    content = json.loads(output_file.read_text())
    assert len(content["images"]) == 1
    assert len(content["annotations"]) == 2
    assert {c["name"] for c in content["categories"]} == {"cat", "dog"}


def test_coco_import_export_roundtrip(tmp_path):
    source = tmp_path / "source.json"
    source.write_text((Path(__file__).parent / "res" / "annotations.json").read_text())
    imported, context = load_coco_from_file(source)

    flattened = []
    for per_file in imported.values():
        flattened.extend(per_file)

    output = tmp_path / "roundtrip.json"
    export_to_coco_file(flattened, output, context=context)
    reimported, _ = load_coco_from_file(output)

    output_payload = json.loads(output.read_text())
    image_to_name = {image["id"]: image["file_name"] for image in output_payload["images"]}
    multi_polygon_anns = [
        ann
        for ann in output_payload["annotations"]
        if isinstance(ann.get("segmentation"), list)
        and len(ann["segmentation"]) == 2
        and image_to_name.get(ann["image_id"]) == "images/b.jpg"
    ]
    assert len(multi_polygon_anns) == 1

    def strip_imported_ids(annotations_map):
        stripped = {}
        for filename, anns in annotations_map.items():
            stripped[filename] = [ann.model_copy(update={"imported_id": None}) for ann in anns]
        return stripped

    assert strip_imported_ids(reimported) == strip_imported_ids(imported)
