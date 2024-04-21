# DagsHub Annotation Converter

This package is intended to be a multi-type importer/exporter/converter
between different annotation formats.

This package is currently in development and has not that many features implemented.
The API is not stable and is subject to change.

Support Matrix for image annotations

| Export > \/ Import V              | YOLO v5+ BBox | YOLO v5+ Segmentation | COCO | DagsHub Datasource (Label Studio) | Label Studio | CVAT Image |
|-----------------------------------|---------------|-----------------------|------|-----------------------------------|--------------|--------------|
| YOLO v5+ BBox                     | -             |                       |      | ✅                                 |              |     |
| YOLO v5+ Segmentation             |               | -                     |      |      ✅                            |              ||
| COCO                              |               |                       | -    |                                   |              ||
| DagsHub Datasource (Label Studio) |               |                       |      | -                                 |              ||
| Label Studio                      |               |                       |      |                                   | -            ||
| CVAT Image                        |               |                       |      |     ✅                            |          |-|

Example usage, importing annotations from [COCO_1K](https://dagshub.com/Dean/COCO_1K) and uploading it into a DagsHub Datasource:

```python
from dagshub_annotation_converter.image.importers import YoloImporter
from dagshub_annotation_converter.image.exporters import DagshubDatasourceExporter

from dagshub.data_engine.datasources import get_datasource

# Assuming that the current worker directory is the root of the repo and images are stored in "data" folder
importer = YoloImporter(
    data_dir="data",                 # Where the images are stored
    annotation_type="segmentation",  # or bbox for bounding boxes
    meta_file="custom_coco.yaml"     # file with the classes
)

proj = importer.parse()

exporter = DagshubDatasourceExporter(
    datasource=get_datasource("<user>/<repo>", "<my datasource>"),
    annotation_field="exported_yolo_annotations"
)
exporter.export(proj)
```
