# DagsHub Annotation Converter

This package is intended to be a multi-type importer/exporter/converter
between different annotation formats.

This package is currently in development and has not that many features implemented.
The API is not stable and is subject to change.

The package consists of the Intermediary Representation (IR) annotation format in Python Objects,
and importers/exporters for different annotation formats.

## Installation

```bash
pip install dagshub-annotation-converter
```

## Importers (Image):
- [YOLO BBox, Segmentation, Poses](dagshub_annotation_converter/converters/yolo.py#L81)
- Label Studio
- CVAT Image

## Exporters (Image):
- YOLO BBox, Segmentation, Poses
- Label Studio
