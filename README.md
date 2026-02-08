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
- [COCO BBox + Segmentation](dagshub_annotation_converter/converters/coco.py)
- [Label Studio](dagshub_annotation_converter/formats/label_studio/task.py#L72) (Only task schema implemented, importing from a project is left up to user):
```python
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask
task_obj = LabelStudioTask.from_json("path/to/label_studio_task.json")

annotations = task_obj.to_ir_annotations()
```
- [CVAT Image](dagshub_annotation_converter/converters/cvat.py#L46)

## Exporters (Image):
- [YOLO BBox, Segmentation, Poses](dagshub_annotation_converter/converters/yolo.py#L126)
- [COCO BBox + Segmentation](dagshub_annotation_converter/converters/coco.py)
- [Label Studio](dagshub_annotation_converter/formats/label_studio/task.py#L225) (Again, only task schema, uploading the task to the project is left to the user)

## Experimental Features:

The package contains some experimental features that can be toggled on by setting a corresponding environment variable:

- `DAGSHUB_ANNOTATION_EXPERIMENTAL_CVAT_POSE_GROUPING_BY_GROUP_ID_ENABLED` - When importing CVAT annotations, 
you can put a single bbox and either a points or skeletons annotation in a group, and then they will be grouped together
into a pose annotation. When exporting that to YOLO, the bbox and the points will be exported as a single annotation.
 
You can read more about the features that exist, their limitations and potential side effects
in the [features.py](dagshub_annotation_converter/features.py) file
