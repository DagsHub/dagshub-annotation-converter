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

## Importers (Images and Video):
- YOLO [BBox](https://docs.ultralytics.com/datasets/detect/), [Segmentation](https://docs.ultralytics.com/datasets/segment/) and [Poses](https://docs.ultralytics.com/datasets/pose/): [`load_yolo_from_fs`](dagshub_annotation_converter/converters/yolo.py)
- COCO [BBox Object Detection and Stuff Segmentation](https://cocodataset.org/#format-data): [`load_coco_from_file`](dagshub_annotation_converter/converters/coco.py)
- [Label Studio](https://labelstud.io/guide/task_format): [`parse_ls_task`](dagshub_annotation_converter/formats/label_studio/task.py) (Only reading JSON tasks from the filesystem is implemented, importing from a project is left up to user):
```python
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask
task_obj = LabelStudioTask.from_json("path/to/label_studio_task.json")

annotations = task_obj.to_ir_annotations()
```
- [CVAT for image (BBox, Segmentation, Poses) and video (BBox)](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/): [`load_cvat_from_zip`](dagshub_annotation_converter/converters/cvat.py)
- [MOT (Video BBox)](https://motchallenge.net) - [`load_mot_from_dir`](dagshub_annotation_converter/converters/mot.py)
- [CVAT for video (BBox)](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/): [`load_cvat_from_zip`](dagshub_annotation_converter/converters/cvat.py)

## Exporters (Images and Video):
- YOLO [BBox](https://docs.ultralytics.com/datasets/detect/), [Segmentation](https://docs.ultralytics.com/datasets/segment/) and [Poses](https://docs.ultralytics.com/datasets/pose/): [`export_to_fs`](dagshub_annotation_converter/converters/yolo.py)
- COCO [BBox Object Detection and Stuff Segmentation](https://cocodataset.org/#format-data): [`export_to_coco_file`](dagshub_annotation_converter/converters/coco.py)
- [Label Studio](https://labelstud.io/guide/task_format): [`task.model_dump_json()`](dagshub_annotation_converter/formats/label_studio/task.py) (Again, only to JSON files, uploading the task to the project is left to the user)
- [MOT (Video BBox)](https://motchallenge.net) - [`export_mot_to_dir`](dagshub_annotation_converter/converters/mot.py)
- [CVAT for video (BBox)](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/): [`export_cvat_video_to_zip`](dagshub_annotation_converter/converters/cvat.py)

## Experimental Features:

The package contains some experimental features that can be toggled on by setting a corresponding environment variable:

- `DAGSHUB_ANNOTATION_EXPERIMENTAL_CVAT_POSE_GROUPING_BY_GROUP_ID_ENABLED` - When importing CVAT annotations, 
you can put a single bbox and either a points or skeletons annotation in a group, and then they will be grouped together
into a pose annotation. When exporting that to YOLO, the bbox and the points will be exported as a single annotation.
 
You can read more about the features that exist, their limitations and potential side effects
in the [features.py](dagshub_annotation_converter/features.py) file
