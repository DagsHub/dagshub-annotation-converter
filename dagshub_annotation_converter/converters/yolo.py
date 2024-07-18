import logging
import os
from pathlib import Path
from typing import Union, Sequence, List, Optional, Dict, Tuple

import PIL.Image

from dagshub_annotation_converter.converters.common import group_annotations_by_filename
from dagshub_annotation_converter.formats.yolo import (
    export_lookup,
    allowed_annotation_types,
    YoloContext,
    import_lookup,
    YoloAnnotationTypes,
)
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase
from dagshub_annotation_converter.util import is_image, replace_folder

logger = logging.getLogger(__name__)


def load_yolo_from_fs_with_context(
    context: YoloContext,
    import_dir: Union[str, Path] = ".",
) -> Dict[str, Sequence[IRImageAnnotationBase]]:
    assert context.path is not None

    annotations: Dict[str, Sequence[IRImageAnnotationBase]] = {}

    import_dir_path = Path(import_dir)

    if context.path.is_absolute():
        data_dir_path = context.path
    else:
        data_dir_path = import_dir_path / context.path

    for dirpath, subdirs, files in os.walk(data_dir_path):
        if context.image_dir_name not in dirpath.split("/"):
            logger.debug(f"{dirpath} is not an image dir, skipping")
            continue
        for filename in files:
            fullpath = os.path.join(dirpath, filename)
            img = Path(fullpath)
            relpath = img.relative_to(data_dir_path)
            if not is_image(img):
                logger.debug(f"Skipping {img} because it's not an image")
                continue
            annotation = replace_folder(img, context.image_dir_name, context.label_dir_name, context.label_extension)
            if annotation is None:
                logger.warning(f"Couldn't generate annotation file path for image file [{img}]")
                continue
            if not annotation.exists():
                logger.warning(f"Couldn't find annotation file [{annotation}] for image file [{img}]")
                continue
            annotations[str(relpath)] = parse_annotation(context, data_dir_path, img, annotation)

    return annotations


def parse_annotation(
    context: YoloContext, base_path: Path, img_path: Path, annotation_path: Path
) -> Sequence[IRImageAnnotationBase]:
    img = PIL.Image.open(img_path)
    img_width, img_height = img.size

    annotation_strings = annotation_path.read_text().strip().split("\n")

    assert context.annotation_type is not None

    convert_func = import_lookup[context.annotation_type]

    res: List[IRImageAnnotationBase] = []
    rel_path = str(img_path.relative_to(base_path))

    for ann in annotation_strings:
        res.append(convert_func(ann, context, img_width, img_height, img).with_filename(rel_path))

    return res


def load_yolo_from_fs(
    annotation_type: YoloAnnotationTypes,
    meta_file: Union[str, Path] = "annotations.yaml",
    image_dir_name: str = "images",
    label_dir_name: str = "labels",
) -> Tuple[Dict[str, Sequence[IRImageAnnotationBase]], YoloContext]:
    meta_file_path = Path(meta_file).absolute()
    context = YoloContext.from_yaml_file(meta_file, annotation_type=annotation_type)
    context.image_dir_name = image_dir_name
    context.label_dir_name = label_dir_name
    context.annotation_type = annotation_type

    return load_yolo_from_fs_with_context(context, import_dir=meta_file_path.parent), context


# ======== Annotation Export ======== #


def annotations_to_string(annotations: Sequence[IRImageAnnotationBase], context: YoloContext) -> Optional[str]:
    """
    Serializes multiple YOLO annotations into the contents of the annotations file.
    Also makes sure that only annotations of the correct type for context.annotation_type are serialized.

    :param annotations: Annotations to serialize (should be single file)
    :param context: Exporting context
    :return: String of the content of the file
    """
    filtered_annotations = [
        ann for ann in annotations if isinstance(ann, allowed_annotation_types[context.annotation_type])
    ]

    if len(filtered_annotations) != len(annotations):
        logger.warning(
            f"{annotations[0].filename} has {len(annotations) - len(filtered_annotations)} "
            f"annotations of the wrong type that won't be exported"
        )

    if len(filtered_annotations) == 0:
        return None

    export_fn = export_lookup[context.annotation_type]

    return "\n".join([export_fn(ann, context) for ann in filtered_annotations])


def export_to_fs(
    context: YoloContext,
    annotations: List[IRImageAnnotationBase],
    export_dir: Union[str, Path] = ".",
    meta_file="yolo_dagshub.yaml",
) -> Path:
    """
    Exports annotations to YOLO format.

    This function exports them in a way that allows you to train with YOLO right away,
    as long as the images have already been copied to the data folder.

    :param context: Context for exporting. Set the ``path`` attribute to specify the directory with the data,
        otherwise exports a ``data`` folder in the current working directory.
    :param annotations: Annotations to export
    :param export_dir: Directory to export to. If not specified, exports to the current working directory.
    :param meta_file: Name of the YAML file of the YOLO dataset definition.
        This file will be written to the parent directory of the data path.

    :return: Path to the YAML file with the exported data
    """
    if context.path is None:
        print(f"`YoloContext.path` was not set. Exporting to {os.path.join(os.getcwd(), 'data')}")
        context.path = Path("data")

    grouped_annotations = group_annotations_by_filename(annotations)

    export_path = Path(export_dir)

    for filename, anns in grouped_annotations.items():
        annotation_filepath = replace_folder(
            Path(filename), context.image_dir_name, context.label_dir_name, context.label_extension
        )
        if annotation_filepath is None:
            logger.warning(f"Couldn't generate annotation file path for image file [{filename}]")
            continue
        annotation_filename = export_path / context.path / annotation_filepath
        annotation_filename.parent.mkdir(parents=True, exist_ok=True)
        annotation_content = annotations_to_string(anns, context)
        if annotation_content is not None:
            with open(annotation_filename, "w") as f:
                f.write(annotation_content)

    # TODO: test/val splitting
    yaml_file_path = export_path / meta_file
    with open(yaml_file_path, "w") as yaml_f:
        yaml_f.write(context.get_yaml_content())

    logger.warning(f"Saved annotations to {context.path}\nand .YAML file at {yaml_file_path}")

    return yaml_file_path.absolute()
