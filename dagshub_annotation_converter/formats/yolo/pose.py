from typing import Union, Tuple, Optional, Sequence

from dagshub_annotation_converter.formats.common import (
    ImageType,
    determine_image_dimensions,
)
from dagshub_annotation_converter.formats.yolo.categories import determine_category
from dagshub_annotation_converter.formats.yolo.context import YoloContext
from dagshub_annotation_converter.ir.image import NormalizationState
from dagshub_annotation_converter.ir.image.annotations.pose import IRPoseAnnotation, IRPosePoint


def import_pose_2dim(
    category: Union[int, str],
    points: Sequence[Tuple[float, float]],
    middle_x: float,
    middle_y: float,
    width: float,
    height: float,
    context: YoloContext,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    image: Optional[ImageType] = None,
) -> IRPoseAnnotation:
    points_3dim = [(x, y, None) for x, y in points]
    return import_pose_3dim(
        category=category,
        points=points_3dim,
        middle_x=middle_x,
        middle_y=middle_y,
        width=width,
        height=height,
        context=context,
        image_width=image_width,
        image_height=image_height,
        image=image,
    )


def import_pose_3dim(
    category: Union[int, str],
    points: Sequence[Tuple[float, float, Optional[bool]]],
    middle_x: float,
    middle_y: float,
    width: float,
    height: float,
    context: YoloContext,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    image: Optional[ImageType] = None,
) -> IRPoseAnnotation:
    image_width, image_height = determine_image_dimensions(
        image_width=image_width, image_height=image_height, image=image
    )
    parsed_category = determine_category(category, context.categories)

    return IRPoseAnnotation(
        category=parsed_category.name,
        image_width=image_width,
        image_height=image_height,
        state=NormalizationState.NORMALIZED,
        top=middle_y - height / 2,
        left=middle_x - width / 2,
        height=height,
        width=width,
        points=[IRPosePoint(x=x, y=y, visible=visible) for x, y, visible in points],
    )


def import_pose_from_string(
    annotation: str,
    context: YoloContext,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    image: Optional[ImageType] = None,
) -> IRPoseAnnotation:
    if len(annotation.split("\n")) > 1:
        raise ValueError("Please pass one annotation at a time")
    parts = annotation.strip().split(" ")
    category = int(parts[0])
    middle_x = float(parts[1])
    middle_y = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    points: Sequence[Tuple[float, float, Optional[bool]]]
    if context.keypoint_dim == 2:
        points = [(float(parts[i]), float(parts[i + 1]), None) for i in range(5, len(parts), 2)]
    elif context.keypoint_dim == 3:
        points = [(float(parts[i]), float(parts[i + 1]), parts[i + 2] == "1") for i in range(5, len(parts), 3)]
    else:
        raise ValueError("Unsupported keypoint dimension {context.keypoint_dim}")

    return import_pose_3dim(
        context=context,
        category=category,
        points=points,
        middle_x=middle_x,
        middle_y=middle_y,
        width=width,
        height=height,
        image_width=image_width,
        image_height=image_height,
        image=image,
    )


def export_pose(annotation: IRPoseAnnotation, context: YoloContext) -> str:
    if context.keypoint_dim == 2:
        point_list = [f"{point.x} {point.y}" for point in annotation.points if point.visible != False]
    else:
        point_list = [f"{point.x} {point.y} {0 if point.visible == False else 1}" for point in annotation.points]

    cat_id = context.categories[annotation.category].id

    return " ".join(
        [
            str(cat_id),
            str(annotation.left + annotation.width / 2),
            str(annotation.top + annotation.height / 2),
            str(annotation.width),
            str(annotation.height),
            *point_list,
        ]
    )
