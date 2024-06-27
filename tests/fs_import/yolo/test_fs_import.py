from pathlib import Path

import pytest

from dagshub_annotation_converter.converters.yolo import load_yolo_from_fs
from dagshub_annotation_converter.ir.image import (
    IRBBoxAnnotation,
    NormalizationState,
    IRSegmentationAnnotation,
    IRSegmentationPoint,
    IRPoseAnnotation,
    IRPosePoint,
)


@pytest.fixture
def data_folder() -> Path:
    return Path(__file__).parent / "res"


@pytest.fixture
def img_path() -> str:
    return "images/testimg.png"


def test_bbox(data_folder, img_path):
    yaml = data_folder / "bbox_and_segmentation.yaml"

    annotations, ctx = load_yolo_from_fs("bbox", yaml, label_dir_name="labels_bbox")

    assert ctx.annotation_type == "bbox"

    expected = {
        img_path: [
            IRBBoxAnnotation(
                filename=img_path,
                category=ctx.categories[0],
                image_width=640,
                image_height=480,
                state=NormalizationState.NORMALIZED,
                top=0.0,
                left=0.0,
                width=0.5,
                height=0.5,
            ),
            IRBBoxAnnotation(
                filename=img_path,
                category=ctx.categories[1],
                image_width=640,
                image_height=480,
                state=NormalizationState.NORMALIZED,
                top=0.5,
                left=0.5,
                width=0.5,
                height=0.5,
            ),
        ]
    }

    assert annotations == expected


def test_segmentation(data_folder, img_path):
    yaml = data_folder / "bbox_and_segmentation.yaml"

    annotations, ctx = load_yolo_from_fs("segmentation", yaml, label_dir_name="labels_segmentation")

    assert ctx.annotation_type == "segmentation"

    raw_points = [
        [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
        [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)],
        [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
    ]

    points = [[IRSegmentationPoint(x=p[0], y=p[1]) for p in pts] for pts in raw_points]

    expected = {
        img_path: [
            IRSegmentationAnnotation(
                filename=img_path,
                category=ctx.categories[0],
                image_width=640,
                image_height=480,
                state=NormalizationState.NORMALIZED,
                points=points[0],
            ),
            IRSegmentationAnnotation(
                filename=img_path,
                category=ctx.categories[1],
                image_width=640,
                image_height=480,
                state=NormalizationState.NORMALIZED,
                points=points[1],
            ),
            IRSegmentationAnnotation(
                filename=img_path,
                category=ctx.categories[0],
                image_width=640,
                image_height=480,
                state=NormalizationState.NORMALIZED,
                points=points[2],
            ),
        ]
    }

    assert annotations == expected


def generate_expected(img_path, ctx, to_keypoints_fn) -> dict:
    points = [
        (
            (0.0, 0.0, 0.5, 0.5),
            to_keypoints_fn([(0.1, 0.1, True), (0.2, 0.2, False), (0.3, 0.3, True), (0.4, 0.4, False)]),
        ),
        (
            (0.5, 0.5, 0.5, 0.5),
            to_keypoints_fn([(0.6, 0.6, True), (0.7, 0.7, False), (0.8, 0.8, True), (0.9, 0.9, False)]),
        ),
    ]

    return {
        img_path: [
            IRPoseAnnotation(
                filename=img_path,
                category=ctx.categories[0],
                image_width=640,
                image_height=480,
                state=NormalizationState.NORMALIZED,
                left=points[0][0][0],
                top=points[0][0][1],
                width=points[0][0][2],
                height=points[0][0][3],
                points=points[0][1],
            ),
            IRPoseAnnotation(
                filename=img_path,
                category=ctx.categories[1],
                image_width=640,
                image_height=480,
                state=NormalizationState.NORMALIZED,
                left=points[1][0][0],
                top=points[1][0][1],
                width=points[1][0][2],
                height=points[1][0][3],
                points=points[1][1],
            ),
        ]
    }


def test_pose_2dim(data_folder, img_path):
    yaml = data_folder / "pose_2dim.yaml"

    annotations, ctx = load_yolo_from_fs("pose", yaml, label_dir_name="labels_pose_2dim")

    assert ctx.annotation_type == "pose"
    assert ctx.keypoint_dim == 2
    assert ctx.keypoints_in_annotation == 4

    def to_keypoints(pts):
        return [IRPosePoint(x=x, y=y, visible=None) for x, y, visible in pts]

    expected = generate_expected(img_path, ctx, to_keypoints)

    assert annotations == expected


def test_pose_3dim(data_folder, img_path):
    yaml = data_folder / "pose_3dim.yaml"

    annotations, ctx = load_yolo_from_fs("pose", yaml, label_dir_name="labels_pose_3dim")

    assert ctx.annotation_type == "pose"
    assert ctx.keypoint_dim == 3
    assert ctx.keypoints_in_annotation == 4

    def to_keypoints(pts):
        return [IRPosePoint(x=x, y=y, visible=visible) for x, y, visible in pts]

    expected = generate_expected(img_path, ctx, to_keypoints)

    assert annotations == expected
