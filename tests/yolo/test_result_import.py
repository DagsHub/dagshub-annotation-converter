import pytest
import numpy as np
import torch
from ultralytics.engine.results import Results

from dagshub_annotation_converter.formats.yolo import import_yolo_result
from dagshub_annotation_converter.ir.image import (
    IRBBoxImageAnnotation,
    CoordinateStyle,
    IRSegmentationPoint,
    IRSegmentationImageAnnotation,
    IRPosePoint,
    IRPoseImageAnnotation,
)


@pytest.fixture
def boxes() -> torch.Tensor:
    return torch.tensor(
        [
            # Layout: [x_tl, y_tl, x_br, y_br, conf, class_id]
            [10, 10, 20, 30, 0.9, 0],
            [30, 30, 50, 40, 0.8, 1],
        ],
        dtype=torch.float64,  # Setting it high to prevent loss of precision on confidence
    )


@pytest.fixture
def masks() -> torch.Tensor:
    # Layout: mask on the image of the segment, 0 is not this class, 1 is this class
    res = torch.zeros(
        (2, 480, 640),
        dtype=torch.int8,
    )
    # Fill out the cat (being lazy and just drawing a rectangle)
    for i in range(10, 21):
        for j in range(10, 31):
            res[0, j, i] = 1

    # Fill out the dog
    for i in range(30, 51):
        for j in range(30, 41):
            res[1, j, i] = 1

    return res


@pytest.fixture
def poses() -> torch.Tensor:
    # Layout: [x, y, confidence] for each keypoint
    return torch.tensor(
        [
            [[10, 10, 0.9], [20, 20, 1.0], [20, 30, 1.0]],
            [[30, 30, 0.9], [30, 35, 0.9], [40, 40, 1.0]],
        ],
        dtype=torch.float64,
    )


@pytest.fixture
def yolo_result(boxes, masks, poses) -> Results:
    img = np.ndarray(shape=(480, 640, 3), dtype=np.uint8)
    names = {0: "cat", 1: "dog"}

    res = Results(orig_img=img, path="test_path.jpg", names=names, boxes=boxes, masks=masks, keypoints=poses)
    return res


def test_bbox_import(yolo_result):
    actual = import_yolo_result("bbox", yolo_result)
    expected = [
        IRBBoxImageAnnotation(
            left=10,
            top=10,
            width=10,
            height=20,
            categories={"cat": 0.9},
            state=CoordinateStyle.DENORMALIZED,
            image_width=640,
            image_height=480,
        ),
        IRBBoxImageAnnotation(
            left=30,
            top=30,
            width=20,
            height=10,
            categories={"dog": 0.8},
            state=CoordinateStyle.DENORMALIZED,
            image_width=640,
            image_height=480,
        ),
    ]

    assert actual == expected


def test_segmentation_import(yolo_result):
    actual = import_yolo_result("segmentation", yolo_result)

    points = [
        [(10, 10), (10, 30), (20, 30), (20, 10)],
        [(30, 30), (30, 40), (50, 40), (50, 30)],
    ]

    points = [[IRSegmentationPoint(x=x, y=y) for x, y in ps] for ps in points]

    expected = [
        IRSegmentationImageAnnotation(
            categories={"cat": 0.9},
            points=points[0],
            state=CoordinateStyle.DENORMALIZED,
            image_width=640,
            image_height=480,
        ),
        IRSegmentationImageAnnotation(
            categories={"dog": 0.8},
            points=points[1],
            state=CoordinateStyle.DENORMALIZED,
            image_width=640,
            image_height=480,
        ),
    ]

    assert actual == expected


def test_pose_import(yolo_result):
    points = [
        [(10, 10), (20, 20), (20, 30)],
        [(30, 30), (30, 35), (40, 40)],
    ]

    points = [[IRPosePoint(x=x, y=y) for x, y in ps] for ps in points]

    expected = [
        IRPoseImageAnnotation(
            categories={"cat": 0.9},
            left=10,
            top=10,
            width=10,
            height=20,
            points=points[0],
            state=CoordinateStyle.DENORMALIZED,
            image_width=640,
            image_height=480,
        ),
        IRPoseImageAnnotation(
            categories={"dog": 0.8},
            left=30,
            top=30,
            width=20,
            height=10,
            points=points[1],
            state=CoordinateStyle.DENORMALIZED,
            image_width=640,
            image_height=480,
        ),
    ]
