from pathlib import Path

from dagshub_annotation_converter.converters.yolo import export_to_fs, _get_common_folder_with_part
from dagshub_annotation_converter.formats.yolo import YoloContext
from dagshub_annotation_converter.ir.image import (
    CoordinateStyle,
    IRBBoxImageAnnotation,
    IRSegmentationImageAnnotation,
    IRSegmentationPoint,
    IRPoseImageAnnotation,
    IRPosePoint,
)

import pytest


def test_bbox_export(tmp_path):
    ctx = YoloContext(annotation_type="bbox", path=Path("data"))
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    annotations = [
        IRBBoxImageAnnotation(
            filename="images/cats/1.jpg",
            categories={"cat": 1.0},
            top=0.0,
            left=0.0,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
        IRBBoxImageAnnotation(
            filename="images/dogs/2.jpg",
            categories={"dog": 1.0},
            top=0.5,
            left=0.5,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations, export_dir=tmp_path)

    assert p == tmp_path / "yolo_dagshub.yaml"

    assert (tmp_path / "yolo_dagshub.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


def test_segmentation_export(tmp_path):
    ctx = YoloContext(annotation_type="segmentation", path=Path("data"))
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    annotations = [
        IRSegmentationImageAnnotation(
            filename="images/cats/1.jpg",
            categories={"cat": 1.0},
            points=[IRSegmentationPoint(x=0.0, y=0.5), IRSegmentationPoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
        IRSegmentationImageAnnotation(
            filename="images/dogs/2.jpg",
            categories={"dog": 1.0},
            points=[IRSegmentationPoint(x=0.0, y=0.5), IRSegmentationPoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations, export_dir=tmp_path)

    assert p == tmp_path / "yolo_dagshub.yaml"

    assert (tmp_path / "yolo_dagshub.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


def test_pose_export(tmp_path):
    ctx = YoloContext(annotation_type="pose", path=Path("data"))
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    ctx.keypoints_in_annotation = 2
    annotations = [
        IRPoseImageAnnotation.from_points(
            filename="images/cats/1.jpg",
            categories={"cat": 1.0},
            points=[IRPosePoint(x=0.0, y=0.5), IRPosePoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
        IRPoseImageAnnotation.from_points(
            filename="images/dogs/2.jpg",
            categories={"dog": 1.0},
            points=[IRPosePoint(x=0.0, y=0.5), IRPosePoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations, export_dir=tmp_path)

    assert p == tmp_path / "yolo_dagshub.yaml"

    assert (tmp_path / "yolo_dagshub.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


def test_not_exporting_wrong_annotations(tmp_path):
    ctx = YoloContext(annotation_type="bbox", path=Path("data"))
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    annotations = [
        IRBBoxImageAnnotation(
            filename="images/cats/1.jpg",
            categories={"cat": 1.0},
            top=0.0,
            left=0.0,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
        IRSegmentationImageAnnotation(
            filename="images/dogs/2.jpg",
            categories={"dog": 1.0},
            points=[IRSegmentationPoint(x=0.0, y=0.5), IRSegmentationPoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations, export_dir=tmp_path)

    assert p == tmp_path / "yolo_dagshub.yaml"

    assert (tmp_path / "yolo_dagshub.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert not (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


@pytest.mark.parametrize(
    "paths, prefix, expected",
    (
        (["/a/b/c", "/a/b/d", "/a/b/e"], "b", "/a/b"),
        (["/a/b/c", "/a/b/d", "/a/b/e"], "b", "/a/b"),
        (["/a/b/c", "/a/b/d", "/a/b/b"], "b", "/a/b"),
        (["/a/b/c", "/a/b/d", "/a/b/e/b"], "b", "/a/b"),
        (["/a/b/c", "/a/e/b", "/a/e/b/b"], "b", "/a/b"),
        (["/a/b/c", "/a/b/d", "/some_other/b/e"], "b", None),  # Fails because there are two different common b folders
        (["/a/b/c", "/a/some_other/d", "/a/b/e"], "b", "/a/b"),
        (["/a/b/c", "/a/bbb/d", "/a/b/e"], "b", "/a/b"),
    ),
)
def test__get_common_folder_with_part(paths, prefix, expected):
    paths = [Path(p) for p in paths]
    actual = _get_common_folder_with_part(paths, prefix)

    if expected is not None:
        expected = Path(expected)

    assert actual == expected


def test_export_with_image_in_path(tmp_path):
    ctx = YoloContext(annotation_type="bbox", path=Path("data/images"))
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    annotations = [
        IRBBoxImageAnnotation(
            filename="cats/1.jpg",
            categories={"cat": 1.0},
            top=0.0,
            left=0.0,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
        IRBBoxImageAnnotation(
            filename="dogs/2.jpg",
            categories={"dog": 1.0},
            top=0.5,
            left=0.5,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations, export_dir=tmp_path)

    assert p == tmp_path / "yolo_dagshub.yaml"

    assert (tmp_path / "yolo_dagshub.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


def test_pose_export_point(tmp_path):
    ctx = YoloContext(annotation_type="pose", path=Path("data"))
    ctx.categories.add(name="person")
    annotations = [
        IRPoseImageAnnotation.from_points(
            filename="images/people/1.jpg",
            categories={"person": 1.0},
            points=[
                IRPosePoint(x=0.1, y=0.1, visibility=2),
                IRPosePoint(x=0.2, y=0.2, visibility=2),
                IRPosePoint(x=0.3, y=0.3, visibility=2),
            ],
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
        IRPoseImageAnnotation.from_points(
            filename="images/people/2.jpg",
            categories={"person": 1.0},
            points=[
                IRPosePoint(x=0.4, y=0.4, visibility=2),
                IRPosePoint(x=0.5, y=0.5, visibility=2),
                IRPosePoint(x=0.6, y=0.6, visibility=2),
            ],
            image_width=100,
            image_height=200,
            coordinate_style=CoordinateStyle.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations, export_dir=tmp_path)

    assert ctx.keypoints_in_annotation == 3  # Should be inferred from the annotations
    assert p == tmp_path / "yolo_dagshub.yaml"
    assert (tmp_path / "yolo_dagshub.yaml").exists()
    assert (tmp_path / "data" / "labels" / "people" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "people" / "2.txt").exists()
