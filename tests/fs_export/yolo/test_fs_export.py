from dagshub_annotation_converter.converters.yolo import export_to_fs
from dagshub_annotation_converter.formats.yolo import YoloContext
from dagshub_annotation_converter.ir.image import (
    NormalizationState,
    IRBBoxAnnotation,
    IRSegmentationAnnotation,
    IRSegmentationPoint,
    IRPoseAnnotation,
    IRPosePoint,
)


def test_bbox_export(tmp_path):
    ctx = YoloContext(annotation_type="bbox", path=tmp_path / "data")
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    annotations = [
        IRBBoxAnnotation(
            filename="images/cats/1.jpg",
            category="cat",
            top=0.0,
            left=0.0,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
        IRBBoxAnnotation(
            filename="images/dogs/2.jpg",
            category="dog",
            top=0.5,
            left=0.5,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations)

    assert p == tmp_path / "annotations.yaml"

    assert (tmp_path / "annotations.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


def test_segmentation_export(tmp_path):
    ctx = YoloContext(annotation_type="segmentation", path=tmp_path / "data")
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    annotations = [
        IRSegmentationAnnotation(
            filename="images/cats/1.jpg",
            category="cat",
            points=[IRSegmentationPoint(x=0.0, y=0.5), IRSegmentationPoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
        IRSegmentationAnnotation(
            filename="images/dogs/2.jpg",
            category="dog",
            points=[IRSegmentationPoint(x=0.0, y=0.5), IRSegmentationPoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations)

    assert p == tmp_path / "annotations.yaml"

    assert (tmp_path / "annotations.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


def test_pose_export(tmp_path):
    ctx = YoloContext(annotation_type="pose", path=tmp_path / "data")
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    ctx.keypoints_in_annotation = 2
    annotations = [
        IRPoseAnnotation.from_points(
            filename="images/cats/1.jpg",
            category="cat",
            points=[IRPosePoint(x=0.0, y=0.5), IRPosePoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
        IRPoseAnnotation.from_points(
            filename="images/dogs/2.jpg",
            category="dog",
            points=[IRPosePoint(x=0.0, y=0.5), IRPosePoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations)

    assert p == tmp_path / "annotations.yaml"

    assert (tmp_path / "annotations.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()


def test_not_exporting_wrong_annotations(tmp_path):
    ctx = YoloContext(annotation_type="bbox", path=tmp_path / "data")
    ctx.categories.add(name="cat")
    ctx.categories.add(name="dog")
    annotations = [
        IRBBoxAnnotation(
            filename="images/cats/1.jpg",
            category="cat",
            top=0.0,
            left=0.0,
            width=0.5,
            height=0.5,
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
        IRSegmentationAnnotation(
            filename="images/dogs/2.jpg",
            category="dog",
            points=[IRSegmentationPoint(x=0.0, y=0.5), IRSegmentationPoint(x=0.5, y=0.5)],
            image_width=100,
            image_height=200,
            state=NormalizationState.NORMALIZED,
        ),
    ]

    p = export_to_fs(ctx, annotations)

    assert p == tmp_path / "annotations.yaml"

    assert (tmp_path / "annotations.yaml").exists()
    assert (tmp_path / "data" / "labels" / "cats" / "1.txt").exists()
    assert not (tmp_path / "data" / "labels" / "dogs" / "2.txt").exists()