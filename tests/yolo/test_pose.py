from dagshub_annotation_converter.formats.yolo.pose import import_pose_from_string, export_pose
from dagshub_annotation_converter.ir.image import CoordinateStyle
from dagshub_annotation_converter.ir.image.annotations.pose import IRPosePoint, IRPoseImageAnnotation


def test_pose_3dim(yolo_context):
    points = [
        IRPosePoint(x=0.5, y=0.5, visible=True),
        IRPosePoint(x=0.75, y=0.75, visible=False),
        IRPosePoint(x=0.5, y=0.75, visible=True),
    ]
    expected = IRPoseImageAnnotation(
        categories={yolo_context.categories[0].name: 1.0},
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    actual = import_pose_from_string(
        context=yolo_context,
        annotation="0 0.75 0.75 0.5 0.5 0.5 0.5 1 0.75 0.75 0 0.5 0.75 1",
        image_width=100,
        image_height=200,
    )

    assert expected == actual


def test_pose_2dim(yolo_context):
    yolo_context.keypoint_dim = 2

    points = [
        IRPosePoint(x=0.5, y=0.5),
        IRPosePoint(x=0.75, y=0.75),
        IRPosePoint(x=0.5, y=0.75),
    ]
    expected = IRPoseImageAnnotation(
        categories={yolo_context.categories[0].name: 1.0},
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    actual = import_pose_from_string(
        context=yolo_context,
        annotation="0 0.75 0.75 0.5 0.5 0.5 0.5 0.75 0.75 0.5 0.75",
        image_width=100,
        image_height=200,
    )

    assert expected == actual


def test_export_pose_3dim(yolo_context):
    points = [
        IRPosePoint(x=0.5, y=0.5, visible=True),
        IRPosePoint(x=0.75, y=0.75, visible=False),
        IRPosePoint(x=0.5, y=0.75, visible=True),
    ]
    annotation = IRPoseImageAnnotation(
        categories={yolo_context.categories[0].name: 1.0},
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    expected = "0 0.75 0.75 0.5 0.5 0.5 0.5 1 0.75 0.75 0 0.5 0.75 1"
    assert expected == export_pose(annotation, yolo_context)


def test_export_pose_2dim(yolo_context):
    yolo_context.keypoint_dim = 2
    # NOTE: 2nd point gets skipped because it's not visible
    points = [
        IRPosePoint(x=0.5, y=0.5, visible=None),
        IRPosePoint(x=0.75, y=0.75, visible=False),
        IRPosePoint(x=0.5, y=0.75, visible=True),
    ]
    annotation = IRPoseImageAnnotation(
        categories={yolo_context.categories[0].name: 1.0},
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        coordinate_style=CoordinateStyle.NORMALIZED,
    )

    expected = "0 0.75 0.75 0.5 0.5 0.5 0.5 0.5 0.75"
    assert expected == export_pose(annotation, yolo_context)
