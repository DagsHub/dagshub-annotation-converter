from dagshub_annotation_converter.refactored.formats.yolo.pose import import_pose_from_string, export_pose
from dagshub_annotation_converter.refactored.ir.image import NormalizationState
from dagshub_annotation_converter.refactored.ir.image.annotations.pose import IRPosePoint, IRPoseAnnotation


def test_pose_3dim(yolo_context):
    points = [
        IRPosePoint(x=0.5, y=0.5, visible=True),
        IRPosePoint(x=0.75, y=0.75, visible=False),
        IRPosePoint(x=0.5, y=0.75, visible=True),
    ]
    expected = IRPoseAnnotation(
        category=yolo_context.categories[0],
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        state=NormalizationState.NORMALIZED,
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
    expected = IRPoseAnnotation(
        category=yolo_context.categories[0],
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        state=NormalizationState.NORMALIZED,
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
    annotation = IRPoseAnnotation(
        category=yolo_context.categories[0],
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        state=NormalizationState.NORMALIZED,
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
    annotation = IRPoseAnnotation(
        category=yolo_context.categories[0],
        top=0.5,
        left=0.5,
        width=0.5,
        height=0.5,
        points=points,
        image_width=100,
        image_height=200,
        state=NormalizationState.NORMALIZED,
    )

    expected = "0 0.75 0.75 0.5 0.5 0.5 0.5 0.5 0.75"
    assert expected == export_pose(annotation, yolo_context)