from lxml.etree import ElementBase
import lxml.etree

from dagshub_annotation_converter.refactored.formats.cvat import (
    CVATContext,
    parse_box,
    parse_polygon,
    parse_points,
    parse_skeleton,
)
from dagshub_annotation_converter.refactored.ir.image import (
    NormalizationState,
    IRBBoxAnnotation,
    IRSegmentationAnnotation,
    IRSegmentationPoint,
    IRPosePoint,
    IRPoseAnnotation,
)


def wrap_in_image_tag(data: str) -> str:
    return f'<image id="0" name="000.png" width="1920" height="1200">{data}</image>'


def to_xml(data: str) -> tuple[ElementBase, ElementBase]:
    """Returns the image element + annotation element"""
    return lxml.etree.fromstring(wrap_in_image_tag(data)), lxml.etree.fromstring(data)


def test_box():
    data = """
    <box label="Person" source="manual" occluded="0" xtl="654.53" ytl="247.76" xbr="1247.56" ybr="1002.98" z_order="0">
    </box>
    """
    image, annotation = to_xml(data)

    ctx = CVATContext()

    actual = parse_box(ctx, annotation, image)

    expected = IRBBoxAnnotation(
        filename="000.png",
        category=ctx.categories["Person"],
        image_width=1920,
        image_height=1200,
        state=NormalizationState.DENORMALIZED,
        top=247.76,
        left=654.53,
        width=1247.56 - 654.53,
        height=1002.98 - 247.76,
    )

    assert expected == actual


def test_segmentation():
    data = """
    <polygon label="Ship" source="manual" occluded="0" 
    points="874.39,919.17;669.02,827.23;645.14,845.14;0.00,562.37;0.00,475.08;863.64,821.26;899.46,858.27;893.49,894.09"
    z_order="-1">
    </polygon>
    """

    image, annotation = to_xml(data)

    ctx = CVATContext()

    actual = parse_polygon(ctx, annotation, image)

    expected_points = []
    points = (
        "874.39,919.17;669.02,827.23;645.14,845.14;0.00,562.37;0.00,475.08;"
        + "863.64,821.26;899.46,858.27;893.49,894.09"
    ).split(";")
    for p in points:
        x, y = p.split(",")
        expected_points.append(IRSegmentationPoint(x=float(x), y=float(y)))

    expected = IRSegmentationAnnotation(
        filename="000.png",
        category=ctx.categories["Ship"],
        image_width=1920,
        image_height=1200,
        state=NormalizationState.DENORMALIZED,
        points=expected_points,
    )

    assert expected == actual


def test_points():
    data = """
    <points label="Baby Yoda" source="manual" occluded="0" 
        points="697.51,665.77;674.63,658.81;672.64,761.29" z_order="0">
    </points>
    """

    image, annotation = to_xml(data)

    ctx = CVATContext()

    actual = parse_points(ctx, annotation, image)

    expected_points = [
        IRPosePoint(x=697.51, y=665.77),
        IRPosePoint(x=674.63, y=658.81),
        IRPosePoint(x=672.64, y=761.29),
    ]

    expected = IRPoseAnnotation.from_points(
        filename="000.png",
        category=ctx.categories["Baby Yoda"],
        image_width=1920,
        image_height=1200,
        state=NormalizationState.DENORMALIZED,
        points=expected_points,
    )

    assert expected == actual


def test_skeleton():
    data = """
    <skeleton label="Yoda" source="manual" z_order="0">
      <points label="4" source="manual" outside="0" occluded="0" points="1249.72,310.54">
      </points>
      <points label="2" source="manual" outside="0" occluded="0" points="969.80,406.69">
      </points>
      <points label="1" source="manual" outside="0" occluded="0" points="797.51,449.77">
      </points>
      <points label="7" source="manual" outside="0" occluded="1" points="966.08,497.59">
      </points>
      <points label="6" source="manual" outside="0" occluded="0" points="846.27,520.51">
      </points>
      <points label="5" source="manual" outside="0" occluded="0" points="908.36,455.94">
      </points>
      <points label="3" source="manual" outside="0" occluded="0" points="349.30,450.10">
      </points>
    </skeleton>
    """

    image, annotation = to_xml(data)

    ctx = CVATContext()

    actual = parse_skeleton(ctx, annotation, image)

    # NOTE: order is important here!
    expected_points = [
        IRPosePoint(x=797.51, y=449.77, visible=True),
        IRPosePoint(x=969.80, y=406.69, visible=True),
        IRPosePoint(x=349.30, y=450.10, visible=True),
        IRPosePoint(x=1249.72, y=310.54, visible=True),
        IRPosePoint(x=908.36, y=455.94, visible=True),
        IRPosePoint(x=846.27, y=520.51, visible=True),
        IRPosePoint(x=966.08, y=497.59, visible=False),
    ]

    expected = IRPoseAnnotation.from_points(
        filename="000.png",
        category=ctx.categories["Yoda"],
        image_width=1920,
        image_height=1200,
        state=NormalizationState.DENORMALIZED,
        points=expected_points,
    )

    assert expected == actual
