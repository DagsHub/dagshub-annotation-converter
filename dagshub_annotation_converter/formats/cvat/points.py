from lxml.etree import ElementBase

from dagshub_annotation_converter.formats.cvat.context import parse_image_tag, CVATContext
from dagshub_annotation_converter.ir.image import IRPoseAnnotation, IRPosePoint, NormalizationState


def parse_points(context: CVATContext, elem: ElementBase, containing_image: ElementBase) -> IRPoseAnnotation:
    points: list[IRPosePoint] = []

    category = context.categories.get_or_create(str(elem.attrib["label"]))

    image_info = parse_image_tag(containing_image)

    for point_str in elem.attrib["points"].split(";"):
        x, y = point_str.split(",")
        points.append(IRPosePoint(x=float(x), y=float(y)))

    return IRPoseAnnotation.from_points(
        category=category,
        points=points,
        state=NormalizationState.DENORMALIZED,
        image_width=image_info.width,
        image_height=image_info.height,
        filename=image_info.name,
    )
