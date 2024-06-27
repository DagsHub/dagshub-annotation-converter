from lxml.etree import ElementBase

from dagshub_annotation_converter.formats.cvat.context import parse_image_tag, CVATContext
from dagshub_annotation_converter.ir.image import IRPoseAnnotation, IRPosePoint, NormalizationState


def parse_skeleton(context: CVATContext, elem: ElementBase, containing_image: ElementBase) -> IRPoseAnnotation:
    # Points also contain the labels, for consistent ordering in LS, they are later sorted
    points: list[tuple[str, IRPosePoint]] = []

    category = context.categories.get_or_create(str(elem.attrib["label"]))

    image_info = parse_image_tag(containing_image)

    for point_elem in elem:
        x, y = point_elem.attrib["points"].split(",")
        points.append(
            (
                point_elem.attrib["label"],
                IRPosePoint(x=float(x), y=float(y), visible=point_elem.attrib["occluded"] == "0"),
            )
        )

    all_labels_ints = all(map(lambda tup: tup[0].isdigit(), points))

    # sort points by the label
    if all_labels_ints:
        points = sorted(points, key=lambda tup: int(tup[0]))
    else:
        points = sorted(points, key=lambda tup: tup[0])

    res_points = list(map(lambda tup: tup[1], points))

    return IRPoseAnnotation.from_points(
        category=category,
        points=res_points,
        state=NormalizationState.DENORMALIZED,
        image_width=image_info.width,
        image_height=image_info.height,
        filename=image_info.name,
    )
