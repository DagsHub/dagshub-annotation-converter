import pytest

from dagshub_annotation_converter.formats.cvat import parse_ellipse
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask
from dagshub_annotation_converter.ir.image import IREllipseImageAnnotation, CoordinateStyle
from tests.cvat.test_parsers import to_xml


@pytest.fixture()
def cvat_ellipse():
    data = """
    <ellipse label="circle1" source="manual" occluded="0" cx="392.23" cy="322.84" rx="205.06" ry="202.84" z_order="0" 
        group_id="1" foo="bar">
    </ellipse> 
    """

    image, annotation = to_xml(data)
    return parse_ellipse(annotation, image)


def test_extra_values(cvat_ellipse):
    expected = IREllipseImageAnnotation(
        filename="000.png",
        categories={"circle1": 1.0},
        image_width=1920,
        image_height=1200,
        coordinate_style=CoordinateStyle.DENORMALIZED,
        center_x=392,
        center_y=323,
        radius_x=205.06,
        radius_y=202.84,
        rotation=0.0,
        meta={"group_id": "1", "foo": "bar", "source": "manual", "occluded": "0", "z_order": "0"},
    )

    assert cvat_ellipse == expected


def test_extras_persist_in_labelstudio(cvat_ellipse):
    ls_task = LabelStudioTask()
    ls_task.add_ir_annotation(cvat_ellipse)
    task_dump = ls_task.model_dump()
    meta = task_dump["annotations"][0]["result"][0].get("meta")
    assert meta == {
        "group_id": "1",
        "foo": "bar",
        "source": "manual",
        "occluded": "0",
        "z_order": "0",
    }
