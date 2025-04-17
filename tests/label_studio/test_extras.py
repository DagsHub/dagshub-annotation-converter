import json

import pytest

from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask


def test_extra_values_persist():
    json_data = {
        "id": 12345,
        "project": "someProject",
        "meta": {},
        "created_at": "2025-03-25T17:30:09.282994Z",
        "updated_at": "2025-03-25T17:30:09.283290Z",
        "data": {"image": "some_stuff"},
        "predictions": [],
        "annotations": [],
        "super_extra_field": "HELLO!!!",
        "super_extra_field2": 25,
        "super_extra_field3": {"Hello": "World"},
    }

    parsed = LabelStudioTask.model_validate_json(json.dumps(json_data))
    serialized = parsed.model_dump_json()
    json_dict = json.loads(serialized)
    assert json_dict["super_extra_field"] == "HELLO!!!"
    assert json_dict["super_extra_field2"] == 25
    assert json_dict["super_extra_field3"] == {"Hello": "World"}


@pytest.fixture
def task_with_extras_in_annotations():
    return {
        "id": 12345,
        "project": "someProject",
        "meta": {},
        "created_at": "2025-03-25T17:30:09.282994Z",
        "updated_at": "2025-03-25T17:30:09.283290Z",
        "data": {"image": "some_stuff"},
        "predictions": [],
        "annotations": [
            {
                "completed_by": 1,
                "result": [
                    {
                        "original_width": 1600.0,
                        "original_height": 600.0,
                        "image_rotation": 0.0,
                        "id": "52ca2c0536",
                        "type": "rectanglelabels",
                        "origin": "manual",
                        "from_name": "bbox",
                        "to_name": "image",
                        "group_id": 0,
                        "super_extra_field": "HELLO!!! result",
                        "super_extra_field2": 255,
                        "super_extra_field3": {"Hello": "World result"},
                        "value": {
                            "x": 0.1,
                            "y": 1.0,
                            "width": 10.0,
                            "height": 14.5,
                            "rotation": 0.0,
                            "rectanglelabels": ["doggy"],
                        },
                    },
                ],
                "ground_truth": True,
                "id": "52ca2c0536",
                "type": "rectanglelabels",
                "super_extra_field": "HELLO!!!",
                "super_extra_field2": 25,
                "super_extra_field3": {"Hello": "World"},
            }
        ],
    }


def test_extra_values_persist_in_annotations(task_with_extras_in_annotations):
    parsed = LabelStudioTask.model_validate_json(json.dumps(task_with_extras_in_annotations))
    serialized = parsed.model_dump_json()
    json_dict = json.loads(serialized)
    annotation = json_dict["annotations"][0]
    assert annotation["super_extra_field"] == "HELLO!!!"
    assert annotation["super_extra_field2"] == 25
    assert annotation["super_extra_field3"] == {"Hello": "World"}

    result = annotation["result"][0]
    assert result["super_extra_field"] == "HELLO!!! result"
    assert result["super_extra_field2"] == 255
    assert result["super_extra_field3"] == {"Hello": "World result"}


def test_extra_values_persist_through_add_annotations(task_with_extras_in_annotations):
    parsed = LabelStudioTask.model_validate_json(json.dumps(task_with_extras_in_annotations))

    new_task = LabelStudioTask()
    new_task.add_ir_annotations(parsed.to_ir_annotations())

    serialized = new_task.model_dump_json()
    json_dict = json.loads(serialized)

    result = json_dict["annotations"][0]["result"][0]
    assert result["super_extra_field"] == "HELLO!!! result"
    assert result["super_extra_field2"] == 255
    assert result["super_extra_field3"] == {"Hello": "World result"}
