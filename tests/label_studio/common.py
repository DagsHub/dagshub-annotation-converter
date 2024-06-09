import json
from typing import List, Dict


def generate_annotation(annotation: Dict, annotation_type: str, annotation_id: str) -> Dict:
    return {
        "original_width": 100,
        "original_height": 200,
        "image_rotation": 0.0,
        "type": annotation_type,
        "id": annotation_id,
        "origin": "manual",
        "to_name": "image",
        "from_name": "label",
        "value": annotation,
    }


def generate_task(annotations: List[Dict]) -> str:
    task_layout = {
        "annotations": [
            {
                "completed_by": 1,
                "result": annotations,
                "ground_truth": True,
            }
        ],
        "meta": {},
        "data": {"image": "/path/to/image.jpg"},
        "project": 0,
        "created_at": "2021-10-01T00:00:00Z",
        "updated_at": "2021-10-01T00:00:00Z",
        "id": 1,
    }
    return json.dumps(task_layout)
