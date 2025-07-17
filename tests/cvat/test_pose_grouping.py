from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Sequence

from dagshub_annotation_converter.converters.cvat import load_cvat_from_xml_file
from dagshub_annotation_converter.features import FEATURE_CVAT_POSE_GROUPING_KEY
from dagshub_annotation_converter.ir.image import IRPoseImageAnnotation, IRBBoxImageAnnotation
from dagshub_annotation_converter.ir.image.annotations.base import IRAnnotationBase
from tests.util import set_env_var

_filepath = Path(__file__).parent / "res/pose_grouping.xml"


def test_pose_grouping():
    with set_env_var(FEATURE_CVAT_POSE_GROUPING_KEY, "t"):
        annotations = load_cvat_from_xml_file(_filepath)["000.png"]

        categorized = _group_by_category(annotations)

        # Pointy_1 - good case (bbox + points)
        assert len(categorized["Pointy_1"]) == 1
        assert isinstance(categorized["Pointy_1"][0], IRPoseImageAnnotation)

        # Pointy_2 - has multiple bboxes, should not be grouped
        assert len(categorized["Pointy_2"]) == 3
        assert any(isinstance(ann, IRPoseImageAnnotation) for ann in categorized["Pointy_2"])
        assert any(isinstance(ann, IRBBoxImageAnnotation) for ann in categorized["Pointy_2"])

        # Pointy_3 - good case (bbox + skeleton)
        assert len(categorized["Pointy_3"]) == 1
        assert isinstance(categorized["Pointy_3"][0], IRPoseImageAnnotation)

        # Pointy_misc - the rest, should be 3 annotations
        assert len(categorized["Pointy_misc"]) == 3


def _group_by_category(annotations: Sequence[IRAnnotationBase]) -> Dict[str, List[IRAnnotationBase]]:
    grouped = defaultdict(list)
    for annotation in annotations:
        category = annotation.ensure_has_one_category()
        grouped[category].append(annotation)
    return grouped
