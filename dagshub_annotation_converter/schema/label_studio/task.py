import datetime
import random
import sys
import uuid
from typing import Any, Sequence

from pydantic import BaseModel, SerializeAsAny, Field

from dagshub_annotation_converter.schema.label_studio.abc import AnnotationResultABC


class AnnotationsContainer(BaseModel):
    completed_by: int = 1
    result: list[SerializeAsAny[AnnotationResultABC]] = []
    ground_truth: bool = False


class LabelStudioTask(BaseModel):
    annotations: list[AnnotationsContainer] = [AnnotationsContainer()]
    meta: dict[str, Any] = {}
    data: dict[str, Any] = {}
    project: int = 0
    created_at: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    updated_at: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
    id: int = Field(default_factory=lambda: random.randint(0, 2**63 - 1))

    def add_annotation(self, annotation: AnnotationResultABC):
        if len(self.annotations) == 0:
            self.annotations.append(AnnotationsContainer())
        self.annotations[0].result.append(annotation)

    def add_annotations(self, annotations: Sequence[AnnotationResultABC]):
        for ann in annotations:
            self.add_annotation(ann)
