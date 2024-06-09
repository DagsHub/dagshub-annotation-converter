import uuid
from abc import abstractmethod
from typing import Sequence

from pydantic import BaseModel, Field

from dagshub_annotation_converter.refactored.ir.image import Categories, IRAnnotationBase


class AnnotationResultABC(BaseModel):
    pass


class ImageAnnotationResultABC(AnnotationResultABC):
    original_width: int
    original_height: int
    image_rotation: float
    type: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    origin: str = "manual"
    to_name: str = "image"
    from_name: str = "label"

    @abstractmethod
    def to_ir_annotation(self, categories: Categories) -> Sequence[IRAnnotationBase]:
        """
        Convert LabelStudio annotation to 0..n DAGsHub IR annotations.

        Note: This method has a potential side effect of adding new categories.
        """
        ...
