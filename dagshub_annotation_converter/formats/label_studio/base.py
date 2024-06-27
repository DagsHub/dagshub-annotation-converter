import uuid
from abc import abstractmethod
from typing import Sequence

from pydantic import BaseModel, Field

from dagshub_annotation_converter.ir.image import IRAnnotationBase


class AnnotationResultABC(BaseModel):
    @abstractmethod
    def to_ir_annotation(self) -> Sequence[IRAnnotationBase]:
        """
        Convert LabelStudio annotation to 0..n DAGsHub IR annotations.

        Note: This method has a potential side effect of adding new categories.
        """
        ...


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
    def to_ir_annotation(self) -> Sequence[IRAnnotationBase]:
        """
        Convert LabelStudio annotation to 0..n DAGsHub IR annotations.

        Note: This method has a potential side effect of adding new categories.
        """
        ...
