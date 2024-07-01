from abc import abstractmethod
from typing import Optional

from pydantic import BaseModel
from typing_extensions import Self

from dagshub_annotation_converter.ir.image import CoordinateStyle


class IRImageAnnotationBase(BaseModel):
    """
    Common class for all intermediary annotations
    """

    filename: Optional[str] = None

    category: str
    image_width: int
    image_height: int
    state: CoordinateStyle
    imported_id: Optional[str] = None

    def with_filename(self, filename: str) -> Self:
        self.filename = filename
        return self

    def normalized(self) -> Self:
        """
        Returns a copy with all parameters in the annotation normalized
        """
        if self.state == CoordinateStyle.NORMALIZED:
            return self.model_copy()
        normalized = self.model_copy()
        normalized._normalize()
        normalized.state = CoordinateStyle.NORMALIZED
        return normalized

    @abstractmethod
    def _normalize(self):
        """
        Every annotation should implement this to normalize itself
        """
        ...

    def denormalized(self) -> Self:
        """
        Returns a copy with all parameters in the annotation denormalized
        """
        if self.state == CoordinateStyle.DENORMALIZED:
            return self.model_copy()
        denormalized = self.model_copy()
        denormalized._denormalize()
        denormalized.state = CoordinateStyle.DENORMALIZED
        return denormalized

    @abstractmethod
    def _denormalize(self):
        """
        Every annotation should implement this to denormalize itself
        """
        ...
