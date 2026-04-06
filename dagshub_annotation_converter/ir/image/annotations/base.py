from abc import abstractmethod

from dagshub_annotation_converter.ir.base import IRAnnotationBase, IRTaskAnnotation, MultipleCategoriesError

__all__ = [
    "IRAnnotationBase",
    "IRTaskAnnotation",
    "IRImageAnnotationBase",
    "MultipleCategoriesError",
]


class IRImageAnnotationBase(IRAnnotationBase):
    """
    Common class for all intermediary image annotations
    """

    image_width: int
    image_height: int

    @abstractmethod
    def _normalize(self):
        """
        Every annotation should implement this to normalize itself
        """
        ...

    @abstractmethod
    def _denormalize(self):
        """
        Every annotation should implement this to denormalize itself
        """
        ...
