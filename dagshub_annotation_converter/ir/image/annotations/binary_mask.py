from typing import List

import numpy as np
from typing_extensions import Self

from dagshub_annotation_converter.ir.image.annotations.base import IRImageAnnotationBase
from dagshub_annotation_converter.ir.image.common import CoordinateStyle


class IRBinaryMaskImageAnnotation(IRImageAnnotationBase):
    """
    Binary pixel mask that covers the whole image. Each pixel is either in the mask or not.

    The mask is stored as uncompressed run-length encoding (RLE) over the image pixels in row-major order.
    The first run is background, then the runs alternate between foreground and background.
    The first run can be 0 if the mask starts with a foreground pixel.

    Example for a 3x2 image::

        mask:    0 1 1
                 1 0 0
        rle:     [1, 3, 2]

    The mask is always in pixels of the image, so normalization does not change it.
    """

    rle: List[int] = []

    def _normalize(self):
        pass

    def _denormalize(self):
        pass

    def to_mask(self) -> np.ndarray:
        """Returns the mask as a boolean array with the shape (image_height, image_width)."""
        total = self.image_width * self.image_height
        if sum(self.rle) != total:
            raise ValueError(
                f"Mask run lengths add up to {sum(self.rle)} pixels, "
                f"but the image has {total} pixels ({self.image_width}x{self.image_height})"
            )
        values = np.arange(len(self.rle)) % 2 == 1
        return np.repeat(values, self.rle).reshape(self.image_height, self.image_width)

    @classmethod
    def from_mask(cls, mask: np.ndarray, **kwargs) -> Self:
        """
        Creates the annotation from a 2D mask. Every non-zero value is foreground.

        ``image_width`` and ``image_height`` come from the shape of the mask.
        Other fields (for example ``categories``) go in ``kwargs``.
        """
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
        kwargs.setdefault("coordinate_style", CoordinateStyle.DENORMALIZED)
        return cls(
            image_height=mask.shape[0],
            image_width=mask.shape[1],
            rle=mask_to_rle(mask),
            **kwargs,
        )


def mask_to_rle(mask: np.ndarray) -> List[int]:
    """Encodes a mask to run lengths in the format of ``IRBinaryMaskImageAnnotation.rle``."""
    flat = mask.ravel() != 0
    if flat.size == 0:
        return []
    change_points = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [flat.size]))
    rle = np.diff(boundaries).tolist()
    if flat[0]:
        rle.insert(0, 0)
    return rle
