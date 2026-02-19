from dagshub_annotation_converter.ir.video.annotations.base import IRVideoAnnotationBase


class IRVideoBBoxAnnotation(IRVideoAnnotationBase):
    """
    Bounding box annotation for video object tracking.

    Coordinate format: (left, top, width, height).
    """

    left: float
    top: float
    width: float
    height: float
    rotation: float = 0.0
    """Rotation in degrees (pivot point: top-left)."""
    visibility: float = 1.0
    """Visibility/occlusion ratio (0-1). 1.0 = fully visible."""

    def _require_dimensions_for_coordinate_conversion(self):
        if self.image_width is None or self.image_height is None:
            raise ValueError("Cannot normalize/denormalize video annotation without image_width/image_height")

    def _normalize(self):
        self._require_dimensions_for_coordinate_conversion()
        assert self.image_width is not None
        assert self.image_height is not None
        self.left = self.left / self.image_width
        self.top = self.top / self.image_height
        self.width = self.width / self.image_width
        self.height = self.height / self.image_height

    def _denormalize(self):
        self._require_dimensions_for_coordinate_conversion()
        assert self.image_width is not None
        assert self.image_height is not None
        self.left = self.left * self.image_width
        self.top = self.top * self.image_height
        self.width = self.width * self.image_width
        self.height = self.height * self.image_height
