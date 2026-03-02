from dagshub_annotation_converter.ir.video.annotations.base import IRVideoAnnotationBase


class IRVideoBBoxAnnotation(IRVideoAnnotationBase):
    """
    Bounding box annotation for video object tracking.
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
        if self.video_width is None or self.video_height is None:
            raise ValueError("Cannot normalize/denormalize video annotation without video_width/video_height")

    def _normalize(self):
        self._require_dimensions_for_coordinate_conversion()
        self.left = self.left / self.video_width
        self.top = self.top / self.video_height
        self.width = self.width / self.video_width
        self.height = self.height / self.video_height

    def _denormalize(self):
        self._require_dimensions_for_coordinate_conversion()
        self.left = self.left * self.video_width
        self.top = self.top * self.video_height
        self.width = self.width * self.video_width
        self.height = self.height * self.video_height
