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
    confidence: float = 1.0
    """Detection confidence score (0-1). Default: 1.0 for ground truth."""
    visibility: float = 1.0
    """Visibility/occlusion ratio (0-1). 1.0 = fully visible."""

    def _normalize(self):
        self.left = self.left / self.image_width
        self.top = self.top / self.image_height
        self.width = self.width / self.image_width
        self.height = self.height / self.image_height

    def _denormalize(self):
        self.left = self.left * self.image_width
        self.top = self.top * self.image_height
        self.width = self.width * self.image_width
        self.height = self.height * self.image_height
