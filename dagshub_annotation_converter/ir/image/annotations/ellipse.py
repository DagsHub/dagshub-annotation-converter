from dagshub_annotation_converter.ir.image import IRImageAnnotationBase


class IREllipseImageAnnotation(IRImageAnnotationBase):
    top: float
    left: float
    radiusX: float
    radiusY: float
    rotation: float = 0.0
    """Rotation in degrees (pivot point - top-left)"""

    def _normalize(self):
        self.top = self.top / self.image_height
        self.left = self.left / self.image_width
        self.radiusX = self.radiusX / self.image_width
        self.radiusY = self.radiusY / self.image_height

    def _denormalize(self):
        self.top = self.top * self.image_height
        self.left = self.left * self.image_width
        self.radiusX = self.radiusX * self.image_width
        self.radiusY = self.radiusY * self.image_height
