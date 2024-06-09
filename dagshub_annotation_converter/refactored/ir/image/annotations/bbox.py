from dagshub_annotation_converter.refactored.ir.image.annotations.base import IRAnnotationBase


class IRBBoxAnnotation(IRAnnotationBase):
    top: float
    left: float
    width: float
    height: float

    def _normalize(self):
        self.top = self.top / self.image_height
        self.left = self.left / self.image_width
        self.width = self.width / self.image_width
        self.height = self.height / self.image_height

    def _denormalize(self):
        self.top = self.top * self.image_height
        self.left = self.left * self.image_width
        self.width = self.width * self.image_width
        self.height = self.height * self.image_height
