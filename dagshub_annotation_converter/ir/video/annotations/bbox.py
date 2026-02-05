"""Video bounding box annotation with tracking support."""

from dagshub_annotation_converter.ir.video.annotations.base import IRVideoAnnotationBase


class IRVideoBBoxAnnotation(IRVideoAnnotationBase):
    """
    Bounding box annotation for video object tracking.
    
    Extends IRVideoAnnotationBase with bounding box coordinates
    and video-specific attributes like visibility and confidence.
    
    Coordinate format: (left, top, width, height) - same as CVAT MOT format.
    """

    left: float
    """X coordinate of the top-left corner."""
    
    top: float
    """Y coordinate of the top-left corner."""
    
    width: float
    """Width of the bounding box."""
    
    height: float
    """Height of the bounding box."""
    
    rotation: float = 0.0
    """Rotation in degrees (pivot point: top-left). Default: 0."""
    
    confidence: float = 1.0
    """Detection confidence score (0-1). Default: 1.0 for ground truth."""
    
    visibility: float = 1.0
    """Visibility/occlusion ratio (0-1). 1.0 = fully visible. Default: 1.0."""

    def _normalize(self):
        """Normalize coordinates to 0-1 range relative to image dimensions."""
        self.left = self.left / self.image_width
        self.top = self.top / self.image_height
        self.width = self.width / self.image_width
        self.height = self.height / self.image_height

    def _denormalize(self):
        """Denormalize coordinates to absolute pixel values."""
        self.left = self.left * self.image_width
        self.top = self.top * self.image_height
        self.width = self.width * self.image_width
        self.height = self.height * self.image_height
