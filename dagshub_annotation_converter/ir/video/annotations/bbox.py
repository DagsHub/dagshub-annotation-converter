from dagshub_annotation_converter.ir.video.annotations.base import IRVideoFrameAnnotationBase


class IRVideoBBoxFrameAnnotation(IRVideoFrameAnnotationBase):
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

    @staticmethod
    def _lerp(start: float, end: float, step_ratio: float) -> float:
        return start + (end - start) * step_ratio

    def interpolate(
        self,
        next_annotation: IRVideoFrameAnnotationBase,
        step_ratio: float,
    ) -> "IRVideoBBoxFrameAnnotation":
        if not isinstance(next_annotation, IRVideoBBoxFrameAnnotation):
            raise TypeError("IRVideoBBoxFrameAnnotation can only interpolate against another bbox annotation")
        if self.coordinate_style != next_annotation.coordinate_style:
            raise ValueError("Cannot interpolate annotations with different coordinate styles")

        interpolated = self.model_copy(deep=True)
        interpolated.frame_number = round(
            self._lerp(self.frame_number, next_annotation.frame_number, step_ratio)
        )
        interpolated.keyframe = False
        interpolated.left = self._lerp(self.left, next_annotation.left, step_ratio)
        interpolated.top = self._lerp(self.top, next_annotation.top, step_ratio)
        interpolated.width = self._lerp(self.width, next_annotation.width, step_ratio)
        interpolated.height = self._lerp(self.height, next_annotation.height, step_ratio)
        interpolated.rotation = self._lerp(self.rotation, next_annotation.rotation, step_ratio)
        interpolated.visibility = self._lerp(self.visibility, next_annotation.visibility, step_ratio)
        if self.timestamp is not None and next_annotation.timestamp is not None:
            interpolated.timestamp = self._lerp(self.timestamp, next_annotation.timestamp, step_ratio)
        else:
            interpolated.timestamp = None
        return interpolated
