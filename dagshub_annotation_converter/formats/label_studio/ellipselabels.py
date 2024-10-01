from typing import List, Sequence

from dagshub_annotation_converter.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase, IREllipseImageAnnotation, CoordinateStyle
from dagshub_annotation_converter.util.pydantic_util import ParentModel


class EllipseLabelsAnnotationsValue(ParentModel):
    x: float
    y: float
    radiusX: float
    radiusY: float
    rotation: float = 0.0
    ellipselabels: List[str]


class EllipseLabelsAnnotation(ImageAnnotationResultABC):
    value: EllipseLabelsAnnotationsValue
    type: str = "ellipselabels"

    def to_ir_annotation(self) -> Sequence[IRImageAnnotationBase]:
        res = IREllipseImageAnnotation(
            categories={self.value.ellipselabels[0]: 1.0},
            coordinate_style=CoordinateStyle.NORMALIZED,
            top=self.value.y / 100,
            left=self.value.x / 100,
            radiusX=self.value.radiusX / 100,
            radiusY=self.value.radiusY / 100,
            rotation=self.value.rotation,
            image_width=self.original_width,
            image_height=self.original_height,
        )
        res.imported_id = self.id
        return [res]

    @staticmethod
    def from_ir_annotation(ir_annotation: IRImageAnnotationBase) -> Sequence["ImageAnnotationResultABC"]:
        assert isinstance(ir_annotation, IREllipseImageAnnotation)

        ir_annotation = ir_annotation.normalized()
        category = ir_annotation.ensure_has_one_category()

        return [
            EllipseLabelsAnnotation(
                original_width=ir_annotation.image_width,
                original_height=ir_annotation.image_height,
                value=EllipseLabelsAnnotationsValue(
                    x=ir_annotation.left * 100,
                    y=ir_annotation.top * 100,
                    radiusX=ir_annotation.radiusX * 100,
                    radiusY=ir_annotation.radiusY * 100,
                    rotation=ir_annotation.rotation,
                    ellipselabels=[category],
                ),
            )
        ]