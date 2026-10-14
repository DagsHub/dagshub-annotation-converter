import numpy as np
import pytest

from dagshub_annotation_converter.formats.label_studio.brushlabels import (
    BrushLabelsAnnotation,
    decode_ls_rle,
    encode_ls_rle,
)
from dagshub_annotation_converter.formats.label_studio.task import LabelStudioTask, parse_ls_task
from dagshub_annotation_converter.ir.image import CoordinateStyle, IRBinaryMaskImageAnnotation
from tests.label_studio.common import generate_annotation, generate_task

# 2x2 mask [[1, 0], [0, 1]], encoded by label_studio_sdk.converter.brush.mask2rle
LS_RLE_2X2 = [0, 0, 0, 16, 57, 27, 252, 127, 243, 128, 71, 254]
MASK_2X2 = np.array([[1, 0], [0, 1]], dtype=bool)


def _random_mask(height: int, width: int) -> np.ndarray:
    return np.random.default_rng(0).random((height, width)) > 0.5


def test_decode_label_studio_rle():
    assert (decode_ls_rle(LS_RLE_2X2, 2, 2) == MASK_2X2).all()


def test_decode_literal_blocks():
    # Header: 8 values, word size 8, run sizes [3, 4, 8, 16]
    # Block: literal (0), run size index 0, length 8 -> 8 words: 4 x 0 then 4 x 255
    bits = f"{8:032b}{7:05b}" + "".join(f"{s - 1:04b}" for s in [3, 4, 8, 16])
    bits += "0" + "00" + f"{7:03b}" + "00000000" * 4 + "11111111" * 4
    bits += "0" * (-len(bits) % 8)
    rle = list(int(bits, 2).to_bytes(len(bits) // 8, "big"))

    assert decode_ls_rle(rle, 2, 1).tolist() == [[False, True]]


@pytest.mark.parametrize(
    "mask",
    [
        MASK_2X2,
        np.zeros((5, 7), dtype=bool),
        np.ones((5, 7), dtype=bool),
        _random_mask(40, 30),
        # Runs longer than 2**16 values have to be split
        np.pad(np.ones((300, 300), dtype=bool), ((10, 10), (5, 5))),
    ],
)
def test_rle_roundtrip(mask):
    height, width = mask.shape
    assert (decode_ls_rle(encode_ls_rle(mask), width, height) == mask).all()


def test_decode_wrong_size_raises():
    with pytest.raises(ValueError):
        decode_ls_rle(LS_RLE_2X2, 3, 2)


def test_brush_parsing_to_ir():
    annotation = generate_annotation({"format": "rle", "rle": LS_RLE_2X2, "brushlabels": ["dog"]}, "brushlabels", "abc")
    annotation["original_width"] = 2
    annotation["original_height"] = 2
    task = parse_ls_task(generate_task([annotation]))

    ls_ann = task.annotations[0].result[0]
    assert isinstance(ls_ann, BrushLabelsAnnotation)

    actual = task.to_ir_annotations()
    assert len(actual) == 1
    ann = actual[0]
    assert isinstance(ann, IRBinaryMaskImageAnnotation)
    assert ann.categories == {"dog": 1.0}
    assert ann.imported_id == "abc"
    assert (ann.to_mask() == MASK_2X2).all()


def test_ir_to_brush_roundtrip():
    mask = _random_mask(20, 30)
    ir = IRBinaryMaskImageAnnotation.from_mask(mask, categories={"dog": 1.0}, meta={"source": "manual"})

    task = LabelStudioTask()
    task.add_ir_annotation(ir)
    ls_ann = task.annotations[0].result[0]
    assert isinstance(ls_ann, BrushLabelsAnnotation)
    assert (ls_ann.original_width, ls_ann.original_height) == (30, 20)
    assert ls_ann.value.brushlabels == ["dog"]

    actual = parse_ls_task(task.model_dump_json()).to_ir_annotations()[0]
    assert isinstance(actual, IRBinaryMaskImageAnnotation)
    assert (actual.to_mask() == mask).all()
    assert actual.meta == {"source": "manual"}


def test_mask_normalization_does_not_change_mask():
    mask = _random_mask(4, 5)
    ir = IRBinaryMaskImageAnnotation.from_mask(mask, categories={"dog": 1.0})

    normalized = ir.normalized()
    assert normalized.coordinate_style == CoordinateStyle.NORMALIZED
    assert (normalized.to_mask() == mask).all()
    assert (normalized.denormalized().to_mask() == mask).all()


def test_mask_rle_starts_with_background():
    mask = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)
    ir = IRBinaryMaskImageAnnotation.from_mask(mask, categories={"dog": 1.0})
    assert ir.rle == [0, 2, 3, 1]
