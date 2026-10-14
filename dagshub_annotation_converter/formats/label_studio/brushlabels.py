from typing import List, Sequence

import numpy as np

from dagshub_annotation_converter.formats.label_studio.base import ImageAnnotationResultABC
from dagshub_annotation_converter.ir.base import IRAnnotationBase
from dagshub_annotation_converter.ir.image import CoordinateStyle, IRBinaryMaskImageAnnotation
from dagshub_annotation_converter.util.pydantic_util import ParentModel

# Label Studio stores a brush mask as an RGBA image of the full image size, packed with the "rle-pack" bit format.
# The label color is not stored. Label Studio sets R = G = B = A, where A is the mask opacity (0-255).
# Edges drawn in the Label Studio editor are anti-aliased, so A can be between 0 and 255.
# We convert to a binary mask, where every pixel with A > 0 is in the mask.
#
# Layout of the packed bits:
#
#   header: 32 bits value count | 5 bits (word size - 1) | 4 x 4 bits (run size - 1)
#   then blocks until all values are read:
#     1 | 2 bits run size index | N bits (run length - 1) | word        -> one value, repeated
#     0 | 2 bits run size index | N bits (block length - 1) | words... -> literal values
#
# N is the run size that the index selects. We write with word size 8 and run sizes [3, 4, 8, 16],
# the same as Label Studio.

_WORD_SIZE = 8
_RUN_SIZES = [3, 4, 8, 16]
_CHANNELS = 4
_FOREGROUND = 255


class BrushLabelsAnnotationValue(ParentModel):
    format: str = "rle"
    rle: List[int]
    brushlabels: List[str]


class BrushLabelsAnnotation(ImageAnnotationResultABC):
    value: BrushLabelsAnnotationValue
    type: str = "brushlabels"

    def to_ir_annotation(self) -> Sequence[IRBinaryMaskImageAnnotation]:
        if self.value.format != "rle":
            raise ValueError(f"Unsupported brush format {self.value.format}, only 'rle' is supported")
        if not self.value.brushlabels:
            raise ValueError(f"Brush annotation {self.id} has no label")
        mask = decode_ls_rle(self.value.rle, self.original_width, self.original_height)
        res = IRBinaryMaskImageAnnotation.from_mask(
            mask,
            categories={self.value.brushlabels[0]: 1.0},
            coordinate_style=CoordinateStyle.NORMALIZED,
            meta=self.meta or {},
        )
        res.imported_id = self.id
        return [res]

    @staticmethod
    def from_ir_annotation(ir_annotation: IRAnnotationBase) -> Sequence["BrushLabelsAnnotation"]:
        assert isinstance(ir_annotation, IRBinaryMaskImageAnnotation)

        category = ir_annotation.ensure_has_one_category()

        return [
            BrushLabelsAnnotation(
                original_width=ir_annotation.image_width,
                original_height=ir_annotation.image_height,
                value=BrushLabelsAnnotationValue(
                    rle=encode_ls_rle(ir_annotation.to_mask()),
                    brushlabels=[category],
                ),
                meta=ir_annotation.meta or None,
            )
        ]


def encode_ls_rle(mask: np.ndarray) -> List[int]:
    """Encodes a 2D mask to Label Studio brush RLE. Every non-zero value is foreground."""
    flat = np.repeat((mask.ravel() != 0).astype(np.uint8) * _FOREGROUND, _CHANNELS)
    num_values = flat.size

    bits: List[str] = [f"{num_values:032b}", f"{_WORD_SIZE - 1:05b}"]
    bits.extend(f"{size - 1:04b}" for size in _RUN_SIZES)

    max_run = 2 ** _RUN_SIZES[-1]
    for value, length in _runs(flat):
        while length > 0:
            chunk = min(length, max_run)
            length -= chunk
            if chunk == 1:
                # Literal block (0) with run size index 0 and a length of one value
                bits.append(f"0{0:02b}{0:0{_RUN_SIZES[0]}b}{value:0{_WORD_SIZE}b}")
                continue
            size_index = next(i for i, size in enumerate(_RUN_SIZES) if chunk <= 2**size)
            size = _RUN_SIZES[size_index]
            bits.append(f"1{size_index:02b}{chunk - 1:0{size}b}{value:0{_WORD_SIZE}b}")

    bit_string = "".join(bits)
    bit_string += "0" * (-len(bit_string) % 8)
    return list(int(bit_string, 2).to_bytes(len(bit_string) // 8, "big"))


def decode_ls_rle(rle: Sequence[int], width: int, height: int) -> np.ndarray:
    """Decodes Label Studio brush RLE to a boolean mask with the shape (height, width)."""
    reader = _BitReader(bytes(rle))
    num_values = reader.read(32)
    word_size = reader.read(5) + 1
    run_sizes = [reader.read(4) + 1 for _ in range(4)]

    if num_values != width * height * _CHANNELS:
        raise ValueError(
            f"Brush RLE has {num_values} values, but a {width}x{height} RGBA image needs {width * height * _CHANNELS}"
        )

    out = np.zeros(num_values, dtype=np.uint32)
    i = 0
    while i < num_values:
        is_run = reader.read(1)
        end = i + 1 + reader.read(run_sizes[reader.read(2)])
        if is_run:
            out[i:end] = reader.read(word_size)
            i = end
        else:
            while i < end:
                out[i] = reader.read(word_size)
                i += 1

    # Label Studio uses the alpha channel as the mask
    return (out[_CHANNELS - 1 :: _CHANNELS] > 0).reshape(height, width)


def _runs(flat: np.ndarray):
    """Yields (value, length) for every run of equal values."""
    if flat.size == 0:
        return
    change_points = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [flat.size]))
    for start, end in zip(starts.tolist(), ends.tolist()):
        yield int(flat[start]), end - start


class _BitReader:
    def __init__(self, data: bytes):
        self._bits = "".join(f"{byte:08b}" for byte in data)
        self._position = 0

    def read(self, size: int) -> int:
        end = self._position + size
        if end > len(self._bits):
            raise ValueError("Brush RLE ended before all values were read")
        value = int(self._bits[self._position : end], 2)
        self._position = end
        return value
