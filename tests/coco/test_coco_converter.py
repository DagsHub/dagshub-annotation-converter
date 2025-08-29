import pytest
from pathlib import Path
from dagshub_annotation_converter.converters.coco import convert_coco_json_to_ir
from dagshub_annotation_converter.ir.image import IRBBoxImageAnnotation, CoordinateStyle

# Get the directory of the current test file
TEST_DIR = Path(__file__).parent

def test_convert_sample_coco_to_ir():
    sample_file = TEST_DIR / "sample_coco_annotations.json"
    ir_annotations = convert_coco_json_to_ir(sample_file)

    assert len(ir_annotations) == 1
    
    ann = ir_annotations[0]
    assert isinstance(ann, IRBBoxImageAnnotation)
    
    # Verify categories (name comes from categories map, confidence is default 1.0)
    assert "cat" in ann.categories
    assert ann.categories["cat"] == 1.0
    
    # Verify bounding box coordinates (COCO: [x, y, width, height])
    # IR: top=y, left=x, width=width, height=height
    assert ann.left == 100
    assert ann.top == 100
    assert ann.width == 200
    assert ann.height == 300
    
    assert ann.rotation == 0.0 # Default, as COCO bbox has no rotation
    
    # Verify image information
    assert ann.image_width == 800
    assert ann.image_height == 600
    assert ann.filename == "test_image.jpg"
    
    assert ann.coordinate_style == CoordinateStyle.DENORMALIZED
    
    # Verify imported ID (should be string representation of annotation ID)
    assert ann.imported_id == "1"
