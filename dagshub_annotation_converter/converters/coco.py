from pathlib import Path
from typing import Union, List, Dict, Tuple

from dagshub_annotation_converter.formats.coco.annotations import COCODataset, COCOAnnotation, COCOImage, COCOCategory
from dagshub_annotation_converter.formats.coco.importer import import_coco
from dagshub_annotation_converter.ir.image import IRBBoxImageAnnotation, CoordinateStyle, IRImageAnnotation

def coco_to_ir(coco_dataset: COCODataset) -> List[IRImageAnnotation]:
    """
    Converts a COCODataset object to a list of IRImageAnnotation objects.
    """
    ir_annotations: List[IRImageAnnotation] = []
    
    # Create a mapping from image_id to COCOImage object for quick lookup
    images_map: Dict[int, COCOImage] = {image.id: image for image in coco_dataset.images}
    
    # Create a mapping from category_id to COCOCategory object for quick lookup
    categories_map: Dict[int, COCOCategory] = {category.id: category for category in coco_dataset.categories}

    for ann in coco_dataset.annotations:
        image_info = images_map.get(ann.image_id)
        category_info = categories_map.get(ann.category_id)

        if image_info is None:
            # Log or handle missing image info
            print(f"Warning: Image with ID {ann.image_id} not found for annotation ID {ann.id}")
            continue
        
        if category_info is None:
            # Log or handle missing category info
            print(f"Warning: Category with ID {ann.category_id} not found for annotation ID {ann.id}")
            continue

        # COCO bbox is [x, y, width, height]
        # IR is top, left, width, height
        x, y, width, height = ann.bbox
        
        # Assuming the IR uses the actual pixel values (denormalized)
        # and the origin is top-left.
        ir_bbox = IRBBoxImageAnnotation(
            categories={category_info.name: 1.0}, # Assuming single category with confidence 1.0
            top=y,
            left=x,
            width=width,
            height=height,
            rotation=0.0, # COCO bbox format doesn't typically include rotation
            image_width=image_info.width,
            image_height=image_info.height,
            filename=image_info.file_name,
            coordinate_style=CoordinateStyle.DENORMALIZED,
            imported_id=str(ann.id)
        )
        ir_annotations.append(ir_bbox)
        
    return ir_annotations

def convert_coco_json_to_ir(file_path: Union[str, Path]) -> List[IRImageAnnotation]:
    """
    Loads a COCO JSON annotation file and converts it to a list of IRImageAnnotation objects.
    """
    coco_dataset = import_coco(file_path)
    return coco_to_ir(coco_dataset)
