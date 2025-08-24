from typing import List, Dict, Any
from pydantic import BaseModel

class COCOImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: int = 0
    flickr_url: str = ""
    coco_url: str = ""
    date_captured: str = ""

class COCOCategory(BaseModel):
    id: int
    name: str
    supercategory: str = ""

class COCOAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: List[List[float]] = [] # For polygons or RLE
    area: float
    bbox: List[float] # [x, y, width, height]
    iscrowd: int # 0 or 1

class COCODataset(BaseModel):
    info: Dict[str, Any] = {}
    licenses: List[Dict[str, Any]] = []
    images: List[COCOImage]
    annotations: List[COCOAnnotation]
    categories: List[COCOCategory]
