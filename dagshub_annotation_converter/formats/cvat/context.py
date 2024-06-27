from lxml.etree import ElementBase
from pydantic import BaseModel

from dagshub_annotation_converter.ir.image import Categories


class CVATContext(BaseModel):
    categories: Categories = Categories()
    """List of categories"""


class CVATImageInfo(BaseModel):
    name: str
    width: int
    height: int


def parse_image_tag(image: ElementBase) -> CVATImageInfo:
    return CVATImageInfo(
        name=image.attrib["name"],
        width=int(image.attrib["width"]),
        height=int(image.attrib["height"]),
    )
