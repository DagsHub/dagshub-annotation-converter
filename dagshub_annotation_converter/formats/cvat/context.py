from typing import Dict, Any

from lxml.etree import ElementBase

from dagshub_annotation_converter.util.pydantic_util import ParentModel


class CVATImageInfo(ParentModel):
    name: str
    width: int
    height: int


def parse_image_tag(image: ElementBase) -> CVATImageInfo:
    return CVATImageInfo(
        name=image.attrib["name"],
        width=int(image.attrib["width"]),
        height=int(image.attrib["height"]),
    )


def parse_metadata(xml: ElementBase) -> Dict[str, Any]:
    res = {}
    group_id = xml.attrib.get("group_id")
    if group_id is not None:
        res["group_id"] = group_id
    return res
