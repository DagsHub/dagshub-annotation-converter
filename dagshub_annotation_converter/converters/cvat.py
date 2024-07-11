import logging
from os import PathLike
from typing import Sequence
from zipfile import ZipFile

import lxml.etree

from dagshub_annotation_converter.formats.cvat import annotation_parsers
from dagshub_annotation_converter.formats.cvat.context import parse_image_tag
from dagshub_annotation_converter.ir.image import IRImageAnnotationBase


logger = logging.getLogger(__name__)


def parse_image_annotations(img: lxml.etree.ElementBase) -> Sequence[IRImageAnnotationBase]:
    annotations: list[IRImageAnnotationBase] = []
    for annotation_elem in img:
        annotation_type = annotation_elem.tag
        if annotation_type not in annotation_parsers:
            logger.warning(f"Unknown CVAT annotation type {annotation_type}")
            continue
        annotations.append(annotation_parsers[annotation_type](annotation_elem, img))

    return annotations


def load_cvat_from_xml_string(
    xml_text: bytes,
) -> dict[str, Sequence[IRImageAnnotationBase]]:
    annotations = {}
    root_elem = lxml.etree.XML(xml_text)

    for image_node in root_elem.xpath("//image"):
        image_info = parse_image_tag(image_node)
        annotations[image_info.name] = parse_image_annotations(image_node)

    return annotations


def load_cvat_from_xml_file(xml_file: PathLike) -> dict[str, Sequence[IRImageAnnotationBase]]:
    with open(xml_file, "rb") as f:
        return load_cvat_from_xml_string(f.read())


def load_cvat_from_zip(zip_path: PathLike) -> dict[str, Sequence[IRImageAnnotationBase]]:
    with ZipFile(zip_path) as proj_zip:
        with proj_zip.open("annotations.xml") as f:
            return load_cvat_from_xml_string(f.read())
