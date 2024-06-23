import logging
from os import PathLike
from typing import Sequence, Union
from zipfile import ZipFile

import lxml.etree

from dagshub_annotation_converter.refactored.formats.cvat import CVATContext, annotation_parsers
from dagshub_annotation_converter.refactored.ir.image import IRAnnotationBase


logger = logging.getLogger(__name__)


def parse_image_annotations(context: CVATContext, img: lxml.etree.ElementBase) -> Sequence[IRAnnotationBase]:
    annotations: list[IRAnnotationBase] = []
    for annotation_elem in img:
        annotation_type = annotation_elem.tag
        if annotation_type not in annotation_parsers:
            logger.warning(f"Unknown CVAT annotation type {annotation_type}")
            continue
        annotations.append(annotation_parsers[annotation_type](context, annotation_elem, img))

    return annotations


def parse_cvat_from_xml_string(xml_text: Union[str, bytes]) -> tuple[CVATContext, Sequence[IRAnnotationBase]]:
    context = CVATContext()
    annotations: list[IRAnnotationBase] = []
    root_elem = lxml.etree.parse(xml_text)

    for image_node in root_elem.xpath("//image"):
        annotations.extend(parse_image_annotations(context, image_node))

    return context, annotations


def parse_cvat_from_xml_file(xml_file: PathLike) -> tuple[CVATContext, Sequence[IRAnnotationBase]]:
    with open(xml_file) as f:
        return parse_cvat_from_xml_string(f.read())


def parse_cvat_from_zip(zip_path: PathLike) -> tuple[CVATContext, Sequence[IRAnnotationBase]]:
    with ZipFile(zip_path) as proj_zip:
        with proj_zip.open("annotations.xml") as f:
            return parse_cvat_from_xml_string(f.read())
