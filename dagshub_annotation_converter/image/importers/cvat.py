import logging
from os import PathLike
from pathlib import Path
from typing import Union, Dict, Callable, List, Tuple
from zipfile import ZipFile

import lxml.etree
import lxml

from dagshub_annotation_converter.image.ir.annotation_ir import (
    AnnotationProject,
    AnnotatedFile,
    BBoxAnnotation,
    NormalizationState,
    AnnotationABC,
    SegmentationAnnotation,
    PoseAnnotation,
    KeyPoint,
)

logger = logging.getLogger(__name__)


class CvatImageImporter:
    def __init__(self, project_file: Union[str, PathLike]):
        self.project_file = Path(project_file)
        if not self.project_file.name.endswith(".zip"):
            raise RuntimeError("CVAT Project file must be a .zip file")

    def parse(self) -> AnnotationProject:
        project = AnnotationProject()
        self._parse_project(project)
        return project

    def _parse_project(self, project: AnnotationProject):
        proj_xml = self._get_annotations()
        for image_node in proj_xml.xpath("//image"):
            project.files.append(self._parse_image_annotations(project, image_node))
        print(project)

    def _parse_image_annotations(self, project: AnnotationProject, img: lxml.etree.ElementBase) -> AnnotatedFile:
        res = AnnotatedFile(
            file=str(img.attrib["name"]), image_width=int(img.attrib["width"]), image_height=int(img.attrib["height"])
        )
        for annotation_elem in img:
            annotation_type = annotation_elem.tag
            if annotation_type not in self._annotation_parsers:
                logger.warning(f"Unknown CVAT annotation type {annotation_type}")
                continue
            res.annotations.append(self._annotation_parsers[annotation_type](project, annotation_elem))

        return res

    @staticmethod
    def _parse_box(project: AnnotationProject, elem: lxml.etree.ElementBase) -> AnnotationABC:
        top = float(elem.attrib["ytl"])
        bottom = float(elem.attrib["ybr"])
        left = float(elem.attrib["xtl"])
        right = float(elem.attrib["xbr"])

        width = right - left
        height = bottom - top

        category = project.categories.get_or_create(str(elem.attrib["label"]))

        return BBoxAnnotation(
            top=top,
            left=left,
            width=width,
            height=height,
            category=category,
            state=NormalizationState.DENORMALIZED,
        )

    @staticmethod
    def _parse_polygon(project: AnnotationProject, elem: lxml.etree.ElementBase) -> AnnotationABC:
        category = project.categories.get_or_create(str(elem.attrib["label"]))
        res = SegmentationAnnotation(category=category, state=NormalizationState.DENORMALIZED)
        for point_str in elem.attrib["points"].split(";"):
            x, y = point_str.split(",")
            res.add_point(x=float(x), y=float(y))

        return res

    @staticmethod
    def _parse_points(project: AnnotationProject, elem: lxml.etree.ElementBase) -> AnnotationABC:
        points: List[KeyPoint] = []

        category = project.categories.get_or_create(str(elem.attrib["label"]))

        for point_str in elem.attrib["points"].split(";"):
            x, y = point_str.split(",")
            points.append(KeyPoint(x=float(x), y=float(y)))

        return PoseAnnotation.from_points(category=category, points=points, state=NormalizationState.DENORMALIZED)

    @staticmethod
    def _parse_skeleton(project: AnnotationProject, elem: lxml.etree.ElementBase) -> AnnotationABC:
        # Points also contain the labels, for consistent ordering in LS, they are later sorted
        points: List[Tuple[str, KeyPoint]] = []

        category = project.categories.get_or_create(str(elem.attrib["label"]))

        for point_elem in elem:
            x, y = point_elem.attrib["points"].split(",")
            points.append(
                (
                    elem.attrib["label"],
                    KeyPoint(x=float(x), y=float(y), is_visible=point_elem.attrib["occluded"] == "0"),
                )
            )

        all_labels_ints = all(map(lambda tup: tup[0].isdigit(), points))

        # sort points by the label
        if all_labels_ints:
            points = sorted(points, key=lambda tup: int(tup[0]))
        else:
            points = sorted(points, key=lambda tup: tup[0])

        res_points = list(map(lambda tup: tup[1], points))

        return PoseAnnotation.from_points(category=category, points=res_points, state=NormalizationState.DENORMALIZED)

    def _get_annotations(self) -> lxml.etree.ElementBase:
        with ZipFile(self.project_file) as proj_zip:
            with proj_zip.open("annotations.xml") as f:
                return lxml.etree.parse(f)

    _annotation_parsers: Dict[str, Callable[[AnnotationProject, lxml.etree.ElementBase], AnnotationABC]] = {
        "box": _parse_box,
        "polygon": _parse_polygon,
        "points": _parse_points,
        "skeleton": _parse_skeleton,
    }


if __name__ == "__main__":
    importer = CvatImageImporter(
        "/Users/kirillbolashev/Job/GitRepos/annotation-converter/dagshub_annotation_converter/scratches/cvat/baby-yoda.zip"
    )
    importer.parse()
