import lxml.etree

from dagshub_annotation_converter.converters.cvat import _detect_cvat_mode


def test_image_elements_override_interpolation_metadata():
    root = lxml.etree.XML(
        b"""
        <annotations>
          <meta><task><mode>interpolation</mode></task></meta>
          <image id="0" name="frame.jpg" width="100" height="100" />
        </annotations>
        """
    )

    assert _detect_cvat_mode(root) == "image"
