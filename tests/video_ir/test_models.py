from dagshub_annotation_converter.ir.video import (
    IRVideoBBoxAnnotation,
    CoordinateStyle,
)


class TestIRVideoBBoxAnnotation:
    def test_create_basic_annotation(self):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
        )

        assert ann.track_id == 1
        assert ann.frame_number == 0
        assert ann.left == 100
        assert ann.top == 150
        assert ann.width == 50
        assert ann.height == 120
        assert ann.image_width == 1920
        assert ann.image_height == 1080
        assert ann.categories == {"person": 1.0}

    def test_annotation_with_optional_fields(self):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=5,
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"car": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
            timestamp=0.167,
            confidence=0.95,
            visibility=0.8,
            video_path="/videos/test.mp4",
        )

        assert ann.timestamp == 0.167
        assert ann.confidence == 0.95
        assert ann.visibility == 0.8
        assert ann.video_path == "/videos/test.mp4"

    def test_normalize_coordinates(self):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=192,  # 192/1920 = 0.1
            top=108,   # 108/1080 = 0.1
            width=384,  # 384/1920 = 0.2
            height=216,  # 216/1080 = 0.2
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
        )

        normalized = ann.normalized()

        assert normalized.coordinate_style == CoordinateStyle.NORMALIZED
        assert abs(normalized.left - 0.1) < 1e-6
        assert abs(normalized.top - 0.1) < 1e-6
        assert abs(normalized.width - 0.2) < 1e-6
        assert abs(normalized.height - 0.2) < 1e-6
        assert normalized.track_id == 1
        assert normalized.frame_number == 0

    def test_denormalize_coordinates(self):
        ann = IRVideoBBoxAnnotation(
            track_id=2,
            frame_number=10,
            left=0.1,
            top=0.1,
            width=0.2,
            height=0.2,
            image_width=1920,
            image_height=1080,
            categories={"car": 1.0},
            coordinate_style=CoordinateStyle.NORMALIZED,
        )

        denormalized = ann.denormalized()

        assert denormalized.coordinate_style == CoordinateStyle.DENORMALIZED
        assert abs(denormalized.left - 192) < 1e-6
        assert abs(denormalized.top - 108) < 1e-6
        assert abs(denormalized.width - 384) < 1e-6
        assert abs(denormalized.height - 216) < 1e-6
        assert denormalized.track_id == 2
        assert denormalized.frame_number == 10

    def test_with_filename(self):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
        )

        ann_with_filename = ann.with_filename("frame_0001.jpg")
        assert ann_with_filename.filename == "frame_0001.jpg"

    def test_ensure_has_one_category(self):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
        )

        category = ann.ensure_has_one_category()
        assert category == "person"


class TestIRVideoAnnotationBase:
    def test_inherits_from_image_annotation(self):
        ann = IRVideoBBoxAnnotation(
            track_id=1,
            frame_number=0,
            left=100,
            top=150,
            width=50,
            height=120,
            image_width=1920,
            image_height=1080,
            categories={"person": 1.0},
            coordinate_style=CoordinateStyle.DENORMALIZED,
        )

        assert hasattr(ann, "image_width")
        assert hasattr(ann, "image_height")

        assert hasattr(ann, "filename")
        assert hasattr(ann, "categories")
        assert hasattr(ann, "coordinate_style")
        assert hasattr(ann, "meta")

        assert hasattr(ann, "track_id")
        assert hasattr(ann, "frame_number")
        assert hasattr(ann, "timestamp")
        assert hasattr(ann, "video_path")
