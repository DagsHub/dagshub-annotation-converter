from pathlib import Path

_supported_image_formats = None


def supported_image_formats() -> set[str]:
    global _supported_image_formats
    if _supported_image_formats is None:
        from PIL import Image

        exts = Image.registered_extensions()
        supported = {ex for ex, f in exts.items() if f in Image.OPEN}
        _supported_image_formats = supported
    return _supported_image_formats


def get_extension(path: Path) -> str:
    name = path.name
    ext_dot_index = name.rfind(".")
    ext = name[ext_dot_index:]
    return ext


def is_image(path: Path) -> bool:
    return get_extension(path) in supported_image_formats()
