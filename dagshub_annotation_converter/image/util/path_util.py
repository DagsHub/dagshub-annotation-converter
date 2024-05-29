from pathlib import Path


class YoloImageDirNotFoundError(Exception):
    def __init__(self, img_path: Path, image_dir_name: str):
        super().__init__()
        self.img_path = img_path
        self.image_dir_name = image_dir_name

    def __str__(self):
        return (
            f'Could not find the image dir "{self.image_dir_name}" in path "{self.img_path}".\n'
            f"Make sure your image_dir_name variable is configured correctly"
        )


def get_extension(path: Path) -> str:
    name = path.name
    ext_dot_index = name.rfind(".")
    ext = name[ext_dot_index:]
    return ext


def yolo_img_path_to_label_path(
    img_path: Path, image_dir_name: str, label_dir_name: str, label_extension: str = ".txt"
) -> Path:
    new_parts = list(img_path.parts)

    # Replace last occurrence of image_dir_name to label_dir_name
    for i, part in enumerate(reversed(img_path.parts)):
        if part == image_dir_name:
            new_parts[len(new_parts) - i - 1] = label_dir_name
            break
    else:
        raise YoloImageDirNotFoundError(img_path, image_dir_name)

    # Replace the extension
    new_parts[-1] = new_parts[-1].replace(get_extension(img_path), label_extension)
    return Path(*new_parts)
